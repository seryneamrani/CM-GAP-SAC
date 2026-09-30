"""
trajectory_predictor.py -- pont entre le pipeline perception et SocialLSTMLite.

Maintient un buffer glissant des OBS_LEN dernieres positions MONDE de chaque
track (keye par track_id), empile tous les pietons presents, appelle le
Social-LSTM Lite, et retourne les positions futures absolues MONDE par track.

Contraintes issues de models/social_lstm_lite.py (NE PAS deroger) :
  - Le modele prend (obs_len, N, 2) en positions ABSOLUES MONDE, en metres.
    PAS de vitesse, PAS de normalisation (le forward fait cur_pos += offset).
  - Le social pooling est relatif entre pietons -> il faut passer TOUS les
    pietons simultanes ensemble, dans le meme repere monde.
  - obs_len = 8 (defaut entrainement). Un track avec < 8 positions ne peut
    pas etre predit par le LSTM -> fallback lineaire (pos + v*dt*k).

Le dt : le modele a ete entraine sur ETH/UCY (dt=0.4s). offset = deplacement
par pas a CE dt. Tant que tu n'as pas fine-tune sur des trajectoires Gazebo
a TON dt, l'echelle des offsets sera celle d'ETH/UCY. Le fine-tune (train.py
--pretrained best_eth_ucy.pt sur data/gazebo) corrige ca. En attendant, traite
l'horizon comme "H pas de 0.4s" et non "H pas de ton control_hz".

Ce module est PUR (pas de ROS). L'env l'alimente avec les tracks monde qu'il
a deja (apres conversion robot->world), et l'interroge a chaque step.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _History:
    """Buffer glissant des positions MONDE d'un track + derniere vitesse."""
    positions: deque           # deque[(x, y)] maxlen=obs_len
    last_velocity: np.ndarray  # (2,) monde, pour le fallback lineaire
    last_age: int = 0

    def __post_init__(self):
        if not isinstance(self.last_velocity, np.ndarray):
            self.last_velocity = np.asarray(self.last_velocity, dtype=np.float32)


class TrajectoryPredictor:
    """Buffer per-track + appel SocialLSTMLite + fallback lineaire.

    Usage (dans l'env, par step) :
        predictor.update(tracks_world)              # alimente les buffers
        futures = predictor.predict(horizon=4)      # {track_id: (H,2) monde}

    tracks_world : liste d'objets ayant .track_id (int), .position_world (2,),
                   .velocity_world (2,), .age_frames (int). C'est exactement
                   ton PedestrianTrack (track_state.py).
    """

    def __init__(
        self,
        model=None,                 # SocialLSTMLite charge, ou None (fallback only)
        obs_len: int = 8,
        device: str = "cpu",
        stale_after_missing: int = 1,  # purge un track absent depuis N updates
    ):
        self.model = model
        self.obs_len = int(obs_len)
        self.device = device
        self._hist: Dict[int, _History] = {}
        self._missing: Dict[int, int] = {}
        self._stale_after = int(stale_after_missing)

        if model is not None:
            try:
                import torch  # noqa
                self._torch = torch
                model.eval()
            except ImportError:
                self._torch = None
                self.model = None
        else:
            self._torch = None

    # ------------------------------------------------------------------
    def update(self, tracks_world) -> None:
        """Pousse les positions monde courantes dans les buffers par id."""
        seen = set()
        for t in tracks_world:
            tid = int(t.track_id)
            seen.add(tid)
            pos = np.asarray(t.position_world, dtype=np.float32)
            vel = np.asarray(t.velocity_world, dtype=np.float32)

            # Reset du buffer si l'age repart a 1 (id DeepSORT reutilise) :
            # un trou de tracking pollue l'historique social. age_frames==1
            # signale un track tout neuf.
            if tid in self._hist and getattr(t, "age_frames", 99) <= 1:
                self._hist.pop(tid, None)

            if tid not in self._hist:
                self._hist[tid] = _History(
                    positions=deque(maxlen=self.obs_len),
                    last_velocity=vel,
                )
            h = self._hist[tid]
            h.positions.append((float(pos[0]), float(pos[1])))
            h.last_velocity = vel
            h.last_age = getattr(t, "age_frames", h.last_age + 1)
            self._missing[tid] = 0

        # Purge des tracks non vus (evite des buffers fantomes).
        for tid in list(self._hist.keys()):
            if tid not in seen:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self._stale_after:
                    self._hist.pop(tid, None)
                    self._missing.pop(tid, None)

    # ------------------------------------------------------------------
    def predict(self, horizon: int, dt: float = 0.4) -> Dict[int, np.ndarray]:
        """Retourne {track_id: (horizon, 2)} positions futures MONDE.

        Strategie :
          - tracks avec buffer plein (== obs_len) -> passes ENSEMBLE au LSTM
            (social pooling correct).
          - tracks avec buffer incomplet -> fallback lineaire individuel.
          - si pas de modele -> tout en fallback lineaire.

        dt : utilise UNIQUEMENT par le fallback lineaire (pos + v*dt*k).
             Le LSTM, lui, predit a son propre pas (cf. note dt du module).
        """
        out: Dict[int, np.ndarray] = {}

        full_ids = [tid for tid, h in self._hist.items()
                    if len(h.positions) == self.obs_len]
        partial_ids = [tid for tid in self._hist if tid not in full_ids]

        # 1. LSTM sur tous les tracks "pleins" ensemble.
        if self.model is not None and self._torch is not None and full_ids:
            torch = self._torch
            # (obs_len, N, 2) en monde absolu.
            seq = np.stack(
                [np.asarray(self._hist[tid].positions, dtype=np.float32)
                 for tid in full_ids], axis=1
            )  # (obs_len, N, 2)
            with torch.no_grad():
                obs = torch.from_numpy(seq).float().to(self.device)
                pred = self.model(obs, pred_len=horizon)  # (H, N, 2)
                pred = pred.cpu().numpy()
            for k, tid in enumerate(full_ids):
                out[tid] = pred[:, k, :].astype(np.float32)  # (H, 2)
        else:
            # Pas de modele : tout le monde en fallback.
            partial_ids = list(self._hist.keys())

        # 2. Fallback lineaire pour les buffers incomplets.
        for tid in partial_ids:
            h = self._hist[tid]
            if not h.positions:
                continue
            p0 = np.asarray(h.positions[-1], dtype=np.float32)
            v = h.last_velocity
            steps = np.arange(1, horizon + 1, dtype=np.float32)[:, None]  # (H,1)
            out[tid] = (p0[None, :] + v[None, :] * dt * steps).astype(np.float32)

        return out

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Vide tous les buffers (entre episodes)."""
        self._hist.clear()
        self._missing.clear()


# ----------------------------------------------------------------------
# Self-test : verifie buffer, fallback, et (si torch+modele) le LSTM.
# ----------------------------------------------------------------------
def _self_test() -> None:
    print("[trajectory_predictor] self-test")

    @dataclass
    class FakeTrack:
        track_id: int
        position_world: np.ndarray
        velocity_world: np.ndarray
        age_frames: int

    # --- Test 1 : fallback lineaire (pas de modele) ---
    pred = TrajectoryPredictor(model=None, obs_len=8)
    # un track qui avance en x a 1 m/s
    for step in range(3):  # buffer incomplet (< 8)
        pred.update([FakeTrack(
            track_id=1,
            position_world=np.array([step * 0.4, 0.0], np.float32),
            velocity_world=np.array([1.0, 0.0], np.float32),
            age_frames=step + 1,
        )])
    fut = pred.predict(horizon=4, dt=0.4)
    print(f"  fallback id1 future[0..3] x = {fut[1][:, 0]}  (attendu +0.4 par pas)")
    assert fut[1].shape == (4, 2)
    # depart x=0.8 (dernier), +1.0*0.4 par pas -> 1.2, 1.6, 2.0, 2.4
    assert abs(fut[1][0, 0] - 1.2) < 1e-4, fut[1][0, 0]
    assert abs(fut[1][-1, 0] - 2.4) < 1e-4, fut[1][-1, 0]
    print("  fallback lineaire OK")

    # --- Test 2 : reset id (age repart a 1) purge le buffer ---
    pred.update([FakeTrack(1, np.array([99., 99.], np.float32),
                           np.array([0., 0.], np.float32), age_frames=1)])
    assert len(pred._hist[1].positions) == 1, "buffer pas reset sur age=1"
    print("  reset-on-age=1 OK")

    # --- Test 3 : LSTM si dispo ---
    try:
        import torch  # noqa
        # Import du modele : en-paquet une fois copie dans perception/.
        # Fallback sur un chemin local pour test standalone hors paquet.
        try:
            from cm_gap_sac_navigation.perception.social_lstm_lite import (
                SocialLSTMLite,
            )
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from social_lstm_lite import SocialLSTMLite
        model = SocialLSTMLite(embedding_dim=16, hidden_size=32,
                               grid_size=4, neighborhood=2.0)
        model.eval()
        p2 = TrajectoryPredictor(model=model, obs_len=8, device="cpu")
        # 2 pietons, buffer plein
        for step in range(8):
            p2.update([
                FakeTrack(10, np.array([step*0.3, 0.0], np.float32),
                          np.array([0.75, 0.], np.float32), age_frames=step+1),
                FakeTrack(11, np.array([step*0.3, 1.0], np.float32),
                          np.array([0.75, 0.], np.float32), age_frames=step+1),
            ])
        fut = p2.predict(horizon=4)
        print(f"  LSTM ids={list(fut.keys())} shapes={[v.shape for v in fut.values()]}")
        assert fut[10].shape == (4, 2) and fut[11].shape == (4, 2)
        print("  LSTM predict OK (2 pietons ensemble, social pooling actif)")
    except ImportError:
        print("  (torch/modele indispo ici -> LSTM teste cote robot)")

    print("[trajectory_predictor] self-test passed.")


if __name__ == "__main__":
    _self_test()