"""Pedestrian state assembly.

Converts a list of pedestrian tracks (positions in world frame, velocities,
ages) into the fixed-size tensor consumed by the encoders.

Output shape contract:
    features: (K, 5)  with columns [x_rel, y_rel, vx_rel, vy_rel, age_norm]
    mask:     (K,)    1 if slot is occupied, 0 if padded

The mask is critical: the cross-attention layer uses it to zero out the
softmax weights of empty slots. Padding alone is not enough.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from cm_gap_sac_navigation.utils.geometry import (
    velocity_world_to_robot,
    world_to_robot,
)


@dataclass
class PedestrianTrack:
    """A single pedestrian track in WORLD frame.

    The tracker (YOLO+DeepSORT or Gazebo GT) produces these. The env then
    transforms them into the robot frame via `assemble_pedestrian_state`.
    """
    track_id: int
    position_world: np.ndarray   # shape (2,), [x, y] in world frame
    velocity_world: np.ndarray   # shape (2,), [vx, vy] in world frame
    age_frames: int              # number of frames the track has lived

    def __post_init__(self) -> None:
        self.position_world = np.asarray(self.position_world, dtype=np.float32)
        self.velocity_world = np.asarray(self.velocity_world, dtype=np.float32)
        if self.position_world.shape != (2,):
            raise ValueError(f"position_world must be (2,), got {self.position_world.shape}")
        if self.velocity_world.shape != (2,):
            raise ValueError(f"velocity_world must be (2,), got {self.velocity_world.shape}")


def assemble_pedestrian_state(
    tracks: List[PedestrianTrack],
    robot_xy: np.ndarray,
    robot_yaw: float,
    k_max: int,
    relevance_radius: float,
    track_age_norm: int,
    futures: dict = None,        # NEW: {track_id: (H,2) positions MONDE}
    prediction_horizon: int = 0, # NEW: H. 0 => comportement actuel exact.
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert world-frame tracks into the (K, 5 + 2H) network input.

    Base columns (unchanged): [x_rel, y_rel, vx_rel, vy_rel, age_norm].
    Future columns (NEW, when prediction_horizon>0): for each of the H
    predicted steps, the predicted pedestrian position in the ROBOT frame
    [fx_1, fy_1, ..., fx_H, fy_H].

    futures: dict track_id -> (H, 2) predicted positions in WORLD frame
             (as produced by TrajectoryPredictor.predict). If a selected
             track is missing from this dict (incomplete buffer, no model),
             its future columns are filled by REPEATING its current world
             position -> in robot frame this means "stays where it is",
             a neutral assumption that does not perturb the policy.

    Selection rule (unchanged): keep the k_max nearest tracks within
    relevance_radius.

    Args:
        tracks: list of tracks in world frame
        robot_xy: shape (2,)
        robot_yaw: scalar
        k_max: maximum number of tracks to retain
        relevance_radius: drop tracks farther than this (m)
        track_age_norm: clamp+normalize age by this many frames
        futures: optional dict track_id -> (H, 2) world-frame predicted positions
        prediction_horizon: H, number of future steps to encode (0 = legacy shape)

    Returns:
        features: (k_max, 5 + 2*H) float32, padded with zeros for empty slots
        mask:     (k_max,)         uint8,  1 = occupied, 0 = padded
    """
    n_base = 5
    H = int(prediction_horizon)
    n_features = n_base + 2 * H
    futures = futures or {}

    if len(tracks) == 0:
        return (
            np.zeros((k_max, n_features), dtype=np.float32),
            np.zeros((k_max,), dtype=np.uint8),
        )

    # 1. Compute distances in world frame (cheaper than transforming first).
    positions = np.stack([t.position_world for t in tracks], axis=0)   # (N, 2)
    deltas = positions - robot_xy[None, :]
    dists = np.linalg.norm(deltas, axis=1)

    # 2. Filter by relevance radius.
    valid = dists < relevance_radius
    if not np.any(valid):
        return (
            np.zeros((k_max, n_features), dtype=np.float32),
            np.zeros((k_max,), dtype=np.uint8),
        )
    idx_valid = np.where(valid)[0]
    dists_valid = dists[idx_valid]

    # 3. Sort ascending and keep up to k_max nearest.
    order = np.argsort(dists_valid)[:k_max]
    selected = idx_valid[order]
    n_kept = selected.size

    # 4. Transform position and velocity into robot frame.
    pos_w = positions[selected]                                       # (n_kept, 2)
    vel_w = np.stack([tracks[i].velocity_world for i in selected])    # (n_kept, 2)
    pos_r = world_to_robot(pos_w, robot_xy, robot_yaw)
    vel_r = velocity_world_to_robot(vel_w, robot_yaw)

    # 5. Normalize age in [0, 1]; clamp at track_age_norm.
    ages = np.array(
        [min(tracks[i].age_frames, track_age_norm) for i in selected],
        dtype=np.float32,
    ) / float(track_age_norm)

    # 6. Assemble fixed-size tensor + mask.
    features = np.zeros((k_max, n_features), dtype=np.float32)
    features[:n_kept, 0] = pos_r[:, 0]
    features[:n_kept, 1] = pos_r[:, 1]
    features[:n_kept, 2] = vel_r[:, 0]
    features[:n_kept, 3] = vel_r[:, 1]
    features[:n_kept, 4] = ages

    # --- Future columns (Phase C) -------------------------------------
    if H > 0:
        for slot, i in enumerate(selected):
            tid = tracks[i].track_id
            fut_w = futures.get(tid, None)
            if fut_w is None or len(fut_w) < H:
                # Fallback : repeter la position courante (reste sur place).
                fut_w = np.repeat(
                    tracks[i].position_world[None, :], H, axis=0
                )
            else:
                fut_w = np.asarray(fut_w[:H], dtype=np.float32)
            # Monde -> robot, homogene avec x_rel,y_rel.
            fut_r = world_to_robot(fut_w, robot_xy, robot_yaw)  # (H,2)
            for k in range(H):
                features[slot, n_base + 2 * k]     = fut_r[k, 0]
                features[slot, n_base + 2 * k + 1] = fut_r[k, 1]

    mask = np.zeros((k_max,), dtype=np.uint8)
    mask[:n_kept] = 1

    return features, mask