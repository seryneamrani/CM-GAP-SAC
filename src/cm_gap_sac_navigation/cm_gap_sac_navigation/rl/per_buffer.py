"""Prioritized Experience Replay buffer with attention-entropy weighting.

This is originality 2 of CM-GAP_SAC: priority of a transition combines the
classical TD-error term (Schaul et al. 2016) with a multiplicative factor
based on the pedestrian attention entropy of that transition.

Priority formula:
    p_i = (|δ_i| + ε)^η · (1 + λ · (1 - Ĥ_i))^ν

where:
    |δ_i|   is the TD-error magnitude
    Ĥ_i     is the normalized pedestrian attention entropy in [0, 1]
    η       is the standard PER exponent (default 0.6)
    ν       is the entropy exponent (default 0.5)
    λ       is the entropy amplification (default 3.0)
    ε       small positive constant to avoid zero priorities

When ν=0 or λ=0, the formula reduces exactly to standard PER.

Importance sampling weights:
    w_i = (1 / (N · P(i)))^β     normalized by 1 / max_j w_j

with β annealed linearly from beta_start (0.4) to beta_end (1.0) over
beta_anneal_steps to correct the bias introduced by non-uniform sampling.

Storage:
    The buffer stores Dict observations (lidar/pedestrians/ped_mask/imu/goal).
    Each modality is held in a separate ring-buffer NumPy array, indexed by
    the same write pointer. This avoids per-transition pickling and is
    essentially zero-copy on retrieval.

Reference:
    Schaul, Quan, Antonoglou, Silver. "Prioritized Experience Replay" (2016).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ======================================================================
# SumTree: efficient O(log N) sampling proportional to priority.
# ======================================================================
class SumTree:
    """Binary tree where each leaf holds a priority and each internal node
    holds the sum of its children. Sampling a uniform value in [0, total]
    and traversing from root yields a leaf with probability proportional
    to its priority.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        # Tree as a flat array. Internal nodes at [0, capacity-1),
        # leaves at [capacity-1, 2*capacity-1).
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._max_priority = 1.0   # for newly added transitions

    # ------------------------------------------------------------------
    def update(self, leaf_idx: int, priority: float) -> None:
        """Set the priority of leaf `leaf_idx` (0-indexed) and propagate."""
        if not (0 <= leaf_idx < self.capacity):
            raise IndexError(f"leaf_idx {leaf_idx} out of range [0, {self.capacity})")
        tree_idx = leaf_idx + self.capacity - 1
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagate up to the root.
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change
        # Track max for new transitions (start with high priority).
        if priority > self._max_priority:
            self._max_priority = priority

    # ------------------------------------------------------------------
    def get(self, value: float) -> Tuple[int, float]:
        """Find the leaf whose cumulative priority covers `value`.

        Returns (leaf_idx, priority).
        """
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self._tree[idx])

    # ------------------------------------------------------------------
    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        return float(self._max_priority)


# ======================================================================
# Storage layout for one transition (Dict obs).
# ======================================================================
@dataclass
class _TransitionShapes:
    n_beams: int
    k_max: int
    ped_features: int
    imu_features: int
    goal_features: int
    action_dim: int


# ======================================================================
# PER buffer
# ======================================================================
class PrioritizedReplayBuffer:
    """Ring buffer with SumTree-based priority sampling for Dict observations.

    Memory layout: each modality has a (capacity, ...) NumPy array. Writes
    and reads use the same integer pointer.
    """

    def __init__(
        self,
        capacity: int,
        n_beams: int,
        k_max: int,
        ped_features: int,
        imu_features: int,
        goal_features: int,
        action_dim: int,
        eta: float = 0.6,
        nu: float = 0.5,
        lam: float = 3.0,
        epsilon: float = 1e-6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 1_000_000,
        use_attention_entropy: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.eta = float(eta)
        self.nu = float(nu)
        self.lam = float(lam)
        self.epsilon = float(epsilon)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_anneal_steps = int(beta_anneal_steps)
        self.use_attention_entropy = bool(use_attention_entropy)

        self.shapes = _TransitionShapes(
            n_beams=n_beams, k_max=k_max,
            ped_features=ped_features, imu_features=imu_features,
            goal_features=goal_features, action_dim=action_dim,
        )

        # Storage arrays.  obs and next_obs share the layout.
        c = self.capacity
        self._obs_lidar = np.zeros((c, n_beams), dtype=np.float32)
        self._obs_ped   = np.zeros((c, k_max, ped_features), dtype=np.float32)
        self._obs_pmask = np.zeros((c, k_max), dtype=np.uint8)
        self._obs_imu   = np.zeros((c, imu_features), dtype=np.float32)
        self._obs_goal  = np.zeros((c, goal_features), dtype=np.float32)

        self._nxt_lidar = np.zeros((c, n_beams), dtype=np.float32)
        self._nxt_ped   = np.zeros((c, k_max, ped_features), dtype=np.float32)
        self._nxt_pmask = np.zeros((c, k_max), dtype=np.uint8)
        self._nxt_imu   = np.zeros((c, imu_features), dtype=np.float32)
        self._nxt_goal  = np.zeros((c, goal_features), dtype=np.float32)

        self._action  = np.zeros((c, action_dim), dtype=np.float32)
        self._reward  = np.zeros(c, dtype=np.float32)
        self._done    = np.zeros(c, dtype=np.float32)
        self._entropy = np.zeros(c, dtype=np.float32)   # Ĥ stored at write time

        # Tree and bookkeeping.
        self.tree = SumTree(capacity)
        self._write = 0
        self._size = 0
        self._step = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: float,
        attention_entropy: float = 0.0,
    ) -> None:
        """Add a transition with maximum current priority (so it gets sampled)."""
        i = self._write
        self._obs_lidar[i] = obs["lidar"]
        self._obs_ped[i]   = obs["pedestrians"]
        self._obs_pmask[i] = obs["ped_mask"]
        self._obs_imu[i]   = obs["imu"]
        self._obs_goal[i]  = obs["goal"]

        self._nxt_lidar[i] = next_obs["lidar"]
        self._nxt_ped[i]   = next_obs["pedestrians"]
        self._nxt_pmask[i] = next_obs["ped_mask"]
        self._nxt_imu[i]   = next_obs["imu"]
        self._nxt_goal[i]  = next_obs["goal"]

        self._action[i]  = action
        self._reward[i]  = reward
        self._done[i]    = float(done)
        self._entropy[i] = float(attention_entropy)

        # Insert with current max priority.
        self.tree.update(i, self.tree.max_priority)

        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def _current_beta(self) -> float:
        """Linear annealing of beta from beta_start to beta_end."""
        if self._step >= self.beta_anneal_steps:
            return self.beta_end
        frac = self._step / max(1, self.beta_anneal_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch with priority-proportional probability.

        Returns dict with:
            obs.{lidar,pedestrians,ped_mask,imu,goal}
            next_obs.{...}
            action, reward, done
            indices  (for priority update after TD-error computation)
            is_weights (importance-sampling weights, normalized)
        """
        if self._size < batch_size:
            raise ValueError(
                f"not enough transitions ({self._size} < {batch_size})"
            )

        beta = self._current_beta()
        total = self.tree.total
        # Stratified sampling: split [0, total] into batch_size segments.
        seg = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for k in range(batch_size):
            v = self._rng.uniform(seg * k, seg * (k + 1))
            idx, prio = self.tree.get(v)
            indices[k] = idx
            priorities[k] = max(prio, 1e-12)

        probs = priorities / max(total, 1e-12)
        is_w = (self._size * probs) ** (-beta)
        is_w /= is_w.max()                      # normalize

        self._step += batch_size

        return {
            "obs_lidar": self._obs_lidar[indices],
            "obs_ped":   self._obs_ped[indices],
            "obs_pmask": self._obs_pmask[indices],
            "obs_imu":   self._obs_imu[indices],
            "obs_goal":  self._obs_goal[indices],
            "nxt_lidar": self._nxt_lidar[indices],
            "nxt_ped":   self._nxt_ped[indices],
            "nxt_pmask": self._nxt_pmask[indices],
            "nxt_imu":   self._nxt_imu[indices],
            "nxt_goal":  self._nxt_goal[indices],
            "action":   self._action[indices],
            "reward":   self._reward[indices],
            "done":     self._done[indices],
            "entropy":  self._entropy[indices],
            "indices":  indices,
            "is_weights": is_w.astype(np.float32),
            "beta":      beta,
        }

    # ------------------------------------------------------------------
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        entropies: Optional[np.ndarray] = None,
    ) -> None:
        """Update priorities after a critic update.

        Standard PER:    p = (|δ| + ε)^η
        Our extension:   p = (|δ| + ε)^η · (1 + λ(1 - Ĥ))^ν
        """
        td_abs = np.abs(td_errors).astype(np.float64) + self.epsilon
        td_term = td_abs ** self.eta

        if self.use_attention_entropy and entropies is not None:
            h_norm = np.clip(entropies.astype(np.float64), 0.0, 1.0)
            ent_term = (1.0 + self.lam * (1.0 - h_norm)) ** self.nu
            priorities = td_term * ent_term
        else:
            priorities = td_term

        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(max(p, self.epsilon)))

    # ------------------------------------------------------------------
    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size


# ======================================================================
# Factory
# ======================================================================
def build_per_from_config(cfg, action_dim: int = 2,
                          seed: Optional[int] = None
                          ) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(
        capacity=cfg.per["capacity"],
        n_beams=cfg.observation.lidar.n_beams,
        k_max=cfg.observation.pedestrian.k_max,
        ped_features=cfg.observation.pedestrian.n_features,
        imu_features=cfg.observation.imu.n_features,
        goal_features=cfg.observation.goal.n_features,
        action_dim=action_dim,
        eta=cfg.per["eta"],
        nu=cfg.per["nu"],
        lam=cfg.per["lam"],
        epsilon=cfg.per["epsilon"],
        beta_start=cfg.per["beta_is_start"],
        beta_end=cfg.per["beta_is_end"],
        beta_anneal_steps=cfg.per["beta_is_anneal_steps"],
        use_attention_entropy=cfg.per["use_attention_entropy"],
        seed=seed,
    )
