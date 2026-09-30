"""Prioritized Experience Replay buffer with GPU-resident storage (v0.4).

This is originality 2 of CM-GAP_SAC: priority of a transition combines an
ASYMMETRIC TD-error term with a multiplicative factor based on the
pedestrian attention entropy.

Priority formula (v0.4 ASYMMETRIC):
    p_i = (max(0, δ_i) + ε)^η · (1 + λ · (1 - Ĥ_i))^ν

where:
    δ_i     is the SIGNED TD-error (positive = critic underestimated =
            successes / progress; negative = critic overestimated =
            collisions / surprise punishments).
    Ĥ_i     is the normalized pedestrian attention entropy in [0, 1]
    η       is the standard PER exponent (default 0.6)
    ν       is the entropy exponent (default 0.5)
    λ       is the entropy amplification (default 3.0)
    ε       small positive constant to avoid zero priorities

Rationale for asymmetric prioritization (v0.4):
    Standard Schaul 2016 PER prioritizes by |δ|, which amplifies BOTH
    successes and collisions. In a navigation task with frequent terminal
    collisions (r_collision negative), collision transitions tend to have
    the largest |δ|, dominate replay, and push the critic into pessimism.
    By clipping at zero we re-sample only positive surprises (successes,
    progress) at high frequency, while collisions still appear in the
    buffer at minimal priority (epsilon) for coverage.

Importance sampling weights:
    w_i = (1 / (N · P(i)))^β     normalized by 1 / max_j w_j

with β annealed linearly from beta_start (0.4) to beta_end (1.0) over
beta_anneal_steps (in SAMPLE steps, i.e., one per gradient update).

Bug fix (v0.4):
    self._step now increments by 1 per sample() call, not by batch_size.
    Previously beta saturated to 1.0 after ~7k env steps, which made
    importance sampling correction too aggressive too early.

Bug fix (v0.4):
    state_dict() / load_state_dict() methods added so the buffer can be
    serialized in checkpoints. Without this, --resume restarted the buffer
    empty and lost the beta annealing state.

Reference:
    Schaul, Quan, Antonoglou, Silver. "Prioritized Experience Replay" (2016).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ======================================================================
# SumTree: efficient O(log N) sampling proportional to priority.
# Lives on CPU. Sequential traversal does not benefit from GPU.
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
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._max_priority = 1.0

    def update(self, leaf_idx: int, priority: float) -> None:
        if not (0 <= leaf_idx < self.capacity):
            raise IndexError(
                f"leaf_idx {leaf_idx} out of range [0, {self.capacity})"
            )
        tree_idx = leaf_idx + self.capacity - 1
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change
        if priority > self._max_priority:
            self._max_priority = priority

    def get(self, value: float) -> Tuple[int, float]:
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
# PER buffer (GPU-resident)
# ======================================================================
class PrioritizedReplayBuffer:
    """Ring buffer with SumTree-based priority sampling for Dict observations.

    All bulk transition data lives on `device` (typically 'cuda'). At write
    time, NumPy arrays from the env are transferred once to GPU storage.
    At sample time, GPU-side indexing yields batch tensors directly on
    device, with no CPU↔GPU transfer in the SAC hot path.
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
        device: Optional[torch.device] = None,
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

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.shapes = _TransitionShapes(
            n_beams=n_beams, k_max=k_max,
            ped_features=ped_features, imu_features=imu_features,
            goal_features=goal_features, action_dim=action_dim,
        )

        # ---- GPU storage ----
        c = self.capacity
        d = self.device

        self._obs_lidar = torch.zeros((c, n_beams), dtype=torch.float32, device=d)
        self._obs_ped   = torch.zeros((c, k_max, ped_features), dtype=torch.float32, device=d)
        self._obs_pmask = torch.zeros((c, k_max), dtype=torch.uint8, device=d)
        self._obs_imu   = torch.zeros((c, imu_features), dtype=torch.float32, device=d)
        self._obs_goal  = torch.zeros((c, goal_features), dtype=torch.float32, device=d)

        self._nxt_lidar = torch.zeros((c, n_beams), dtype=torch.float32, device=d)
        self._nxt_ped   = torch.zeros((c, k_max, ped_features), dtype=torch.float32, device=d)
        self._nxt_pmask = torch.zeros((c, k_max), dtype=torch.uint8, device=d)
        self._nxt_imu   = torch.zeros((c, imu_features), dtype=torch.float32, device=d)
        self._nxt_goal  = torch.zeros((c, goal_features), dtype=torch.float32, device=d)

        self._action = torch.zeros((c, action_dim), dtype=torch.float32, device=d)
        self._reward = torch.zeros(c, dtype=torch.float32, device=d)
        self._done   = torch.zeros(c, dtype=torch.float32, device=d)

        self._entropy_cpu = np.zeros(c, dtype=np.float32)

        # SumTree + bookkeeping live on CPU.
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
        """Add a transition with maximum current priority."""
        i = self._write

        self._obs_lidar[i].copy_(torch.from_numpy(obs["lidar"]), non_blocking=True)
        self._obs_ped[i].copy_(torch.from_numpy(obs["pedestrians"]), non_blocking=True)
        self._obs_pmask[i].copy_(torch.from_numpy(obs["ped_mask"]), non_blocking=True)
        self._obs_imu[i].copy_(torch.from_numpy(obs["imu"]), non_blocking=True)
        self._obs_goal[i].copy_(torch.from_numpy(obs["goal"]), non_blocking=True)

        self._nxt_lidar[i].copy_(torch.from_numpy(next_obs["lidar"]), non_blocking=True)
        self._nxt_ped[i].copy_(torch.from_numpy(next_obs["pedestrians"]), non_blocking=True)
        self._nxt_pmask[i].copy_(torch.from_numpy(next_obs["ped_mask"]), non_blocking=True)
        self._nxt_imu[i].copy_(torch.from_numpy(next_obs["imu"]), non_blocking=True)
        self._nxt_goal[i].copy_(torch.from_numpy(next_obs["goal"]), non_blocking=True)

        self._action[i].copy_(torch.from_numpy(action), non_blocking=True)
        self._reward[i] = float(reward)
        self._done[i] = float(done)
        self._entropy_cpu[i] = float(attention_entropy)

        self.tree.update(i, self.tree.max_priority)
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def _current_beta(self) -> float:
        """Linear annealing of beta from beta_start to beta_end.

        _step is in SAMPLE units (one increment per sample() call). With
        the typical 1:1 ratio of gradient_steps : env_steps, this is also
        equal to the number of gradient updates so far.
        """
        if self._step >= self.beta_anneal_steps:
            return self.beta_end
        frac = self._step / max(1, self.beta_anneal_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, object]:
        """Sample a batch with priority-proportional probability."""
        if self._size < batch_size:
            raise ValueError(
                f"not enough transitions ({self._size} < {batch_size})"
            )

        beta = self._current_beta()
        total = self.tree.total
        seg = total / batch_size

        # Vectorized random value generation: one call instead of batch_size
        # Python-level calls. Saves ~1-2ms per sample on batch_size=256.
        bounds = np.linspace(0.0, total, batch_size + 1)
        v_array = self._rng.uniform(bounds[:-1], bounds[1:])

        indices_np = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for k in range(batch_size):
            idx, prio = self.tree.get(float(v_array[k]))
            indices_np[k] = idx
            priorities[k] = max(prio, 1e-12)

        probs = priorities / max(total, 1e-12)
        is_w = (self._size * probs) ** (-beta)
        is_w /= is_w.max()

        # BUG #3 FIX: increment by 1, not by batch_size.
        # beta_anneal_steps is now in SAMPLE units (= gradient steps).
        self._step += 1

        d = self.device
        idx_gpu = torch.from_numpy(indices_np).to(d)
        is_w_gpu = torch.from_numpy(is_w.astype(np.float32)).to(d)

        return {
            "obs_lidar":  self._obs_lidar[idx_gpu],
            "obs_ped":    self._obs_ped[idx_gpu],
            "obs_pmask":  self._obs_pmask[idx_gpu],
            "obs_imu":    self._obs_imu[idx_gpu],
            "obs_goal":   self._obs_goal[idx_gpu],
            "nxt_lidar":  self._nxt_lidar[idx_gpu],
            "nxt_ped":    self._nxt_ped[idx_gpu],
            "nxt_pmask":  self._nxt_pmask[idx_gpu],
            "nxt_imu":    self._nxt_imu[idx_gpu],
            "nxt_goal":   self._nxt_goal[idx_gpu],
            "action":     self._action[idx_gpu],
            "reward":     self._reward[idx_gpu],
            "done":       self._done[idx_gpu],
            "is_weights": is_w_gpu,
            "indices":    indices_np,
            "beta":       beta,
        }

    # ------------------------------------------------------------------
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        entropies: Optional[np.ndarray] = None,
    ) -> None:
        """Update priorities after a critic update.

        ASYMMETRIC PER (v0.4 with bug #1 fix in sac.py):
            p = (max(0, δ) + ε)^η · (1 + λ(1 - Ĥ))^ν

        td_errors arrives SIGNED from sac.py (the .abs() was removed).
        We clip negative TD-errors (collisions, surprise punishments) to
        zero so they receive minimal priority epsilon. Positive TD-errors
        (successes, progress) are prioritized normally.
        """
        # BUG #1 INTERACTION: td_errors is now signed. Clip at 0 to
        # implement asymmetric prioritization.
        td_pos = np.clip(td_errors.astype(np.float64), 0.0, None) + self.epsilon
        td_term = td_pos ** self.eta

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

    # ------------------------------------------------------------------
    # BUG #4 FIX: serialization for checkpoint resume
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        """Serialize buffer state for checkpointing.

        Tensors are moved to CPU for portability. Only the populated slice
        [:size] is saved (no need to save unused capacity). Total size on
        a 200k-capacity buffer is ~1.3 GB when full.
        """
        size = self._size
        return {
            "obs_lidar":  self._obs_lidar[:size].cpu(),
            "obs_ped":    self._obs_ped[:size].cpu(),
            "obs_pmask":  self._obs_pmask[:size].cpu(),
            "obs_imu":    self._obs_imu[:size].cpu(),
            "obs_goal":   self._obs_goal[:size].cpu(),
            "nxt_lidar":  self._nxt_lidar[:size].cpu(),
            "nxt_ped":    self._nxt_ped[:size].cpu(),
            "nxt_pmask":  self._nxt_pmask[:size].cpu(),
            "nxt_imu":    self._nxt_imu[:size].cpu(),
            "nxt_goal":   self._nxt_goal[:size].cpu(),
            "action":     self._action[:size].cpu(),
            "reward":     self._reward[:size].cpu(),
            "done":       self._done[:size].cpu(),
            "entropy_cpu": self._entropy_cpu[:size].copy(),
            "tree_array": self.tree._tree.copy(),
            "tree_max_priority": self.tree._max_priority,
            "write": self._write,
            "size":  self._size,
            "step":  self._step,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        """Restore buffer from a state_dict. Validates capacity match."""
        size = int(sd["size"])
        if size > self.capacity:
            raise ValueError(
                f"Saved buffer size {size} exceeds current capacity {self.capacity}"
            )
        d = self.device
        self._obs_lidar[:size] = sd["obs_lidar"].to(d)
        self._obs_ped[:size]   = sd["obs_ped"].to(d)
        self._obs_pmask[:size] = sd["obs_pmask"].to(d)
        self._obs_imu[:size]   = sd["obs_imu"].to(d)
        self._obs_goal[:size]  = sd["obs_goal"].to(d)
        self._nxt_lidar[:size] = sd["nxt_lidar"].to(d)
        self._nxt_ped[:size]   = sd["nxt_ped"].to(d)
        self._nxt_pmask[:size] = sd["nxt_pmask"].to(d)
        self._nxt_imu[:size]   = sd["nxt_imu"].to(d)
        self._nxt_goal[:size]  = sd["nxt_goal"].to(d)
        self._action[:size]    = sd["action"].to(d)
        self._reward[:size]    = sd["reward"].to(d)
        self._done[:size]      = sd["done"].to(d)
        self._entropy_cpu[:size] = sd["entropy_cpu"]
        self.tree._tree[:] = sd["tree_array"]
        self.tree._max_priority = float(sd["tree_max_priority"])
        self._write = int(sd["write"])
        self._size  = size
        self._step  = int(sd["step"])
        self._rng.bit_generator.state = sd["rng_state"]


# ======================================================================
# Factory
# ======================================================================
def build_per_from_config(
    cfg,
    action_dim: int = 2,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> PrioritizedReplayBuffer:
    """Builds the PER buffer. `device` defaults to CUDA if available."""
    return PrioritizedReplayBuffer(
        capacity=cfg.per["capacity"],
        n_beams=cfg.observation.lidar.n_beams,
        k_max=cfg.observation.pedestrian.k_max,
        ped_features=cfg.observation.pedestrian.n_features,
        imu_features=cfg.observation.imu.n_features,
        goal_features=cfg.observation.goal.n_features,
        action_dim=action_dim,
        device=device,
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