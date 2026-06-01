"""Prioritized Experience Replay buffer with GPU-resident storage (v0.3).

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

Storage (v0.3):
    All transition tensors are resident on the training device (typically
    'cuda'). add() transfers NumPy observations from the env to GPU storage
    once via copy_(); sample() performs GPU-side indexing and returns batches
    that are already on device, removing the CPU→GPU transfer that previously
    happened inside SacAgent._batch_to_tensors at every gradient step.

    The SumTree stays on CPU: its sampling logic is sequential and
    branch-heavy, which does not benefit from GPU parallelism.

Memory footprint (capacity 200k, default obs shape):
    ~1.3 GB VRAM total. Fits comfortably on 8 GB+ GPUs.

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

    Auxiliary scalars (rewards, dones, indices, IS weights) follow the
    same convention: numerical ones live on GPU, indices stay on CPU because
    the SumTree update path consumes them as Python ints.
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

        # Entropy stored at write time — kept on CPU because PER priority update
        # uses freshly computed entropies from the current policy (see SAC).
        # This slot is retained for compatibility and optional diagnostics.
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
        """Add a transition with maximum current priority.

        Each call transfers ~6.5 kB from CPU to GPU. With multi-env IPC at
        e.g. 100 sps aggregated, that is < 1 MB/s on the PCIe bus — negligible.
        """
        i = self._write

        # GPU writes via copy_(): single per-tensor DMA into the storage slot.
        # non_blocking=True is harmless without pinned memory (becomes sync).
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
        """Linear annealing of beta from beta_start to beta_end."""
        if self._step >= self.beta_anneal_steps:
            return self.beta_end
        frac = self._step / max(1, self.beta_anneal_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, object]:
        """Sample a batch with priority-proportional probability.

        Returns a dict whose keys point to:
            - bulk obs/action/reward/done/is_weights tensors on `self.device`
            - 'indices' as a CPU NumPy int64 array (consumed by update_priorities)
            - 'beta' as a Python float (for TensorBoard logging)
        """
        if self._size < batch_size:
            raise ValueError(
                f"not enough transitions ({self._size} < {batch_size})"
            )

        beta = self._current_beta()
        total = self.tree.total
        seg = total / batch_size

        # ---- CPU phase: sample indices and priorities from the SumTree ----
        indices_np = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for k in range(batch_size):
            v = self._rng.uniform(seg * k, seg * (k + 1))
            idx, prio = self.tree.get(v)
            indices_np[k] = idx
            priorities[k] = max(prio, 1e-12)

        probs = priorities / max(total, 1e-12)
        is_w = (self._size * probs) ** (-beta)
        is_w /= is_w.max()

        self._step += batch_size

        # ---- GPU phase: tensor-indexed gather (no host transfer of bulk data) ----
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
            "indices":    indices_np,   # CPU numpy, consumed by update_priorities
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

        Standard PER:    p = (|δ| + ε)^η
        Our extension:   p = (|δ| + ε)^η · (1 + λ(1 - Ĥ))^ν

        td_errors and entropies arrive as CPU numpy from SAC (which calls
        .cpu().numpy() at the end of update()), matching the SumTree path.
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