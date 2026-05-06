"""Cross-modal gated attention (CM-GAP_SAC originality 1).

Three mechanisms operating in parallel, then fused by a per-feature gate:

  1. Self-attention over the M LiDAR patches.
     Lets the network identify salient angular sectors (gap edges,
     narrow openings, close obstacles) by relating each patch to all
     others.

  2. Cross-attention over the K pedestrian tokens.
     Query is built from [h_goal, h_imu], not just h_goal as in
     Liu et al. 2020 (RGL). This couples proprioception and social
     attention: a robot feeling a physical perturbation (IMU) attends
     to pedestrians differently. Mask is enforced in the softmax so
     padded slots receive exactly zero weight.

  3. Direct IMU branch.
     The IMU latent is not attended (it is a single vector, not a set
     or a sequence). It is used as part of the query above and as a
     third source in the gated fusion below.

Fusion: softmax 3-way per-feature gate.
  For each feature dimension d, the network outputs three logits
  (one per source). Softmax forces them to sum to 1 along that axis,
  so each feature is a learned partition between LiDAR, pedestrian,
  and IMU evidence.

Output: ξ = [z; h_goal] consumed by the SAC heads.

Side output: entropy of the pedestrian attention distribution
(scalar per sample). The PER buffer uses it to amplify the priority
of transitions where attention focused sharply.

References:
  - Vaswani et al. 2017 (multi-head attention).
  - Chen et al. 2019 SARL, Liu et al. 2020 RGL (crowd-aware attention).
  - Srivastava et al. 2015 (Highway Networks: per-feature gates).
  - Lee et al. 2020 ANYmal (proprioception + exteroception fusion).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Helper: scaled dot-product attention with optional mask
# ======================================================================
def _scaled_dot_product_attention(
    q: torch.Tensor,                       # (B, Tq, d)
    k: torch.Tensor,                       # (B, Tk, d)
    v: torch.Tensor,                       # (B, Tk, d)
    mask: Optional[torch.Tensor] = None,   # (B, Tk) with 1 = valid, 0 = pad
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention output and the (B, Tq, Tk) weight tensor.

    Mask handling: invalid keys get a large negative pre-softmax score
    so they receive zero weight after softmax.
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)   # (B, Tq, Tk)

    if mask is not None:
        # mask: (B, Tk) -> (B, 1, Tk) so it broadcasts over the Tq axis.
        mask_b = mask.unsqueeze(1).to(scores.dtype)
        scores = scores.masked_fill(mask_b == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    # If a row is all -inf (no valid key for this query), softmax produces
    # NaN. Replace by zeros so the downstream weighted sum is zero.
    if mask is not None:
        weights = torch.nan_to_num(weights, nan=0.0)

    out = torch.matmul(weights, v)                               # (B, Tq, d)
    return out, weights


# ======================================================================
# Self-attention over LiDAR patches
# ======================================================================
class LidarSelfAttention(nn.Module):
    """Multi-head self-attention over the M LiDAR patches.

    The output is mean-pooled across patches to produce a single context
    vector c_lidar of dimension d.
    """

    def __init__(self, feature_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        if feature_dim % n_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.d = feature_dim
        self.h = n_heads
        self.dh = feature_dim // n_heads

        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_o = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, h_lidar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_lidar: (B, M, d)

        Returns:
            (B, d) pooled context vector.
        """
        B, M, _ = h_lidar.shape
        # Project Q, K, V then split into heads: (B, h, M, dh)
        q = self.W_q(h_lidar).reshape(B, M, self.h, self.dh).transpose(1, 2)
        k = self.W_k(h_lidar).reshape(B, M, self.h, self.dh).transpose(1, 2)
        v = self.W_v(h_lidar).reshape(B, M, self.h, self.dh).transpose(1, 2)

        # Per-head scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)                      # (B, h, M, dh)

        # Concat heads back and project
        attended = attended.transpose(1, 2).reshape(B, M, self.d)
        out = self.W_o(attended)                                 # (B, M, d)

        # Mean-pool across patches: c_lidar = mean_m out_m
        return out.mean(dim=1)                                   # (B, d)


# ======================================================================
# Cross-attention over pedestrians, query from [goal, imu]
# ======================================================================
class PedestrianCrossAttention(nn.Module):
    """Multi-head cross-attention with masked padded slots.

    Query is built from a concatenation of h_goal and (optionally) h_imu.
    Toggling `use_imu_in_query` to False reduces to the goal-only query
    of Liu et al. 2020 (RGL); this is exactly ablation A1 (no IMU).

    Returns:
        c_ped:   (B, d)   attended pedestrian context
        alpha_p: (B, K)   per-pedestrian attention weights, summed over heads
                          and used by the PER for entropy weighting
    """

    def __init__(
        self,
        feature_dim: int = 64,           # d for pedestrians, lidar, imu
        goal_dim: int = 32,              # d/2 for the goal encoder
        n_heads: int = 4,
        use_imu_in_query: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim % n_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.d = feature_dim
        self.h = n_heads
        self.dh = feature_dim // n_heads
        self.use_imu_in_query = use_imu_in_query

        # Query input dimension depends on whether we concatenate IMU.
        q_in_dim = goal_dim + (feature_dim if use_imu_in_query else 0)

        self.W_q = nn.Linear(q_in_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_o = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(
        self,
        h_ped: torch.Tensor,             # (B, K, d)
        ped_mask: torch.Tensor,          # (B, K)
        h_goal: torch.Tensor,            # (B, d_goal)
        h_imu: torch.Tensor,             # (B, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = h_ped.shape

        # Build query.
        if self.use_imu_in_query:
            q_in = torch.cat([h_goal, h_imu], dim=-1)            # (B, d_goal + d)
        else:
            q_in = h_goal                                        # (B, d_goal)

        # Single query vector per sample (not per-token), so Tq = 1.
        q_full = self.W_q(q_in).unsqueeze(1)                     # (B, 1, d)
        k_full = self.W_k(h_ped)                                 # (B, K, d)
        v_full = self.W_v(h_ped)                                 # (B, K, d)

        # Reshape to heads: q (B, h, 1, dh), k/v (B, h, K, dh)
        q = q_full.reshape(B, 1, self.h, self.dh).transpose(1, 2)
        k = k_full.reshape(B, K, self.h, self.dh).transpose(1, 2)
        v = v_full.reshape(B, K, self.h, self.dh).transpose(1, 2)

        # Apply mask to scores (broadcasts over heads).
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)
        # scores: (B, h, 1, K)
        mask_b = ped_mask.unsqueeze(1).unsqueeze(1).to(scores.dtype)
        scores = scores.masked_fill(mask_b == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)              # all-padded case

        attended = torch.matmul(weights, v)                       # (B, h, 1, dh)
        attended = attended.transpose(1, 2).reshape(B, 1, self.d).squeeze(1)
        c_ped = self.W_o(attended)                                # (B, d)

        # Aggregate per-head attention weights into a single (B, K)
        # distribution by averaging across heads. This is what the PER
        # uses for entropy weighting.
        alpha_p = weights.squeeze(2).mean(dim=1)                  # (B, K)

        return c_ped, alpha_p


# ======================================================================
# Tri-source softmax gate (per-feature partition)
# ======================================================================
class TriSourceGate(nn.Module):
    """Per-feature softmax gate over three sources: LiDAR, pedestrians, IMU.

    For each feature dimension d, outputs (g_lidar_d, g_ped_d, g_imu_d) that
    sum to 1, giving a learned partition of attention budget across modalities.
    """

    def __init__(self, feature_dim: int = 64, goal_dim: int = 32,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        # Input: concatenation of the three sources + goal context.
        #   c_lidar (d) + c_ped (d) + h_imu (d) + h_goal (d_goal)
        in_dim = 3 * feature_dim + goal_dim
        # Output: 3 gate vectors of size feature_dim, stacked as 3*d logits.
        out_dim = 3 * feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        c_lidar: torch.Tensor,
        c_ped:   torch.Tensor,
        h_imu:   torch.Tensor,
        h_goal:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            g_lidar, g_ped, g_imu : each (B, d), summing to 1 along the
            "source" axis when stacked: g_lidar + g_ped + g_imu = 1 elementwise.
        """
        x = torch.cat([c_lidar, c_ped, h_imu, h_goal], dim=-1)    # (B, 3d + d_goal)
        logits = self.mlp(x)                                       # (B, 3d)
        B = logits.shape[0]
        # Reshape to (B, 3, d) and softmax over the source axis.
        logits = logits.reshape(B, 3, self.feature_dim)
        gates = torch.softmax(logits, dim=1)                       # (B, 3, d)
        return gates[:, 0], gates[:, 1], gates[:, 2]


# ======================================================================
# Top-level: CrossModalAttention
# ======================================================================
class CrossModalAttention(nn.Module):
    """Full attention + gated fusion stage.

    Inputs are the dict produced by MultiModalEncoder; output is the latent
    state vector ξ consumed by the SAC heads, plus the pedestrian attention
    distribution α_p for PER consumption.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        goal_dim: int = 32,
        n_heads: int = 4,
        use_imu_in_query: bool = True,
        gate_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.lidar_self_attn = LidarSelfAttention(feature_dim, n_heads)
        self.ped_cross_attn  = PedestrianCrossAttention(
            feature_dim=feature_dim,
            goal_dim=goal_dim,
            n_heads=n_heads,
            use_imu_in_query=use_imu_in_query,
        )
        self.gate = TriSourceGate(feature_dim, goal_dim, gate_hidden_dim)

        self.feature_dim = feature_dim
        self.goal_dim = goal_dim
        self.latent_dim = feature_dim + goal_dim   # dim of ξ

    def forward(
        self,
        h_lidar: torch.Tensor,        # (B, M, d)
        h_ped:   torch.Tensor,        # (B, K, d)
        ped_mask: torch.Tensor,       # (B, K)
        h_imu:   torch.Tensor,        # (B, d)
        h_goal:  torch.Tensor,        # (B, d_goal)
    ) -> dict[str, torch.Tensor]:
        # 1. Self-attention on LiDAR.
        c_lidar = self.lidar_self_attn(h_lidar)                   # (B, d)

        # 2. Cross-attention on pedestrians, query from [goal, imu].
        c_ped, alpha_p = self.ped_cross_attn(
            h_ped=h_ped, ped_mask=ped_mask, h_goal=h_goal, h_imu=h_imu,
        )                                                          # (B, d), (B, K)

        # 3. Gates partitioned over (LiDAR, pedestrians, IMU).
        g_l, g_p, g_i = self.gate(c_lidar, c_ped, h_imu, h_goal)

        # 4. Per-feature partition fusion.
        z = g_l * c_lidar + g_p * c_ped + g_i * h_imu              # (B, d)

        # 5. Final latent: append goal context.
        xi = torch.cat([z, h_goal], dim=-1)                        # (B, d + d_goal)

        return {
            "xi": xi,
            "alpha_ped": alpha_p,            # (B, K) for PER
            "z": z,
            "c_lidar": c_lidar,
            "c_ped": c_ped,
            "gate_lidar": g_l,
            "gate_ped":   g_p,
            "gate_imu":   g_i,
        }


# ======================================================================
# Helper: pedestrian attention entropy (used by PER)
# ======================================================================
def pedestrian_attention_entropy(
    alpha_p: torch.Tensor,           # (B, K)
    ped_mask: torch.Tensor,          # (B, K)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized Shannon entropy of the pedestrian attention.

    Normalization: divide by log(K_eff) where K_eff is the number of
    valid pedestrian slots in each sample. Returns NaN-safe values in
    [0, 1]; samples with zero or one valid pedestrian get entropy 0.
    """
    # Effective K per sample.
    k_eff = ped_mask.sum(dim=-1).clamp_min(1).to(alpha_p.dtype)   # (B,)

    # Shannon entropy.
    log_a = torch.log(alpha_p.clamp_min(eps))
    h_raw = -(alpha_p * log_a).sum(dim=-1)                        # (B,)

    # Normalize by log(K_eff). When k_eff <= 1, log(k_eff) <= 0 -> set to 0.
    log_k = torch.log(k_eff.clamp_min(2.0))                       # avoid log(1)=0
    h_norm = h_raw / log_k

    # Force samples with no valid pedestrians to entropy 0.
    no_ped = ped_mask.sum(dim=-1) == 0
    h_norm = torch.where(no_ped, torch.zeros_like(h_norm), h_norm)

    return h_norm.clamp(0.0, 1.0)


# ======================================================================
# Factory
# ======================================================================
def build_attention_from_config(cfg) -> CrossModalAttention:
    """Construct a CrossModalAttention from a parsed Config object."""
    att_cfg = cfg.attention
    return CrossModalAttention(
        feature_dim=att_cfg["feature_dim"],
        goal_dim=cfg.encoders["goal_mlp"]["feature_dim"],
        n_heads=att_cfg["n_heads"],
        use_imu_in_query=att_cfg.get("use_imu_in_query", True),
    )
