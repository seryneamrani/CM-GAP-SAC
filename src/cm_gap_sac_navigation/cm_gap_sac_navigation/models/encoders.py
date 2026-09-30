"""Encoders for the multi-modal CM-GAP_SAC state (v0.2: tri-modal + goal).

Four independent encoders, each producing a fixed-shape latent that the
attention layer fuses.

Design contract
---------------
LidarEncoder:
    in:  (B, n_beams)        raw downsampled scan
    out: (B, M, d)            M=9 patches of dimension d=64

PedestrianEncoder:
    in:  (B, K, n_features)   per-track features in robot frame
         (B, K)                binary mask
    out: (B, K, d)             per-track latent

ImuEncoder:                   NEW in v0.2
    in:  (B, n_features)       6 normalized IMU features
    out: (B, d)                IMU latent vector

GoalEncoder:
    in:  (B, n_goal)          [d_goal, theta, v, omega]
    out: (B, d_goal)           compact context (d/2)

Why these choices
-----------------
LiDAR is a 1D ordered, periodic signal: Conv1D with circular padding.
Pedestrians are an unordered SET: shared MLP (Deep Sets, Zaheer 2017).
IMU is a small dense vector: standard MLP.
Goal is a small dense vector: standard MLP.

The lidar, pedestrian, and IMU encoders all output dimension d (=64 by
config) so the gated fusion at the attention stage can combine them
without extra projection. Goal stays at d/2 (=32): it is the query and is
also concatenated to the fused state, but does not need to live in the
same space as the attended modalities.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# LiDAR encoder
# ======================================================================
class LidarEncoder(nn.Module):
    """1D CNN over a downsampled LiDAR scan with circular padding."""

    def __init__(
        self,
        n_beams: int,
        in_channels: int = 1,
        hidden_channels: tuple[int, ...] = (32, 64),
        feature_dim: int = 64,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("LidarEncoder expects exactly 2 hidden stages")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep symmetric padding")

        c1, c2 = hidden_channels
        pad = kernel_size // 2

        self._pad = pad
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size,
                               stride=2, padding=0)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size,
                               stride=2, padding=0)
        self.proj = nn.Conv1d(c2, feature_dim, kernel_size=1, stride=1)

        self.feature_dim = feature_dim
        self._n_beams = n_beams
        self._n_patches = self._compute_n_patches(n_beams, kernel_size)

    @staticmethod
    def _compute_n_patches(n_beams: int, k: int) -> int:
        n = (n_beams + 1) // 2
        n = (n + 1) // 2
        return n

    @property
    def n_patches(self) -> int:
        return self._n_patches

    @staticmethod
    def _circular_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad == 0:
            return x
        return F.pad(x, (pad, pad), mode="circular")

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        if lidar.dim() != 2:
            raise ValueError(f"expected (B, n_beams), got {tuple(lidar.shape)}")

        x = lidar.unsqueeze(1)                      # (B, 1, n_beams)
        x = self._circular_pad(x, self._pad)
        x = F.relu(self.conv1(x))
        x = self._circular_pad(x, self._pad)
        x = F.relu(self.conv2(x))
        x = self.proj(x)                            # (B, d, M)
        x = x.transpose(1, 2).contiguous()          # (B, M, d)
        return x


# ======================================================================
# Pedestrian encoder
# ======================================================================
class PedestrianEncoder(nn.Module):
    """Permutation-equivariant per-track MLP (Deep Sets, Zaheer 2017)."""

    def __init__(
        self,
        n_features: int = 5,
        hidden_dims: tuple[int, ...] = (64, 64),
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, feature_dim))
        self.mlp = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def forward(
        self,
        ped: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ped.dim() != 3:
            raise ValueError(f"expected (B, K, F), got {tuple(ped.shape)}")

        B, K, F_in = ped.shape
        h = self.mlp(ped.reshape(B * K, F_in)).reshape(B, K, self.feature_dim)

        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)

        return h


# ======================================================================
# IMU encoder (NEW in v0.2)
# ======================================================================
class ImuEncoder(nn.Module):
    """Standard MLP for the small dense IMU vector.

    Input is the normalized 6-feature IMU [ax, ay, az, wx, wy, wz] with
    each entry already clipped to [-1, 1] by the env. Output is a
    feature_dim-vector that the attention stage will treat as a third
    modality alongside LiDAR and pedestrians.
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden_dims: tuple[int, ...] = (64, 64),
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, feature_dim))
        self.mlp = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu: (B, n_features) float32 in [-1, 1]

        Returns:
            (B, feature_dim)
        """
        if imu.dim() != 2:
            raise ValueError(f"expected (B, F), got {tuple(imu.shape)}")
        return self.mlp(imu)


# ======================================================================
# Goal encoder
# ======================================================================
class GoalEncoder(nn.Module):
    """Small MLP turning the 4-d goal vector into a context embedding."""

    def __init__(
        self,
        n_features: int = 4,
        hidden_dims: tuple[int, ...] = (32,),
        feature_dim: int = 32,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, feature_dim))
        self.mlp = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        if goal.dim() != 2:
            raise ValueError(f"expected (B, F), got {tuple(goal.shape)}")
        return self.mlp(goal)


# ======================================================================
# MultiModalEncoder (4 modalities in v0.2)
# ======================================================================
class MultiModalEncoder(nn.Module):
    """Container that runs all four encoders and returns the latents.

    Returns the latents separately so the attention layer can fuse them
    with explicit knowledge of which is which.
    """

    def __init__(
        self,
        lidar_encoder: LidarEncoder,
        pedestrian_encoder: PedestrianEncoder,
        imu_encoder: ImuEncoder,                 # NEW positional arg in v0.2
        goal_encoder: GoalEncoder,
    ) -> None:
        super().__init__()
        self.lidar = lidar_encoder
        self.pedestrian = pedestrian_encoder
        self.imu = imu_encoder
        self.goal = goal_encoder

        # Sanity: lidar, ped, imu must share feature_dim for downstream
        # gated fusion (the goal stays at d/2 by design).
        if not (lidar_encoder.feature_dim
                == pedestrian_encoder.feature_dim
                == imu_encoder.feature_dim):
            raise ValueError(
                "LidarEncoder, PedestrianEncoder and ImuEncoder must share "
                "feature_dim for the tri-source gated fusion. Got "
                f"lidar={lidar_encoder.feature_dim}, "
                f"ped={pedestrian_encoder.feature_dim}, "
                f"imu={imu_encoder.feature_dim}"
            )

    def forward(
        self,
        lidar: torch.Tensor,
        pedestrians: torch.Tensor,
        ped_mask: torch.Tensor,
        imu: torch.Tensor,                       # NEW in v0.2
        goal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "h_lidar":      self.lidar(lidar),                       # (B, M, d)
            "h_pedestrian": self.pedestrian(pedestrians, ped_mask),  # (B, K, d)
            "h_imu":        self.imu(imu),                           # (B, d)
            "h_goal":       self.goal(goal),                         # (B, d/2)
            "ped_mask":     ped_mask,                                # (B, K)
        }


# ======================================================================
# Factory
# ======================================================================
def build_encoders_from_config(cfg) -> MultiModalEncoder:
    """Construct the full MultiModalEncoder from a parsed Config object."""
    lidar_cfg = cfg.encoders["lidar_cnn"]
    ped_cfg   = cfg.encoders["pedestrian_mlp"]
    imu_cfg   = cfg.encoders["imu_mlp"]
    goal_cfg  = cfg.encoders["goal_mlp"]

    lidar_enc = LidarEncoder(
        n_beams=cfg.observation.lidar.n_beams,
        in_channels=lidar_cfg["in_channels"],
        hidden_channels=tuple(lidar_cfg["hidden_channels"]),
        feature_dim=lidar_cfg["feature_dim"],
        kernel_size=lidar_cfg["kernel_size"],
    )
    ped_enc = PedestrianEncoder(
        n_features=cfg.observation.pedestrian.n_features,
        hidden_dims=tuple(ped_cfg["hidden_dims"]),
        feature_dim=ped_cfg["feature_dim"],
    )
    imu_enc = ImuEncoder(
        n_features=cfg.observation.imu.n_features,
        hidden_dims=tuple(imu_cfg["hidden_dims"]),
        feature_dim=imu_cfg["feature_dim"],
    )
    goal_enc = GoalEncoder(
        n_features=cfg.observation.goal.n_features,
        hidden_dims=tuple(goal_cfg["hidden_dims"]),
        feature_dim=goal_cfg["feature_dim"],
    )
    return MultiModalEncoder(lidar_enc, ped_enc, imu_enc, goal_enc)
