"""SAC twin critics: two independent Q networks.

The min(Q1, Q2) trick comes from TD3 (Fujimoto et al. 2018) and reduces the
positive bias that single-network Q-learning is prone to. SAC (Haarnoja et al.
2018) adopted it directly.

Each critic is a small MLP that takes (ξ, action) -> scalar Q value. The two
networks share the same architecture but have independent parameters.

A `TargetTwinCritic` wrapper is provided that creates a deep-copy of a
TwinCritic for use as the target network with soft updates τ.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q network: (xi, action) -> scalar."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 2,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim + action_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi:     (B, latent_dim)
            action: (B, action_dim)
        Returns:
            (B,) Q values.
        """
        if xi.dim() != 2 or action.dim() != 2:
            raise ValueError(
                f"xi and action must be (B, *); got {tuple(xi.shape)} "
                f"and {tuple(action.shape)}"
            )
        x = torch.cat([xi, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent Q networks, used jointly with min(Q1, Q2)."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 2,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(latent_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(latent_dim, action_dim, hidden_dims)

    def forward(
        self, xi: torch.Tensor, action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(xi, action), self.q2(xi, action)

    def q_min(self, xi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(xi, action)
        return torch.min(q1, q2)


def make_target(critic: TwinCritic) -> TwinCritic:
    """Deep-copy a TwinCritic to be used as a target network.

    Target parameters do not require gradients; they are updated by
    soft Polyak averaging in the SAC training loop:
        θ_target ← τ θ + (1 - τ) θ_target
    """
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place Polyak averaging: target = τ source + (1 - τ) target.

    Should be called after each critic update.
    """
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


# ======================================================================
# Factory
# ======================================================================
def build_critic_from_config(cfg, latent_dim: int) -> TwinCritic:
    h_cfg = cfg.heads
    return TwinCritic(
        latent_dim=latent_dim,
        action_dim=cfg.action.get("dim", 2),
        hidden_dims=tuple(h_cfg["critic_hidden"]),
    )
