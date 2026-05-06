"""SAC actor: squashed Gaussian policy with reparameterization trick.

Reference: Haarnoja et al. 2018, "Soft Actor-Critic Algorithms and Applications".
The actor outputs the mean and log-std of a diagonal Gaussian over a tanh-
squashed action. The reparameterization trick allows backpropagation through
the stochastic sampling. The log-prob requires a correction term to account
for the change of variables introduced by tanh (Appendix C of the paper).

The action space of the LIMO Pro is rectangular but not centered at zero:
v ∈ [-0.3, 0.6] m/s, ω ∈ [-1.0, 1.0] rad/s. We therefore apply an affine
rescaling from tanh space [-1, 1] to the actual robot bounds.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical floor for tanh squashing correction.
_LOG_STD_EPS = 1e-6


class SquashedGaussianActor(nn.Module):
    """Stochastic policy producing actions in [low, high] via tanh squashing.

    Forward returns four things:
        action       : (B, action_dim)  reparameterized sample, IN ROBOT BOUNDS
        log_prob     : (B,)              log π(action | state), squash-corrected
        mean_action  : (B, action_dim)  deterministic mean (for evaluation)
        pre_tanh     : (B, action_dim)  raw u before tanh, for diagnostics
    """

    def __init__(
        self,
        latent_dim: int,                       # dim of ξ produced by attention
        action_dim: int = 2,                   # (v, ω)
        hidden_dims: tuple[int, ...] = (256, 256),
        action_low:  tuple[float, float] = (-0.3, -1.0),
        action_high: tuple[float, float] = ( 0.6,  1.0),
        log_std_min: float = -20.0,
        log_std_max: float =   2.0,
    ) -> None:
        super().__init__()
        if action_dim != len(action_low) or action_dim != len(action_high):
            raise ValueError("action_low / action_high length mismatch with action_dim")

        # Trunk MLP
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Two heads: mean and log_std
        self.head_mean    = nn.Linear(prev, action_dim)
        self.head_log_std = nn.Linear(prev, action_dim)

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Affine rescaling from [-1, 1] to [low, high].
        low  = torch.tensor(action_low,  dtype=torch.float32)
        high = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale",  (high - low) / 2.0)
        self.register_buffer("action_bias",   (high + low) / 2.0)

    # ------------------------------------------------------------------
    # Forward: stochastic sample
    # ------------------------------------------------------------------
    def forward(
        self,
        xi: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xi: (B, latent_dim)  output of CrossModalAttention
            deterministic: if True, return tanh(mean) without sampling.
                           Used for evaluation / deployment.
        """
        h = self.trunk(xi)
        mean = self.head_mean(h)
        log_std = self.head_log_std(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            # Deterministic path: skip sampling, no log_prob needed for inference.
            u = mean
            squashed = torch.tanh(u)
            action = squashed * self.action_scale + self.action_bias
            log_prob = torch.zeros(xi.shape[0], device=xi.device, dtype=xi.dtype)
            mean_action = action
            return action, log_prob, mean_action, u

        # Reparameterization trick: u = mean + std * eps, eps ~ N(0, I).
        eps = torch.randn_like(mean)
        u = mean + std * eps                                     # pre-tanh sample
        squashed = torch.tanh(u)                                  # in [-1, 1]
        action = squashed * self.action_scale + self.action_bias  # in [low, high]

        # Log-prob with tanh squashing correction.
        # Base Gaussian log prob:  -0.5 * ((u - mean) / std)^2 - log(std) - 0.5*log(2π)
        # which simplifies (since u = mean + std*eps) to:
        #   log N(eps) - log(std)
        # Sum over action dimensions.
        log_prob_gauss = (
            -0.5 * eps.pow(2)
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)                                            # (B,)

        # Tanh correction:  log |det( d a_tanh / d u )| = sum log(1 - tanh(u)^2)
        # Numerically stable form:  2 * (log(2) - u - softplus(-2u))
        # See: Haarnoja et al. 2018, Appendix C, and the SAC reference impl.
        log_correction = (
            2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        ).sum(dim=-1)                                            # (B,)

        # Affine rescaling: log(action_scale) per dimension, summed.
        # This term is constant in mean/std but matters for absolute log_prob.
        log_scale = torch.log(self.action_scale + _LOG_STD_EPS).sum()

        log_prob = log_prob_gauss - log_correction - log_scale

        # Deterministic mean for diagnostic / evaluation usage.
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action, u

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, xi: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Deployment helper. Returns action only, on the same device as xi."""
        action, _, _, _ = self.forward(xi, deterministic=deterministic)
        return action


# ======================================================================
# Factory
# ======================================================================
def build_actor_from_config(cfg, latent_dim: int) -> SquashedGaussianActor:
    h_cfg = cfg.heads
    return SquashedGaussianActor(
        latent_dim=latent_dim,
        action_dim=cfg.action.get("dim", 2),
        hidden_dims=tuple(h_cfg["actor_hidden"]),
        action_low=(cfg.robot.v_min, cfg.robot.omega_min),
        action_high=(cfg.robot.v_max, cfg.robot.omega_max),
        log_std_min=h_cfg["log_std_min"],
        log_std_max=h_cfg["log_std_max"],
    )
