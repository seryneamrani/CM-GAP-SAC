"""End-to-end policy network for CM-GAP_SAC.

Wraps the four stages into one module:
    obs Dict -> encoders -> attention -> actor -> action

This is the object the training loop interacts with for action sampling.
The critic uses (ξ, action) directly so it does not need this wrapper.

The forward signature accepts both individual tensors and a batched
observation dict, mirroring the gym Dict observation produced by
LimoGazeboEnv.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cm_gap_sac_navigation.models.actor import (
    SquashedGaussianActor,
    build_actor_from_config,
)
from cm_gap_sac_navigation.models.attention import (
    CrossModalAttention,
    build_attention_from_config,
    pedestrian_attention_entropy,
)
from cm_gap_sac_navigation.models.encoders import (
    MultiModalEncoder,
    build_encoders_from_config,
)


class CmGapSacPolicy(nn.Module):
    """Forward stack: encoders + attention + actor.

    The encoders + attention path is shared between actor and critic
    updates: critics call `encode_state(obs)` to get ξ, then their own
    Q networks. This single module owns all the parameters of that
    shared trunk.
    """

    def __init__(
        self,
        encoder: MultiModalEncoder,
        attention: CrossModalAttention,
        actor: SquashedGaussianActor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.actor = actor

        # Sanity: actor.latent_dim must equal attention.latent_dim
        # (We don't store latent_dim on the actor directly, but the
        # first Linear layer's in_features must match.)
        first_layer = self.actor.trunk[0]
        if first_layer.in_features != self.attention.latent_dim:
            raise ValueError(
                f"Actor latent input ({first_layer.in_features}) does not match "
                f"attention output ({self.attention.latent_dim}). Check feature_dim "
                "and goal_mlp.feature_dim in the config."
            )

    # ------------------------------------------------------------------
    # State encoding (shared between actor and critic update paths)
    # ------------------------------------------------------------------
    def encode_state(
        self,
        lidar: torch.Tensor,
        pedestrians: torch.Tensor,
        ped_mask: torch.Tensor,
        imu: torch.Tensor,
        goal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run encoders + attention. Returns the attention output dict."""
        latents = self.encoder(
            lidar=lidar,
            pedestrians=pedestrians,
            ped_mask=ped_mask,
            imu=imu,
            goal=goal,
        )
        return self.attention(
            h_lidar=latents["h_lidar"],
            h_ped=latents["h_pedestrian"],
            ped_mask=latents["ped_mask"],
            h_imu=latents["h_imu"],
            h_goal=latents["h_goal"],
        )

    # ------------------------------------------------------------------
    # Full forward: state -> action
    # ------------------------------------------------------------------
    def forward(
        self,
        lidar: torch.Tensor,
        pedestrians: torch.Tensor,
        ped_mask: torch.Tensor,
        imu: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample an action from the policy.

        Returns a dict with:
            action       : (B, action_dim)
            log_prob     : (B,)
            mean_action  : (B, action_dim)
            xi           : (B, latent_dim)            for critic
            alpha_ped    : (B, K)                     for PER
            ped_attn_entropy : (B,)                   normalized entropy
            gates        : dict of (B, d) per-source gates (for diagnostics)
        """
        attn_out = self.encode_state(lidar, pedestrians, ped_mask, imu, goal)
        xi = attn_out["xi"]

        action, log_prob, mean_action, _ = self.actor(xi, deterministic=deterministic)

        entropy = pedestrian_attention_entropy(
            alpha_p=attn_out["alpha_ped"],
            ped_mask=ped_mask,
        )

        return {
            "action": action,
            "log_prob": log_prob,
            "mean_action": mean_action,
            "xi": xi,
            "alpha_ped": attn_out["alpha_ped"],
            "ped_attn_entropy": entropy,
            "gates": {
                "lidar": attn_out["gate_lidar"],
                "ped":   attn_out["gate_ped"],
                "imu":   attn_out["gate_imu"],
            },
        }

    # ------------------------------------------------------------------
    # Convenience: from a numpy obs dict (for the env interaction loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_from_obs(
        self,
        obs: Dict,                              # numpy arrays from env
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a single observation (un-batched) from the env and act.

        Returns:
            action    : (action_dim,) tensor on `device`
            log_prob  : scalar tensor
            entropy   : scalar tensor (for PER bookkeeping)
        """
        def _b(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

        out = self.forward(
            lidar=_b(obs["lidar"]),
            pedestrians=_b(obs["pedestrians"]),
            ped_mask=_b(obs["ped_mask"]),
            imu=_b(obs["imu"]),
            goal=_b(obs["goal"]),
            deterministic=deterministic,
        )
        return (
            out["action"].squeeze(0),
            out["log_prob"].squeeze(0),
            out["ped_attn_entropy"].squeeze(0),
        )


# ======================================================================
# Factory
# ======================================================================
def build_policy_from_config(cfg) -> CmGapSacPolicy:
    encoder = build_encoders_from_config(cfg)
    attention = build_attention_from_config(cfg)
    actor = build_actor_from_config(cfg, latent_dim=attention.latent_dim)
    return CmGapSacPolicy(encoder=encoder, attention=attention, actor=actor)
