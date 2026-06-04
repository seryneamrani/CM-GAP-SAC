"""Soft Actor-Critic (SAC) update loop with auto-tuned temperature.

Reference:
    Haarnoja et al. 2018, "Soft Actor-Critic Algorithms and Applications".

Implements:
    - Twin Q targets, taking min(Q1', Q2') for the bootstrap value
    - Importance-sampling-weighted critic loss (compatible with PER)
    - Auto-tuned temperature α via gradient on a target entropy
    - Soft target update with τ
    - Returns the per-transition TD errors so the PER buffer can update
      priorities

The agent owns:
    - policy:        CmGapSacPolicy (encoders + attention + actor) — trained
    - critic:        TwinCritic on shared latent ξ                  — trained
    - target_critic: deep-copy of critic, frozen, soft-updated      — target
    - log_alpha:     learnable temperature                          — trained

Training flow per gradient step:
    1. Sample batch from PER, compute IS weights and current α.
    2. With no_grad: compute target Q via target critic on next-state action
       sampled from current policy (next_log_prob included).
       y = r + γ (1 - done) (min(Q1', Q2') - α · next_log_prob)
    3. Critic update: minimize IS-weighted MSE between (Q1, Q2) and y.
       Returns TD errors |Q - y| for priority update.
    4. Actor update: minimize α · log_prob - min(Q1, Q2) on freshly sampled
       action through the encoder + attention (gradients flow into the trunk).
    5. Alpha update: minimize -log_alpha · (log_prob.detach() + target_entropy)
    6. Soft-update target critic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cm_gap_sac_navigation.models.critic import TwinCritic, soft_update, make_target
from cm_gap_sac_navigation.models.policy import CmGapSacPolicy


@dataclass
class SacUpdateInfo:
    """Diagnostics returned by one update step."""
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    log_prob_mean: float
    q_min_mean: float
    td_error_mean: float
    td_error_max: float


class SacAgent:
    """SAC agent built around an existing CmGapSacPolicy and a fresh critic."""

    def __init__(
        self,
        policy: CmGapSacPolicy,
        critic: TwinCritic,
        device: torch.device,
        gamma: float = 0.98,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        target_entropy: float = -2.0,
        init_log_alpha: float = 0.0,
    ) -> None:
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.target_critic = make_target(critic).to(device)
        self.device = device

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = float(target_entropy)

        # Trainable temperature: optimize log α so α stays positive.
        # Init à -1.0 → α ≈ 0.37 au lieu de 1.0 (évite sur-exploration initiale)
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=device, dtype=torch.float32),
            requires_grad=True,
        )

        # Optimizers.
        # Note: encoder + attention + actor live in `policy`. The critic has
        # its own parameters. Gradient w.r.t. critic loss flows through
        # encoder + attention because the critic consumes ξ produced by them;
        # however we don't want critic loss to update the actor, so the
        # actor-loss path uses freshly sampled actions through the same
        # encoder. This is the standard SAC pattern.
        self.actor_optim = torch.optim.Adam(
            self.policy.parameters(), lr=lr_actor,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=lr_alpha,
        )

    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def _batch_to_tensors(
        self, batch: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        """Move batch to device with dtype normalization.

        PER v0.3 returns tensors already on self.device; .to(d) is a no-op
        in that case. The .float() on pmask is needed because PER stores
        the mask as uint8 for memory but arithmetic needs float32.
        """
        d = self.device
        return {
            "obs_lidar":   batch["obs_lidar"].to(d),
            "obs_ped":     batch["obs_ped"].to(d),
            "obs_pmask":   batch["obs_pmask"].to(d).float(),
            "obs_imu":     batch["obs_imu"].to(d),
            "obs_goal":    batch["obs_goal"].to(d),
            "nxt_lidar":   batch["nxt_lidar"].to(d),
            "nxt_ped":     batch["nxt_ped"].to(d),
            "nxt_pmask":   batch["nxt_pmask"].to(d).float(),
            "nxt_imu":     batch["nxt_imu"].to(d),
            "nxt_goal":    batch["nxt_goal"].to(d),
            "action":      batch["action"].to(d),
            "reward":      batch["reward"].to(d),
            "done":        batch["done"].to(d),
            "is_weights":  batch["is_weights"].to(d),
        }

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, np.ndarray]
               ) -> Tuple[SacUpdateInfo, np.ndarray, np.ndarray]:
        """One SAC update step on a PER batch.

        Returns:
            info:       SacUpdateInfo (diagnostics)
            td_errors:  per-transition TD error magnitudes (for PER update)
            entropies:  per-transition pedestrian attention entropies
                        (for PER update; computed from current state)
        """
        t = self._batch_to_tensors(batch)
        alpha = self.alpha.detach()

        # ----- 1. Compute target y -----
        with torch.no_grad():
            nxt_out = self.policy(
                lidar=t["nxt_lidar"], pedestrians=t["nxt_ped"],
                ped_mask=t["nxt_pmask"], imu=t["nxt_imu"], goal=t["nxt_goal"],
            )
            nxt_action = nxt_out["action"]
            nxt_logp = nxt_out["log_prob"]
            nxt_xi = nxt_out["xi"]
            q1_t, q2_t = self.target_critic(nxt_xi, nxt_action)
            q_t = torch.min(q1_t, q2_t) - alpha * nxt_logp
            y = t["reward"] + self.gamma * (1.0 - t["done"]) * q_t

        # ----- 2. Critic update -----
        # Use stop-gradient on the encoder for the critic so the critic
        # update does not also update the trunk through the action input.
        # Standard SAC: critic gradients DO flow through the encoder of the
        # current state. We keep that path; only the next-state path is
        # frozen (already done via no_grad above).
        cur_out = self.policy.encode_state(
            lidar=t["obs_lidar"], pedestrians=t["obs_ped"],
            ped_mask=t["obs_pmask"], imu=t["obs_imu"], goal=t["obs_goal"],
        )
        xi = cur_out["xi"]
        q1, q2 = self.critic(xi, t["action"])
        td1 = q1 - y
        td2 = q2 - y
        # IS-weighted MSE.
        critic_loss = (
            (t["is_weights"] * (td1.pow(2) + td2.pow(2))).mean() * 0.5
        )
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # Gradient clipping: protect against early-stage gradient spikes
        # caused by large terminal rewards (r_goal=200, r_collision=-250)
        # before the critic has stabilized.
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optim.step()

        # ----- 3. Actor update -----
        # Recompute through full forward to get fresh action and log_prob.
        # This time gradients flow into encoder + attention + actor.
        cur_full = self.policy(
            lidar=t["obs_lidar"], pedestrians=t["obs_ped"],
            ped_mask=t["obs_pmask"], imu=t["obs_imu"], goal=t["obs_goal"],
        )
        new_action = cur_full["action"]
        new_logp = cur_full["log_prob"]
        new_xi = cur_full["xi"]
        q1_pi, q2_pi = self.critic(new_xi, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * new_logp - q_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.actor_optim.step()

        # ----- 4. Alpha update -----
        alpha_loss = -(self.log_alpha * (new_logp.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ----- 5. Soft target update -----
        soft_update(source=self.critic, target=self.target_critic, tau=self.tau)

        # ----- 6. Per-transition TD errors and entropies for PER -----
        td_errors = (0.5 * (td1.detach() + td2.detach())).abs().cpu().numpy()
        entropies = cur_full["ped_attn_entropy"].detach().cpu().numpy()

        info = SacUpdateInfo(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            alpha_loss=float(alpha_loss.detach().item()),
            alpha=float(self.alpha.detach().item()),
            log_prob_mean=float(new_logp.detach().mean().item()),
            q_min_mean=float(q_pi.detach().mean().item()),
            td_error_mean=float(td_errors.mean()),
            td_error_max=float(td_errors.max()),
        )
        return info, td_errors, entropies

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.critic.load_state_dict(sd["critic"])
        self.target_critic.load_state_dict(sd["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.actor_optim.load_state_dict(sd["actor_optim"])
        self.critic_optim.load_state_dict(sd["critic_optim"])
        self.alpha_optim.load_state_dict(sd["alpha_optim"])


# ======================================================================
# Factory
# ======================================================================
def build_sac_agent(cfg, policy: CmGapSacPolicy, device: torch.device) -> SacAgent:
    from cm_gap_sac_navigation.models.critic import build_critic_from_config
    critic = build_critic_from_config(cfg, latent_dim=policy.attention.latent_dim)
    return SacAgent(
        policy=policy,
        critic=critic,
        device=device,
        gamma=cfg.sac["gamma"],
        tau=cfg.sac["tau"],
        lr_actor=cfg.sac["lr_actor"],
        lr_critic=cfg.sac["lr_critic"],
        lr_alpha=cfg.sac["lr_alpha"],
        target_entropy=cfg.sac["target_entropy"],
    )
