"""Soft Actor-Critic (SAC) update loop with auto-tuned temperature.

Reference:
    Haarnoja et al. 2018, "Soft Actor-Critic Algorithms and Applications".

v0.4 CRITICAL FIX — shared-trunk gradient flow.
    Previous versions had a representation-learning bug: the optimizers
    were built as
        actor_optim  = Adam(policy.parameters())   # encoder+attn+actor
        critic_optim = Adam(critic.parameters())    # Q-nets only
    so the critic loss never updated the encoder (its gradients on the
    trunk were computed then discarded), while the actor loss DID update
    the encoder. The encoder was therefore shaped only to maximize Q,
    never to make Q predictable. The critic chased a representation that
    moved under it every actor step, producing the classic divergence
    signature: td_error climbing, actor_loss climbing, q_min sinking.

    Standard fix (DrQ / SAC-AE convention): the SHARED TRUNK (encoder +
    attention) is trained by the CRITIC loss, and the actor consumes a
    DETACHED latent so its gradient stops at the actor's own input.

        critic_optim = Adam(critic.params + trunk.params)   # trunk = encoder+attn
        actor_optim  = Adam(actor.params)                    # actor only
        actor update uses xi.detach()

    This gives the representation a stable learning signal (the Bellman
    target) and stops the actor from reshaping it.
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
from cm_gap_sac_navigation.models.attention import pedestrian_attention_entropy


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
    """SAC agent built around an existing CmGapSacPolicy and a fresh critic.

    Parameter ownership (v0.4):
        - trunk_params : encoder + attention (everything in policy EXCEPT
                         policy.actor). Trained by the CRITIC optimizer.
        - actor_params : policy.actor only. Trained by the ACTOR optimizer.
        - critic_params: the twin Q networks.
    """

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

        self.log_alpha = torch.nn.Parameter(
            torch.tensor(float(init_log_alpha), device=device, dtype=torch.float32),
            requires_grad=True,
        )

        # ----- Parameter partition (v0.4 fix) --------------------------
        # Trunk = encoder + attention = every policy param NOT under actor.
        self.trunk_params = [
            p for n, p in self.policy.named_parameters()
            if not n.startswith("actor.")
        ]
        self.actor_params = list(self.policy.actor.parameters())

        # Critic optimizer trains the Q-nets AND the shared trunk, so the
        # representation is shaped by the stable Bellman target.
        self.critic_optim = torch.optim.Adam(
            list(self.critic.parameters()) + self.trunk_params,
            lr=lr_critic,
        )
        # Actor optimizer trains ONLY the actor head. The trunk is frozen
        # for the actor update (it consumes a detached latent).
        self.actor_optim = torch.optim.Adam(
            self.actor_params, lr=lr_actor,
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
        """One SAC update step on a PER batch."""
        t = self._batch_to_tensors(batch)
        alpha = self.alpha.detach()

        # ===== 1. Target y (no grad) ==================================
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

        # ===== 2. Critic update (trains Q-nets AND the trunk) =========
        # The encoder/attention gradients from this loss ARE applied,
        # because trunk_params are in critic_optim. This is the key v0.4
        # change: the representation is learned to make Q predictable.
        cur_out = self.policy.encode_state(
            lidar=t["obs_lidar"], pedestrians=t["obs_ped"],
            ped_mask=t["obs_pmask"], imu=t["obs_imu"], goal=t["obs_goal"],
        )
        xi = cur_out["xi"]
        q1, q2 = self.critic(xi, t["action"])
        td1 = q1 - y
        td2 = q2 - y
        critic_loss = (
            (t["is_weights"] * (td1.pow(2) + td2.pow(2))).mean() * 0.5
        )
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # Clip both the critic and the trunk (all params in critic_optim).
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + self.trunk_params, max_norm=10.0,
        )
        self.critic_optim.step()

        # Pedestrian attention entropy for PER, taken from the critic-path
        # encoding we just computed (detached: PER bookkeeping only).
        entropies = pedestrian_attention_entropy(
            alpha_p=cur_out["alpha_ped"],
            ped_mask=t["obs_pmask"],
        ).detach().cpu().numpy()

        # ===== 3. Actor update (trunk FROZEN, detached latent) ========
        # Re-encode with no grad to the trunk, then detach. The actor
        # gradient stops at its own input, so it cannot reshape the
        # representation (only the critic does).
        with torch.no_grad():
            enc_out = self.policy.encode_state(
                lidar=t["obs_lidar"], pedestrians=t["obs_ped"],
                ped_mask=t["obs_pmask"], imu=t["obs_imu"], goal=t["obs_goal"],
            )
        xi_detached = enc_out["xi"].detach()

        new_action, new_logp, _, _ = self.policy.actor(xi_detached)
        q1_pi, q2_pi = self.critic(xi_detached, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * new_logp - q_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=10.0)
        self.actor_optim.step()

        # ===== 4. Alpha update ========================================
        alpha_loss = -(
            self.log_alpha * (new_logp.detach() + self.target_entropy)
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ===== 5. Soft target update ==================================
        soft_update(source=self.critic, target=self.target_critic, tau=self.tau)

        # ===== 6. Diagnostics =========================================
        td_errors = (0.5 * (td1.detach() + td2.detach())).abs().cpu().numpy()

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