"""Soft Actor-Critic (SAC) update loop with auto-tuned temperature.

Reference:
    Haarnoja et al. 2018, "Soft Actor-Critic Algorithms and Applications".

v0.5 — Bug fixes from end-to-end review:

  Bug #1: td_errors.abs() killed the sign before the PER buffer could
          apply asymmetric prioritization. Fixed by passing signed TD
          to the buffer; the buffer is responsible for the asymmetric clip.

  Bug #2 (lite): Actor backward used to compute (and discard) gradients
          for the entire critic network. Fixed by freezing critic params
          via requires_grad=False during the actor phase. Saves compute
          and VRAM (5-10% SPS gain).

  Bug #2 (full): Double-encoding of the current state (once for critic,
          once for actor) was wasteful. Fixed by reusing xi.detach() from
          the critic phase, so the actor sees the trunk *before* the
          critic step. Standard DrQ-style optimization.

v0.4 — Critical shared-trunk gradient flow fix (kept):
    The shared trunk (encoder + attention) is trained by the CRITIC loss,
    and the actor consumes a DETACHED latent so its gradient stops at the
    actor's own input. The representation is shaped by the stable Bellman
    target, not by the actor's local objective.
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
        self.trunk_params = [
            p for n, p in self.policy.named_parameters()
            if not n.startswith("actor.")
        ]
        self.actor_params = list(self.policy.actor.parameters())

        self.critic_optim = torch.optim.Adam(
            list(self.critic.parameters()) + self.trunk_params,
            lr=lr_critic,
        )
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
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + self.trunk_params, max_norm=10.0,
        )
        self.critic_optim.step()

        # Pedestrian attention entropy for PER (computed once, reused below)
        entropies = pedestrian_attention_entropy(
            alpha_p=cur_out["alpha_ped"],
            ped_mask=t["obs_pmask"],
        ).detach().cpu().numpy()

        # ===== 3. Actor update =========================================
        # Freeze critic gradients during actor backward (compute savings).
        # Trunk is NOT frozen explicitly because xi.detach() already cuts
        # the computational graph at xi — gradients cannot reach the trunk.
        for p in self.critic.parameters():
            p.requires_grad = False

        xi_detached = xi.detach()

        new_action, new_logp, _, _ = self.policy.actor(xi_detached)
        q1_pi, q2_pi = self.critic(xi_detached, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * new_logp - q_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=10.0)
        self.actor_optim.step()

        # Restore critic gradients for next iteration's critic update.
        for p in self.critic.parameters():
            p.requires_grad = True

        # ===== 4. Alpha update ========================================
        alpha_loss = -(
            self.log_alpha * (new_logp.detach() + self.target_entropy)
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ===== 5. Soft target update ==================================
        soft_update(source=self.critic, target=self.target_critic, tau=self.tau)

        # ===== 6. Diagnostics + signed TD-errors for PER ==============
        # Bug #1 fix: pass SIGNED TD-errors to the PER buffer. The buffer
        # is responsible for the asymmetric clip (positive surprises
        # prioritized = successes; negative surprises = collisions get
        # minimal priority).
        # Bug #1 + Bug #2 fix: pass SIGNED TD-errors to the PER buffer
        # with the correct sign convention (positive = critic underestimated
        # = success/progress surprise; negative = critic overestimated =
        # collision/punishment surprise).
        # Convention: TD = target - prediction (Sutton & Barto), NOT
        # prediction - target. This means a positive TD signals "reality
        # was better than predicted" which is what we want to prioritize.
        td_signed = -((td1.detach() + td2.detach()) * 0.5).cpu().numpy()
        # Equivalent to: 0.5 * ((y - q1) + (y - q2)).cpu().numpy()
        # We use the negation because td1 = q1 - y was already computed
        # for the critic loss (where the squared form makes sign irrelevant).

        td_abs = np.abs(td_signed)


        info = SacUpdateInfo(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            alpha_loss=float(alpha_loss.detach().item()),
            alpha=float(self.alpha.detach().item()),
            log_prob_mean=float(new_logp.detach().mean().item()),
            q_min_mean=float(q_pi.detach().mean().item()),
            td_error_mean=float(td_abs.mean()),
            td_error_max=float(td_abs.max()),
        )
        return info, td_signed, entropies

    # ------------------------------------------------------------------
    def reset_critic_late_layers(self) -> int:
        """Reset the last 2 Linear layers of each Q-head (Nikishin 2022 variant B).

        Keeps the first Linear that encodes (xi, action) → hidden, resets
        deeper layers (which carry accumulated pessimistic estimates).
        Also resets target_critic in sync via hard copy, and clears the
        critic optimizer state so Adam doesn't apply stale momentum to
        freshly-initialized weights.

        Returns the number of Linear layers reset (should be 4: 2 per Q-head).
        """
        n_reset = 0

        def _reset_qnetwork(qnet):
            nonlocal n_reset
            # qnet.net Sequential structure:
            #   [0] Linear(in, 256)         ← KEEP
            #   [1] LayerNorm(256)          ← KEEP
            #   [2] ReLU                    ← no params
            #   [3] Linear(256, 256)        ← RESET
            #   [4] LayerNorm(256)          ← RESET
            #   [5] ReLU                    ← no params
            #   [6] Linear(256, 1)          ← RESET
            modules = list(qnet.net)
            for idx in (3, 4, 6):
                m = modules[idx]
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    n_reset += 1
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Reset both critic Q-heads in the LIVE critic.
        _reset_qnetwork(self.critic.q1)
        _reset_qnetwork(self.critic.q2)

        # Hard sync target critic to the freshly-reset live critic.
        # Copy parameter tensors directly (not state_dict to avoid buffer
        # issues with running statistics in LayerNorm).
        with torch.no_grad():
            for p_target, p_live in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):
                p_target.data.copy_(p_live.data)

        # Clear Adam momentum on the critic optimizer.
        # IMPORTANT: use .clear() to preserve the defaultdict(dict) type.
        # Replacing with {} would break the next call to .step() because
        # Adam accesses state[p] for new params via defaultdict semantics.
        self.critic_optim.state.clear()

        return n_reset
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