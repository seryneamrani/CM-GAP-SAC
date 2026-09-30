"""Pure (no-ROS) reward function for CM-GAP_SAC.

Extracted from the env so it can be unit-tested without needing rclpy
or Gazebo. The env imports `compute_reward` from here.

The function returns a dict with the per-term breakdown, which is logged
by the env in `info` for diagnostics and ablation studies.

Reward decomposition (seven additive terms):
    r_goal      : success terminal              (>0)
    r_collision : collision terminal            (<0)
    r_progress  : potential-based shaping       (Ng et al. 1999)
    r_prox      : Hall 1966 intimate-zone       (<=0)
    r_smooth    : action jerk penalty           (<=0)   Lee et al. 2020
    r_time      : cost of living per step       (<0)
    r_reverse   : asymmetric forward bias       (<=0)

The asymmetry |r_collision| > r_goal (250 vs 200 by default) implements
asymmetric pessimism: collisions are strictly worse than non-arrival,
which biases the policy toward conservative behavior.

Note on r_shield:
    The CBF shield penalty is intentionally NOT computed here.
    `compute_reward` must remain a pure function of MDP dynamics
    (state, action, transition). The shield is a controller augmentation,
    not a property of the MDP. Mixing them would violate separation of
    responsibilities and complicate ablation studies.
    The r_shield term is computed in train.py, which is the sole source
    of that penalty. To disable it, pass shield=None in the config.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cm_gap_sac_navigation.utils.config_loader import RewardCfg


def min_pedestrian_distance(obs: Dict[str, np.ndarray]) -> float:
    """Euclidean distance to the nearest pedestrian in the robot frame.

    Returns +inf if no pedestrian is currently tracked.
    """
    mask = obs["ped_mask"]
    if mask.sum() == 0:
        return float("inf")
    active = obs["pedestrians"][mask.astype(bool)]
    return float(np.linalg.norm(active[:, :2], axis=1).min())


def compute_reward(
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    prev_action: np.ndarray,
    prev_d_goal: float | None,
    backward_dist_acc: float,
    r_reflex: float,
    terminated: bool,
    info: Dict[str, Any],
    cfg: RewardCfg,
) -> Dict[str, float]:
    """Compute the 7-term reward, returning a per-term breakdown.

    Args:
        obs:         current observation Dict (with goal, pedestrians, ped_mask, lidar)
        action:      current action (v, omega), shape (2,)
        prev_action: previous action; (0, 0) on first step.
        prev_d_goal: previous distance to goal; None on first step.
        terminated:  episode terminated flag
        info:        info dict, must have "outcome" if terminated
        cfg:         RewardCfg with the reward coefficients

    Returns:
        dict {r_goal, r_collision, r_progress, r_prox, r_smooth, r_time,
              r_reverse, total, d_min_ped}

    Note:
        r_shield is NOT included here; see train.py for that penalty.
    """
    # Term 1: terminal success
    r_goal = float(cfg.r_goal) if (terminated and info.get("outcome") == "success") else 0.0

    # Term 2: terminal collision
    r_coll = float(cfg.r_collision) if (terminated and info.get("outcome") == "collision") else 0.0

    # Term 3: potential-based progress shaping (Ng et al. 1999)
    d_now = float(obs["goal"][0])
    if prev_d_goal is not None:
        r_progress = float(cfg.c_progress * (prev_d_goal - d_now))
    else:
        r_progress = 0.0

   # ------------------------------------------------------------------
    # Social closing-velocity (shared by Term 9 and the progress gate).
    # v_close > 0 means the robot-pedestrian distance is shrinking.
    # ASSUMES obs["pedestrians"][:, 2:4] = pedestrian velocity RELATIVE
    # to the robot, in the robot frame. Verify on a logged episode.
    # ------------------------------------------------------------------
    d_min_ped = min_pedestrian_distance(obs)
    v_close = 0.0
    if np.isfinite(d_min_ped) and d_min_ped < cfg.d_social:
        active = obs["pedestrians"][obs["ped_mask"].astype(bool)]
        i = int(np.argmin(np.linalg.norm(active[:, :2], axis=1)))
        px, py = float(active[i, 0]), float(active[i, 1])
        vx, vy = float(active[i, 2]), float(active[i, 3])
        v_close = max(0.0, -(px * vx + py * vy) / max(d_min_ped, 1e-3))



    # Term 4: proxemic penalty, normalisée pour que alpha_prox = pénalité AU contact,
    # invariante à d_intimate. intrusion_norm: 0 au bord de confort, 1 au contact.
    #deleteed d_min_ped = min_pedestrian_distance(obs)
    denom = max(cfg.d_intimate - cfg.collision_radius_ped, 1e-6)
    intrusion_norm = float(np.clip(
        (cfg.d_intimate - d_min_ped) / denom, 0.0, 1.0))
    r_prox = float(-cfg.alpha_prox * intrusion_norm ** 2)

    # Term 5: smoothness penalty (Lee et al. 2020)
    delta_v = abs(float(action[0]) - float(prev_action[0]))
    delta_w = abs(float(action[1]) - float(prev_action[1]))
    r_smooth = float(-cfg.alpha_smooth * (delta_v + delta_w))

    # Term 6: cost of living
    r_time = float(cfg.r_time)

    # Term 7: gentle asymmetric forward bias with cumulative tolerance.
    # The first cfg.d_reverse_free meters of backward motion per episode are
    # unpenalized: the robot may freely back out of tight spots. Beyond that,
    # the penalty ramps in over cfg.d_reverse_sat additional meters and
    # saturates. This permits legitimate avoidance maneuvers while
    # discouraging backward locomotion as a dominant mode.
    v_back = max(0.0, -float(action[0]))
    gate = float(np.clip(
        (backward_dist_acc - cfg.d_reverse_free) / max(cfg.d_reverse_sat, 1e-6),
        0.0, 1.0,
    ))
    r_reverse = -cfg.alpha_reverse * v_back * gate

    # Term 8 (NEW): static-obstacle proximity penalty.
    # Discourages frôlement of walls/furniture before collision happens.
    # Same hinge-quadratic shape as r_prox for pedestrians, for consistency.
    # d_min_static comes from the 360° LiDAR scan: the closest point on
    # any static obstacle. Note that pedestrians also appear in the LiDAR,
    # so this term will spuriously fire near pedestrians; we accept that
    # double-counting because r_prox already handles the pedestrian case
    # with finer (ground-truth) information, and the static term simply
    # reinforces the avoidance signal.
    d_min_static = float(obs["lidar"].min())
    denom_s = max(cfg.d_static_safe - cfg.collision_radius_static, 1e-6)
    intrusion_static = float(np.clip(
        (cfg.d_static_safe - d_min_static) / denom_s, 0.0, 1.0))
    r_static_prox = float(-cfg.alpha_static_prox * intrusion_static ** 2)


    # Term 9 (A): social closing-velocity penalty.
    if v_close > 0.0:
        r_social = float(-cfg.alpha_social * v_close
                         * (1.0 - d_min_ped / cfg.d_social) ** 2)
    else:
        r_social = 0.0

    # Progress gate (B), blended by cfg.social_gate_blend (0=off, 1=full).
    if v_close > 0.1 and np.isfinite(d_min_ped) and d_min_ped < cfg.d_social:
        gate = float(np.clip(d_min_ped / cfg.d_social, 0.2, 1.0))
        gate = 1.0 + cfg.social_gate_blend * (gate - 1.0)
        r_progress *= gate



    return {
        "r_goal":         r_goal,
        "r_collision":    r_coll,
        "r_progress":     r_progress,
        "r_prox":         r_prox,
        "r_smooth":       r_smooth,
        "r_social":       r_social,
        "v_close":        v_close, 
        "r_time":         r_time,
        "r_reverse":      r_reverse,
        "r_reflex":       r_reflex,
        "r_static_prox":  r_static_prox,
        "total":          r_goal + r_coll + r_progress + r_prox
                          + r_smooth + r_time + r_reverse + r_reflex
                          + r_static_prox + r_social,
        "d_min_ped":      d_min_ped,
        "d_min_static":   d_min_static,
    }