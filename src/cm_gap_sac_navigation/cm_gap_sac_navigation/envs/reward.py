"""Pure (no-ROS) reward function for CM-GAP_SAC.

Extracted from the env so it can be unit-tested without needing rclpy
or Gazebo. The env imports `compute_reward` from here.

The function returns a dict with the per-term breakdown, which is logged
by the env in `info` for diagnostics and ablation studies.

Reward decomposition (six additive terms):
    r_goal      : success terminal              (>0)
    r_collision : collision terminal            (<0)
    r_progress  : potential-based shaping       (Ng et al. 1999)
    r_prox      : Hall 1966 intimate-zone       (<=0)
    r_smooth    : action jerk penalty           (<=0)   Lee et al. 2020
    r_time      : cost of living per step       (<0)

The asymmetry |r_collision| > r_goal (250 vs 200 by default) implements
asymmetric pessimism: collisions are strictly worse than non-arrival,
which biases the policy toward conservative behavior.
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
    terminated: bool,
    info: Dict[str, Any],
    cfg: RewardCfg,
) -> Dict[str, float]:
    """Compute the 6-term reward, returning a per-term breakdown.

    Args:
        obs:         current observation Dict (with goal, pedestrians, ped_mask, lidar)
        action:      current action (v, omega), shape (2,)
        prev_action: previous action; (0, 0) on first step.
        prev_d_goal: previous distance to goal; None on first step.
        terminated:  episode terminated flag
        info:        info dict, must have "outcome" if terminated
        cfg:         RewardCfg with the 7 coefficients

    Returns:
        dict {r_goal, r_collision, r_progress, r_prox, r_smooth, r_time,
              total, d_min_ped}
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

    # Term 4: proxemic penalty (Hall 1966 intimate zone)
    d_min_ped = min_pedestrian_distance(obs)
    intrusion = max(0.0, cfg.d_intimate - d_min_ped)
    r_prox = float(-cfg.alpha_prox * intrusion ** 2)

    # Term 5: smoothness penalty (Lee et al. 2020)
    delta_v = abs(float(action[0]) - float(prev_action[0]))
    delta_w = abs(float(action[1]) - float(prev_action[1]))
    r_smooth = float(-cfg.alpha_smooth * (delta_v + delta_w))

    # Term 6: cost of living
    r_time = float(cfg.r_time)

    return {
        "r_goal": r_goal,
        "r_collision": r_coll,
        "r_progress": r_progress,
        "r_prox": r_prox,
        "r_smooth": r_smooth,
        "r_time": r_time,
        "total": r_goal + r_coll + r_progress + r_prox + r_smooth + r_time,
        "d_min_ped": d_min_ped,
    }
