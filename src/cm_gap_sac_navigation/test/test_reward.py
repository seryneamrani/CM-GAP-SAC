"""Tests for the 6-term reward function (pure module, no ROS required)."""
from __future__ import annotations

import numpy as np
import pytest

from cm_gap_sac_navigation.envs.reward import (
    compute_reward,
    min_pedestrian_distance,
)
from cm_gap_sac_navigation.utils.config_loader import RewardCfg


@pytest.fixture
def reward_cfg():
    return RewardCfg(
        r_goal=200.0, r_collision=-250.0, c_progress=5.0,
        d_intimate=0.45, alpha_prox=10.0, alpha_smooth=0.05, r_time=-0.01,
    )


def make_obs(d_goal: float = 1.0,
             ped_positions=None,
             min_lidar: float = 5.0):
    """Build an observation Dict for testing."""
    if ped_positions is None:
        peds = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros(5, dtype=np.uint8)
    else:
        peds = np.zeros((5, 5), dtype=np.float32)
        mask = np.zeros(5, dtype=np.uint8)
        for i, (x, y) in enumerate(ped_positions):
            peds[i] = (x, y, 0.0, 0.0, 0.5)
            mask[i] = 1
    lidar = np.full(36, min_lidar, dtype=np.float32)
    goal = np.array([d_goal, 0.0, 0.0, 0.0], dtype=np.float32)
    return {"lidar": lidar, "pedestrians": peds, "ped_mask": mask, "goal": goal}


# ----------------------------------------------------------------------
# Goal terminal
# ----------------------------------------------------------------------
def test_goal_terminal_reward(reward_cfg):
    obs = make_obs(d_goal=0.1)
    r = compute_reward(
        obs=obs, action=np.array([0.0, 0.0]),
        prev_action=np.zeros(2), prev_d_goal=2.0,
        terminated=True, info={"outcome": "success"}, cfg=reward_cfg,
    )
    assert r["r_goal"] == pytest.approx(200.0)
    assert r["r_collision"] == 0.0


def test_collision_terminal_reward(reward_cfg):
    obs = make_obs(d_goal=2.0, min_lidar=0.1)
    r = compute_reward(
        obs=obs, action=np.array([0.0, 0.0]),
        prev_action=np.zeros(2), prev_d_goal=2.0,
        terminated=True, info={"outcome": "collision"}, cfg=reward_cfg,
    )
    assert r["r_collision"] == pytest.approx(-250.0)
    assert r["r_goal"] == 0.0


# ----------------------------------------------------------------------
# Progress shaping (potential-based)
# ----------------------------------------------------------------------
def test_progress_positive_when_approaching(reward_cfg):
    obs = make_obs(d_goal=1.8)
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_progress"] == pytest.approx(1.0)


def test_progress_negative_when_receding(reward_cfg):
    obs = make_obs(d_goal=2.5)
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_progress"] == pytest.approx(-2.5)


def test_progress_zero_when_no_prev(reward_cfg):
    obs = make_obs(d_goal=2.0)
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=None,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_progress"] == 0.0


# ----------------------------------------------------------------------
# Proxemic penalty
# ----------------------------------------------------------------------
def test_prox_zero_when_no_pedestrian(reward_cfg):
    obs = make_obs(d_goal=2.0)
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_prox"] == 0.0
    assert r["d_min_ped"] == float("inf")


def test_prox_zero_when_pedestrian_outside_intimate_zone(reward_cfg):
    obs = make_obs(d_goal=2.0, ped_positions=[(1.0, 0.0)])
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_prox"] == 0.0


def test_prox_negative_when_intrusion(reward_cfg):
    """Pedestrian at 0.20 m -> intrusion 0.25, penalty = -10*0.25^2 = -0.625"""
    obs = make_obs(d_goal=2.0, ped_positions=[(0.20, 0.0)])
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_prox"] == pytest.approx(-0.625, abs=1e-6)
    assert r["d_min_ped"] == pytest.approx(0.20)


def test_prox_uses_nearest_pedestrian(reward_cfg):
    obs = make_obs(d_goal=2.0,
                   ped_positions=[(2.0, 0.0), (0.30, 0.0), (1.5, 0.0)])
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                      prev_action=np.zeros(2), prev_d_goal=2.0,
                      terminated=False, info={}, cfg=reward_cfg)
    expected = -10.0 * (0.45 - 0.30) ** 2
    assert r["r_prox"] == pytest.approx(expected, abs=1e-6)
    assert r["d_min_ped"] == pytest.approx(0.30)


# ----------------------------------------------------------------------
# Smoothness penalty
# ----------------------------------------------------------------------
def test_smooth_zero_when_action_unchanged(reward_cfg):
    obs = make_obs(d_goal=2.0)
    r = compute_reward(obs=obs, action=np.array([0.5, 0.2]),
                       prev_action=np.array([0.5, 0.2]), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_smooth"] == 0.0


def test_smooth_negative_on_jerk(reward_cfg):
    """|Δv| + |Δω| = 0.4 + 0.3 = 0.7 -> r_smooth = -0.05 * 0.7 = -0.035"""
    obs = make_obs(d_goal=2.0)
    r = compute_reward(obs=obs, action=np.array([0.4, 0.3]),
                       prev_action=np.array([0.0, 0.0]), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_smooth"] == pytest.approx(-0.035, abs=1e-6)


# ----------------------------------------------------------------------
# Time cost & total
# ----------------------------------------------------------------------
def test_time_cost_constant(reward_cfg):
    obs = make_obs(d_goal=2.0)
    r = compute_reward(obs=obs, action=np.array([0.0, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    assert r["r_time"] == pytest.approx(-0.01)


def test_total_is_sum_of_terms(reward_cfg):
    obs = make_obs(d_goal=1.8, ped_positions=[(0.30, 0.0)])
    r = compute_reward(obs=obs, action=np.array([0.4, 0.0]),
                       prev_action=np.zeros(2), prev_d_goal=2.0,
                       terminated=False, info={}, cfg=reward_cfg)
    expected = (r["r_goal"] + r["r_collision"] + r["r_progress"]
                + r["r_prox"] + r["r_smooth"] + r["r_time"])
    assert r["total"] == pytest.approx(expected, abs=1e-6)


def test_collision_penalty_larger_than_goal_reward(reward_cfg):
    """The architecture relies on |r_collision| > r_goal for risk aversion."""
    assert abs(reward_cfg.r_collision) > reward_cfg.r_goal


def test_min_pedestrian_distance_helper(reward_cfg):
    """The helper is also exposed at module level."""
    obs = make_obs(d_goal=2.0, ped_positions=[(1.0, 0.0), (0.5, 0.0)])
    assert min_pedestrian_distance(obs) == pytest.approx(0.5)
    obs_empty = make_obs(d_goal=2.0)
    assert min_pedestrian_distance(obs_empty) == float("inf")
