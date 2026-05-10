"""Tests for the CBF safety shield."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cm_gap_sac_navigation.rl.safety_shield import (
    CbfSafetyShield,
    ShieldResult,
    build_shield_from_config,
)
from cm_gap_sac_navigation.utils.config_loader import load_config


@pytest.fixture
def shield():
    return CbfSafetyShield(
        r_safe=0.30, gamma=2.0, active_radius=2.0,
        v_min=-0.3, v_max=0.6, omega_min=-1.0, omega_max=1.0,
    )


# ----------------------------------------------------------------------
# Trivial cases
# ----------------------------------------------------------------------
def test_no_obstacles_no_modification(shield):
    """No lidar, no pedestrians, action within bounds -> shield is identity."""
    u_sac = np.array([0.4, 0.2])
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        lidar_scan=None,
        pedestrians_rel=None,
        ped_mask=None,
    )
    assert isinstance(result, ShieldResult)
    np.testing.assert_allclose(result.safe_action, u_sac, atol=1e-3)
    assert not result.was_modified
    assert result.n_active_constraints == 0
    assert not result.infeasible


def test_action_outside_bounds_clipped(shield):
    """If u_sac is outside the action box, the QP enforces the box."""
    u_sac = np.array([5.0, 5.0])  # way outside
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
    )
    # v should be clipped to v_max=0.6, omega to omega_max=1.0
    assert result.safe_action[0] <= 0.6 + 1e-3
    assert result.safe_action[1] <= 1.0 + 1e-3
    assert result.was_modified


def test_far_obstacles_dont_constrain(shield):
    """Lidar shows far obstacles only -> no active constraints, action unchanged."""
    # Uniform far lidar (5 m everywhere, well beyond active_radius=2.0)
    lidar = np.full(36, 5.0, dtype=np.float32)
    u_sac = np.array([0.4, 0.0])
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        lidar_scan=lidar,
    )
    assert result.n_active_constraints == 0
    np.testing.assert_allclose(result.safe_action, u_sac, atol=1e-3)


# ----------------------------------------------------------------------
# Active filtering
# ----------------------------------------------------------------------
def test_close_frontal_obstacle_reduces_velocity():
    """Wall close in front -> shield must reduce forward velocity."""
    shield = CbfSafetyShield(
        r_safe=0.30, gamma=2.0, active_radius=2.0,
        v_min=-0.3, v_max=0.6, omega_min=-1.0, omega_max=1.0,
    )
    # Lidar with obstacle at 0.40 m straight ahead (index 0).
    lidar = np.full(36, 5.0, dtype=np.float32)
    lidar[0] = 0.40

    u_sac = np.array([0.6, 0.0])  # full speed forward
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        lidar_scan=lidar,
    )
    assert result.n_active_constraints >= 1
    # Forward velocity must be reduced.
    assert result.safe_action[0] < u_sac[0]
    assert result.was_modified


def test_pedestrian_in_intimate_zone_blocks_approach():
    """Pedestrian very close in front -> shield should block forward motion."""
    shield = CbfSafetyShield(
        r_safe=0.30, gamma=2.0, active_radius=2.0,
        v_min=-0.3, v_max=0.6, omega_min=-1.0, omega_max=1.0,
    )
    # One pedestrian at 0.35 m straight ahead, not moving.
    peds = np.zeros((5, 5), dtype=np.float32)
    peds[0] = (0.35, 0.0, 0.0, 0.0, 0.5)
    mask = np.array([1, 0, 0, 0, 0], dtype=np.uint8)

    u_sac = np.array([0.6, 0.0])  # full speed forward
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        pedestrians_rel=peds,
        ped_mask=mask,
    )
    assert result.n_active_constraints >= 1
    assert result.safe_action[0] < u_sac[0]
    assert result.was_modified


def test_obstacle_behind_doesnt_block_forward(shield):
    """Obstacle behind robot -> can still move forward."""
    lidar = np.full(36, 5.0, dtype=np.float32)
    # Obstacle at 0.40 m directly behind (index 18 = 180 degrees).
    lidar[18] = 0.40

    u_sac = np.array([0.5, 0.0])
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        lidar_scan=lidar,
    )
    # Forward motion should be allowed (h_dot allows moving away from
    # obstacle behind).
    np.testing.assert_allclose(result.safe_action, u_sac, atol=0.05)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
def test_action_dimensions(shield):
    """Output is always (2,) float32."""
    result = shield.filter(
        u_sac=np.array([0.0, 0.0]),
        robot_xy_yaw=(0.0, 0.0, 0.0),
    )
    assert result.safe_action.shape == (2,)
    assert result.safe_action.dtype == np.float32
    assert result.delta_action.shape == (2,)


def test_omega_passes_through(shield):
    """Position-only CBF: omega should not be filtered."""
    u_sac = np.array([0.0, 0.5])  # zero v, nonzero omega
    result = shield.filter(
        u_sac=u_sac,
        robot_xy_yaw=(0.0, 0.0, 0.0),
        lidar_scan=np.full(36, 0.5, dtype=np.float32),  # obstacles all around
    )
    # omega component should be approximately preserved since v=0 satisfies
    # all constraints regardless.
    np.testing.assert_allclose(result.safe_action[1], 0.5, atol=0.05)


def test_pedestrian_with_velocity_taken_into_account():
    """Moving pedestrian: shield should use velocity to compute h_dot."""
    shield = CbfSafetyShield(r_safe=0.30, gamma=2.0, active_radius=2.0)
    # Pedestrian at 0.50 m ahead, moving towards robot at 0.5 m/s
    peds_approaching = np.zeros((5, 5), dtype=np.float32)
    peds_approaching[0] = (0.50, 0.0, -0.5, 0.0, 0.5)  # vx_rel = -0.5
    mask = np.array([1, 0, 0, 0, 0], dtype=np.uint8)

    # Same pedestrian but stationary
    peds_static = peds_approaching.copy()
    peds_static[0, 2] = 0.0

    u_sac = np.array([0.4, 0.0])
    result_approach = shield.filter(
        u_sac=u_sac, robot_xy_yaw=(0.0, 0.0, 0.0),
        pedestrians_rel=peds_approaching, ped_mask=mask,
    )
    result_static = shield.filter(
        u_sac=u_sac, robot_xy_yaw=(0.0, 0.0, 0.0),
        pedestrians_rel=peds_static, ped_mask=mask,
    )
    # Approaching pedestrian should be more constraining than static one.
    assert result_approach.safe_action[0] <= result_static.safe_action[0] + 1e-3


# ----------------------------------------------------------------------
# Build from config
# ----------------------------------------------------------------------
def test_build_shield_from_config_enabled():
    cfg_path = Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml"
    cfg = load_config(cfg_path)
    shield = build_shield_from_config(cfg)
    assert shield is not None
    assert isinstance(shield, CbfSafetyShield)
    assert shield.r_safe == cfg.cbf["r_safe"]


def test_build_shield_from_config_disabled():
    cfg_path = Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml"
    cfg = load_config(cfg_path)
    cfg.cbf["enabled"] = False
    shield = build_shield_from_config(cfg)
    assert shield is None
