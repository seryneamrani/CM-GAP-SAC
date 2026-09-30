"""Tests for geometry utilities."""
import numpy as np
import pytest

from cm_gap_sac_navigation.utils.geometry import (
    downsample_lidar,
    goal_polar_in_robot,
    velocity_world_to_robot,
    world_to_robot,
    wrap_to_pi,
    yaw_from_quaternion,
)


def test_yaw_from_quaternion_zero():
    assert yaw_from_quaternion(0, 0, 0, 1) == pytest.approx(0.0)


def test_yaw_from_quaternion_90deg():
    # Rotation of pi/2 around z: q = (0, 0, sin(pi/4), cos(pi/4))
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    assert yaw_from_quaternion(0, 0, s, c) == pytest.approx(np.pi / 2, abs=1e-6)


def test_wrap_to_pi():
    assert wrap_to_pi(3 * np.pi) == pytest.approx(np.pi, abs=1e-6) or \
           wrap_to_pi(3 * np.pi) == pytest.approx(-np.pi, abs=1e-6)
    assert wrap_to_pi(0.5) == pytest.approx(0.5)
    assert wrap_to_pi(-0.5) == pytest.approx(-0.5)


def test_world_to_robot_identity():
    """Robot at origin facing forward: world == robot frame."""
    p_w = np.array([1.0, 2.0], dtype=np.float32)
    p_r = world_to_robot(p_w, robot_xy=np.zeros(2, dtype=np.float32),
                         robot_yaw=0.0)
    np.testing.assert_allclose(p_r, [1.0, 2.0], atol=1e-6)


def test_world_to_robot_90deg():
    """Robot at origin facing +y (yaw=pi/2): a point at (1, 0) world is at (0, -1) robot."""
    p_w = np.array([1.0, 0.0], dtype=np.float32)
    p_r = world_to_robot(p_w, robot_xy=np.zeros(2, dtype=np.float32),
                         robot_yaw=np.pi / 2)
    np.testing.assert_allclose(p_r, [0.0, -1.0], atol=1e-6)


def test_velocity_world_to_robot_no_translation():
    """Velocities only rotate, never translate."""
    v_w = np.array([1.0, 0.0], dtype=np.float32)
    v_r = velocity_world_to_robot(v_w, robot_yaw=np.pi / 2)
    np.testing.assert_allclose(v_r, [0.0, -1.0], atol=1e-6)


def test_downsample_lidar_basic():
    raw = np.linspace(0.5, 5.0, 720, dtype=np.float32)
    out = downsample_lidar(raw, n_beams=36, range_min=0.05, range_max=6.0)
    assert out.shape == (36,)
    # Each output beam is the MIN of its bucket; for a monotonically
    # increasing scan, beam i is the first element of bucket i.
    assert out[0] == pytest.approx(raw[0], abs=1e-3)


def test_downsample_lidar_clips_inf():
    raw = np.full(720, np.inf, dtype=np.float32)
    out = downsample_lidar(raw, n_beams=36, range_min=0.05, range_max=6.0)
    assert out.shape == (36,)
    np.testing.assert_array_equal(out, np.full(36, 6.0, dtype=np.float32))


def test_goal_polar_in_robot_straight_ahead():
    robot = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([3.0, 0.0], dtype=np.float32)
    d, theta = goal_polar_in_robot(robot, robot_yaw=0.0, goal_xy=goal)
    assert d == pytest.approx(3.0)
    assert theta == pytest.approx(0.0, abs=1e-6)


def test_goal_polar_in_robot_to_the_left():
    """Robot facing +x, goal at (0, 1) world -> bearing pi/2 in robot frame."""
    robot = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([0.0, 1.0], dtype=np.float32)
    d, theta = goal_polar_in_robot(robot, robot_yaw=0.0, goal_xy=goal)
    assert d == pytest.approx(1.0)
    assert theta == pytest.approx(np.pi / 2, abs=1e-6)
