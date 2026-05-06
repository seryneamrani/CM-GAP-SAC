"""Geometry utilities.

All functions are pure-numpy and stateless so they can be called from any
module (env step, perception, CBF shield) without ROS dependencies.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from a unit quaternion.

    Convention: ROS REP-103 (x forward, z up, yaw rotates around z).
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def world_to_robot(point_world: np.ndarray,
                   robot_xy: np.ndarray,
                   robot_yaw: float) -> np.ndarray:
    """Transform a 2D point from world frame to robot frame.

    Args:
        point_world: shape (2,) or (N, 2)
        robot_xy:    shape (2,)
        robot_yaw:   scalar

    Returns:
        same shape as point_world, expressed in robot frame.
    """
    delta = point_world - robot_xy
    c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return delta @ R.T


def velocity_world_to_robot(v_world: np.ndarray, robot_yaw: float) -> np.ndarray:
    """Rotate a velocity vector from world to robot frame.

    Translation does not apply (velocities are free vectors).
    """
    c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return v_world @ R.T


def downsample_lidar(ranges: np.ndarray,
                     n_beams: int,
                     range_min: float,
                     range_max: float) -> np.ndarray:
    """Downsample a raw 360-deg LiDAR scan to `n_beams` evenly-spaced beams.

    Strategy: split the full scan into `n_beams` angular sectors and take the
    minimum range in each sector. Minimum (rather than mean) is the safety-
    conservative choice: we never erase a closer obstacle.

    Args:
        ranges: raw LIDAR ranges (e.g., 720 for LIMO Pro). NaN/inf are clipped.
        n_beams: target resolution.
        range_min, range_max: clip bounds in metres.

    Returns:
        np.ndarray of shape (n_beams,), float32.
    """
    raw = np.asarray(ranges, dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, range_max)
    raw = np.clip(raw, range_min, range_max)

    n_raw = raw.shape[0]
    if n_raw == n_beams:
        return raw.copy()

    # Bucket by angle. Using np.array_split keeps this exact even when n_raw
    # is not a multiple of n_beams.
    buckets = np.array_split(raw, n_beams)
    return np.array([b.min() for b in buckets], dtype=np.float32)


def goal_polar_in_robot(robot_xy: np.ndarray,
                        robot_yaw: float,
                        goal_xy: np.ndarray) -> Tuple[float, float]:
    """Distance and bearing of the goal in the robot's frame.

    Returns:
        (d_goal, theta_goal) where theta_goal is wrapped to [-pi, pi].
    """
    dx = goal_xy[0] - robot_xy[0]
    dy = goal_xy[1] - robot_xy[1]
    d = float(np.hypot(dx, dy))
    theta_world = float(np.arctan2(dy, dx))
    theta_robot = wrap_to_pi(theta_world - robot_yaw)
    return d, theta_robot
