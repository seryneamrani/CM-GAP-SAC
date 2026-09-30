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



def robot_to_world(point_robot: np.ndarray,
                   robot_xy: np.ndarray,
                   robot_yaw: float) -> np.ndarray:
    """Transform a 2D point from ROBOT frame to WORLD frame.

    Exact inverse of world_to_robot. Accepts (2,) or (N, 2).

    Args:
        point_robot: shape (2,) or (N, 2), expressed in robot frame
        robot_xy:    shape (2,), robot position in world
        robot_yaw:   scalar, robot heading in world

    Returns:
        same shape as point_robot, expressed in world frame.
    """
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)   # R(+yaw)
    rotated = point_robot @ R.T
    return rotated + robot_xy


def velocity_robot_to_world(v_robot: np.ndarray, robot_yaw: float) -> np.ndarray:
    """Rotate a velocity vector from ROBOT frame to WORLD frame.

    Exact inverse of velocity_world_to_robot. No translation (free vector).
    """
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)   # R(+yaw)
    return v_robot @ R.T


# =====================================================================
#  TEST round-trip (à lancer une fois, puis supprimer si tu veux):
#      python3 geometry.py
#  Vérifie que robot_to_world ∘ world_to_robot == identité.
# =====================================================================
def _roundtrip_test() -> None:
    from numpy.random import default_rng

    # Réimplémentation locale de world_to_robot pour un test autonome
    # (dans ton fichier réel, importe-la directement).
    def world_to_robot(point_world, robot_xy, robot_yaw):
        delta = point_world - robot_xy
        c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return delta @ R.T

    def velocity_world_to_robot(v_world, robot_yaw):
        c, s = np.cos(-robot_yaw), np.sin(-robot_yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return v_world @ R.T

    rng = default_rng(0)
    for _ in range(1000):
        p_world = rng.uniform(-10, 10, size=(2,)).astype(np.float32)
        v_world = rng.uniform(-3, 3, size=(2,)).astype(np.float32)
        robot_xy = rng.uniform(-10, 10, size=(2,)).astype(np.float32)
        yaw = float(rng.uniform(-np.pi, np.pi))

        p_r = world_to_robot(p_world, robot_xy, yaw)
        p_back = robot_to_world(p_r, robot_xy, yaw)
        assert np.allclose(p_back, p_world, atol=1e-4), (p_back, p_world)

        v_r = velocity_world_to_robot(v_world, yaw)
        v_back = velocity_robot_to_world(v_r, yaw)
        assert np.allclose(v_back, v_world, atol=1e-4), (v_back, v_world)

    # Batched (N,2) form too.
    pts_r = rng.uniform(-5, 5, size=(8, 2)).astype(np.float32)
    rxy = np.array([1.0, -2.0], dtype=np.float32)
    yaw = 0.7
    pts_w = robot_to_world(pts_r, rxy, yaw)
    pts_back = world_to_robot(pts_w, rxy, yaw)
    assert np.allclose(pts_back, pts_r, atol=1e-4)

    print("[geometry] robot_to_world round-trip: 1000 scalar + batched OK")


if __name__ == "__main__":
    _roundtrip_test()
