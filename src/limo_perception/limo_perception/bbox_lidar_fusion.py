"""Bbox <-> LiDAR fusion (MVP, ray-casting).

Pure-numpy, zero ROS. Converts a YOLO bbox (pixel coordinates) into a
metric (x, y) position in the ROBOT frame, by casting the bbox horizontal
center to a LiDAR bearing and reading the range at that bearing.

This is the cheap, sim-to-real-friendly path: no depth camera, no TF,
no camera-LiDAR extrinsic calibration beyond a single yaw offset and the
camera horizontal FoV. The Orbbec Dabai (RGB-D) can replace this later
with a depth-based projector; the OUTPUT CONTRACT stays identical.

Geometry, step by step
-----------------------
1. Pixel u (horizontal center of the bbox) -> normalized [-0.5, +0.5].
2. * camera horizontal FoV -> angle of the target in the CAMERA frame.
   Sign convention: image u increases to the RIGHT; in ROS REP-103 the
   robot's +y (and positive bearing) is to the LEFT. So a target on the
   right of the image has a NEGATIVE bearing. We flip the sign.
3. + cam_yaw_offset -> bearing in the ROBOT frame (the camera may not be
   perfectly aligned with base_link; this scalar absorbs it).
4. bearing -> nearest LiDAR beam index -> range r.
   We take the MINIMUM range over a small angular window around the
   bearing (robust to a single bad beam, and safety-conservative: a
   pedestrian's nearest surface is what matters for avoidance).
5. (r, bearing) -> (x, y) = (r cos bearing, r sin bearing) in robot frame.

What this module does NOT do
----------------------------
- No tracking, no velocity, no world frame. That is the tracker's job.
- No camera intrinsics (fx, fy, cx, cy). For the LiDAR MVP we only need
  the horizontal FoV and the image width. Depth-based projection (later)
  is where intrinsics come back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LidarGeometry:
    """Describes how to index a raw LaserScan by bearing.

    These three numbers come straight from the sensor_msgs/LaserScan
    header fields (angle_min, angle_increment) and len(ranges). Capture
    them once from the live topic; they are constant for a given LiDAR.

    For the EAI T-mini Pro on the LIMO Pro, a 360-deg scan is typical, so
    angle_min ~= -pi and the increment ~= 2*pi / n_beams. But DO NOT assume:
    read them from the actual message (see RealPerceptionTracker).
    """
    angle_min: float          # rad, bearing of ranges[0]
    angle_increment: float    # rad, bearing step between consecutive beams
    n_beams: int              # len(ranges)

    def bearing_to_index(self, bearing: float) -> int:
        """Nearest beam index for a given bearing (rad), wrapped into range."""
        idx = int(round((bearing - self.angle_min) / self.angle_increment))
        # Wrap: 360-deg LiDARs are circular, so a bearing just past the last
        # beam belongs near the first. Modulo keeps us in [0, n_beams).
        return idx % self.n_beams


@dataclass
class CameraGeometry:
    """Minimal camera model for the LiDAR-MVP (no intrinsics needed).

    image_width:    horizontal resolution in pixels.
    h_fov:          horizontal field of view in radians.
    cam_yaw_offset: rad, added to the camera-frame bearing to get the
                    robot-frame bearing. Calibrate with the 2 m test:
                    put a person dead-ahead, adjust until y ~= 0.
    """
    image_width: int
    h_fov: float
    cam_yaw_offset: float = 0.0


def bbox_center_to_bearing(
    u_center: float,
    cam: CameraGeometry,
) -> float:
    """Pixel horizontal center -> bearing in the ROBOT frame (rad).

    See module docstring for the sign convention. Result is NOT wrapped;
    the caller wraps when indexing the LiDAR.
    """
    # 1. normalized horizontal position, [-0.5, +0.5], 0 = image center.
    u_norm = (u_center / float(cam.image_width)) - 0.5
    # 2. camera-frame angle. Negative sign: image-right -> robot-right (-y).
    angle_cam = -u_norm * cam.h_fov
    # 3. into robot frame.
    return angle_cam + cam.cam_yaw_offset


def range_at_bearing(
    ranges: np.ndarray,
    lidar: LidarGeometry,
    bearing: float,
    window_halfwidth: int = 2,
    range_min: float = 0.05,
    range_max: float = 12.0,
) -> Optional[float]:
    """Robust range read at a bearing: min over a small beam window.

    Returns None if no valid beam falls in the window (all NaN/inf or out
    of [range_min, range_max]) -- the caller then drops the detection.

    window_halfwidth is in BEAMS, not radians: +/- this many beams around
    the central index. 2 beams on a 360-deg/720-beam LiDAR is ~1 deg each
    side, enough to dodge a single dropout without smearing onto a
    neighbouring object.
    """
    center = lidar.bearing_to_index(bearing)
    idxs = [(center + k) % lidar.n_beams
            for k in range(-window_halfwidth, window_halfwidth + 1)]
    window = np.asarray([ranges[i] for i in idxs], dtype=np.float32)

    valid = window[
        np.isfinite(window) & (window >= range_min) & (window <= range_max)
    ]
    if valid.size == 0:
        return None
    return float(valid.min())


def bbox_to_robot_xy(
    u_center: float,
    ranges: np.ndarray,
    cam: CameraGeometry,
    lidar: LidarGeometry,
    window_halfwidth: int = 2,
    range_min: float = 0.05,
    range_max: float = 12.0,
) -> Optional[np.ndarray]:
    """Full pipeline: bbox horizontal center + scan -> (x, y) robot frame.

    Returns a (2,) float32 array, or None if the LiDAR has no valid return
    at that bearing (occlusion, max range, sensor dropout).
    """
    bearing = bbox_center_to_bearing(u_center, cam)
    r = range_at_bearing(
        ranges, lidar, bearing,
        window_halfwidth=window_halfwidth,
        range_min=range_min, range_max=range_max,
    )
    if r is None:
        return None
    x = r * np.cos(bearing)
    y = r * np.sin(bearing)
    return np.array([x, y], dtype=np.float32)


# ----------------------------------------------------------------------
# Standalone self-test: synthetic scan, known target, assert recovery.
# ----------------------------------------------------------------------
def _self_test() -> None:
    print("[bbox_lidar_fusion] self-test")

    # A 720-beam, 360-deg LiDAR. ranges[0] at -pi, going CCW.
    n = 720
    lidar = LidarGeometry(
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / n,
        n_beams=n,
    )
    # 640-wide image, 70-deg horizontal FoV, no yaw offset.
    cam = CameraGeometry(
        image_width=640,
        h_fov=np.deg2rad(70.0),
        cam_yaw_offset=0.0,
    )

    # Build a scan that is all "far" except a target dead-ahead at 2.0 m.
    # Dead-ahead in robot frame = bearing 0 = index for bearing 0.
    ranges = np.full(n, 10.0, dtype=np.float32)
    ahead_idx = lidar.bearing_to_index(0.0)
    for k in (-2, -1, 0, 1, 2):
        ranges[(ahead_idx + k) % n] = 2.0

    # Target at image center -> should be dead ahead -> x~2, y~0.
    xy = bbox_to_robot_xy(u_center=320.0, ranges=ranges, cam=cam, lidar=lidar)
    assert xy is not None, "expected a valid return dead-ahead"
    print(f"  center bbox -> robot xy = {xy}  (expect ~[2.0, 0.0])")
    assert abs(xy[0] - 2.0) < 0.05, f"x off: {xy[0]}"
    assert abs(xy[1] - 0.0) < 0.05, f"y off: {xy[1]}"

    # Target on the RIGHT of the image -> negative y (robot's right).
    # Put a 3 m return at the bearing the right-edge maps to.
    cam_right_bearing = bbox_center_to_bearing(640.0, cam)  # far right pixel
    print(f"  right-edge pixel -> bearing = {np.rad2deg(cam_right_bearing):.1f} deg "
          f"(expect ~ -35 deg)")
    assert cam_right_bearing < 0, "right of image must be negative bearing"

    # Verify a dropout returns None rather than a bogus point.
    empty = np.full(n, np.inf, dtype=np.float32)
    none_xy = bbox_to_robot_xy(u_center=320.0, ranges=empty, cam=cam, lidar=lidar)
    assert none_xy is None, "all-inf scan must yield None"
    print("  dropout scan -> None  (correct)")

    print("[bbox_lidar_fusion] all assertions passed.")


if __name__ == "__main__":
    _self_test()
