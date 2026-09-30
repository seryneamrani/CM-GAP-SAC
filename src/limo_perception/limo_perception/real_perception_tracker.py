"""Real-perception pedestrian tracker (LiDAR MVP, pose-agnostic).

Mirrors the public lifecycle of GroundTruthTracker / PerceptionTracker
(get_tracks + reset_tracks) but does NOT know the robot's world pose.
By design (Option C): this node produces tracks in the ROBOT frame only.
The env owns the world transform (GT pose in sim, AMCL/SLAM on hardware),
so the sim-to-real seam stays in exactly one place.

Pipeline
--------
    /tracked_obstacles (vision_msgs/Detection2DArray, pixel bboxes + DeepSORT id)
    /scan              (sensor_msgs/LaserScan)
            |
            v
    for each confirmed detection:
        bbox horizontal center  --bbox_lidar_fusion-->  (x, y) robot frame
        keyed by DeepSORT id, finite-difference + EMA  ->  (vx, vy) robot frame
            |
            v
    get_tracks_robot_frame() -> List[TrackRobot]

Why a separate dataclass (TrackRobot) instead of PedestrianTrack?
    PedestrianTrack carries position_world / velocity_world (the env's
    contract). This node has no world frame. Returning a robot-frame
    dataclass makes the frame explicit in the type and prevents anyone
    from mistakenly feeding robot-frame numbers into a world-frame slot.
    The env converts TrackRobot -> PedestrianTrack in one place.

Threading
    rclpy delivers messages on the executor's spin thread; the env calls
    get_tracks_robot_frame() from the step thread. All shared state is
    guarded by a single lock, same pattern as the other trackers.

ADAPT points (search "ADAPT-"):
    ADAPT-CAM   : camera FoV / image width / yaw offset. Defaults match a
                  640x480 70-deg-FoV cam; override via constructor.
    ADAPT-TOPIC : topic names. Sim vs real differ; pass via constructor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray  # type: ignore

from limo_perception.bbox_lidar_fusion import (
    CameraGeometry,
    LidarGeometry,
    bbox_to_robot_xy,
)


@dataclass
class TrackRobot:
    """A single track in the ROBOT frame (NOT world).

    The env converts this to PedestrianTrack (world) before handing it to
    assemble_pedestrian_state.
    """
    track_id: int
    position_robot: np.ndarray   # (2,) [x, y] robot frame
    velocity_robot: np.ndarray   # (2,) [vx, vy] robot frame
    age_frames: int


@dataclass
class _RobotTrackHistory:
    """Per-id state for finite-difference velocity in the robot frame."""
    last_xy: np.ndarray          # (2,)
    last_t: float                # seconds
    age_frames: int = 0
    velocity: np.ndarray = field(default=None)  # (2,)

    def __post_init__(self) -> None:
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=np.float32)


class _LatestScan:
    """Thread-safe holder for the most recent LaserScan + its geometry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ranges: Optional[np.ndarray] = None
        self._geom: Optional[LidarGeometry] = None
        self._stamp_s: float = 0.0

    def update(self, msg: LaserScan) -> None:
        with self._lock:
            self._ranges = np.asarray(msg.ranges, dtype=np.float32)
            # Read geometry from the live message; never hardcode it.
            self._geom = LidarGeometry(
                angle_min=float(msg.angle_min),
                angle_increment=float(msg.angle_increment),
                n_beams=len(msg.ranges),
            )
            self._stamp_s = (msg.header.stamp.sec
                             + 1e-9 * msg.header.stamp.nanosec)

    def get(self):
        with self._lock:
            if self._ranges is None:
                return None, None
            return self._ranges.copy(), self._geom


class RealPerceptionTracker(Node):
    """Pose-agnostic tracker: pixel bboxes + LiDAR -> robot-frame tracks.

    Velocity smoothing and stale-track eviction mirror PerceptionTracker so
    the dynamics seen by the policy are consistent across tracker_modes.
    """

    STALE_TIMEOUT_S = 0.5
    VEL_EMA_ALPHA = 0.7

    def __init__(
        self,
        # ADAPT-TOPIC ---------------------------------------------------
        tracks_topic: str = "/tracked_obstacles",
        scan_topic: str = "/scan",
        # ADAPT-CAM -----------------------------------------------------
        image_width: int = 640,
        h_fov_deg: float = 70.0,
        cam_yaw_offset: float = 0.0,
        # LiDAR read window / clipping ----------------------------------
        window_halfwidth: int = 2,
        range_min: float = 0.05,
        range_max: float = 12.0,
        node_name: str = "cm_gap_sac_real_perception_tracker",
    ) -> None:
        super().__init__(node_name)

        self._cam = CameraGeometry(
            image_width=image_width,
            h_fov=np.deg2rad(h_fov_deg),
            cam_yaw_offset=cam_yaw_offset,
        )
        self._win = int(window_halfwidth)
        self._range_min = float(range_min)
        self._range_max = float(range_max)

        # Best-effort QoS: high-rate sensor/detection publishers, don't
        # block. Matches perception_node's RELIABLE image sub upstream, but
        # for OUR subscriptions best-effort is the safe high-rate default.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._scan = _LatestScan()
        self.create_subscription(LaserScan, scan_topic, self._scan.update, qos)
        self.create_subscription(
            Detection2DArray, tracks_topic, self._on_tracks, qos
        )

        self._lock = Lock()
        self._histories: Dict[int, _RobotTrackHistory] = {}

        self._n_dropped_no_range = 0  # diagnostics: bboxes with no LiDAR hit
        self.get_logger().info(
            f"RealPerceptionTracker up: tracks={tracks_topic}, scan={scan_topic}, "
            f"FoV={h_fov_deg:.1f} deg, img_w={image_width}, "
            f"yaw_offset={cam_yaw_offset:.3f} rad"
        )

    # ------------------------------------------------------------------
    # Detection callback
    # ------------------------------------------------------------------
    def _on_tracks(self, msg: Detection2DArray) -> None:
        ranges, geom = self._scan.get()
        if ranges is None or geom is None:
            return  # no LiDAR yet; nothing to fuse against

        stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

        with self._lock:
            for det in msg.detections:
                track_id = self._extract_track_id(det)
                if track_id is None:
                    continue

                u_center = float(det.bbox.center.position.x)
                xy = bbox_to_robot_xy(
                    u_center=u_center,
                    ranges=ranges,
                    cam=self._cam,
                    lidar=geom,
                    window_halfwidth=self._win,
                    range_min=self._range_min,
                    range_max=self._range_max,
                )
                if xy is None:
                    # No valid LiDAR return at that bearing (occlusion /
                    # max range). Drop this frame for this track; keep its
                    # history so it survives a brief dropout.
                    self._n_dropped_no_range += 1
                    continue

                if track_id in self._histories:
                    h = self._histories[track_id]
                    dt = max(stamp_s - h.last_t, 1e-3)
                    inst_v = (xy - h.last_xy) / dt
                    h.velocity = (
                        self.VEL_EMA_ALPHA * h.velocity
                        + (1.0 - self.VEL_EMA_ALPHA) * inst_v
                    )
                    h.last_xy = xy
                    h.last_t = stamp_s
                    h.age_frames += 1
                else:
                    self._histories[track_id] = _RobotTrackHistory(
                        last_xy=xy, last_t=stamp_s, age_frames=1,
                    )

            # Evict stale tracks (DeepSORT keeps occluded ids briefly).
            stale = [
                tid for tid, h in self._histories.items()
                if (stamp_s - h.last_t) > self.STALE_TIMEOUT_S
            ]
            for tid in stale:
                del self._histories[tid]

    @staticmethod
    def _extract_track_id(det) -> Optional[int]:
        """Recover an int DeepSORT id from a Detection2D.

        perception_node sets det.id = str(track_id). We coerce to int and
        fall back to a 'class|id' split, matching PerceptionTracker so the
        two implementations behave identically on the same upstream.
        """
        raw = getattr(det, "id", None)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            try:
                return int(str(raw).split("|")[-1])
            except (TypeError, ValueError):
                return None

    # ------------------------------------------------------------------
    # Public API (consumed by the env)
    # ------------------------------------------------------------------
    def get_tracks_robot_frame(self) -> List[TrackRobot]:
        """Snapshot of active tracks, ROBOT frame."""
        with self._lock:
            return [
                TrackRobot(
                    track_id=tid,
                    position_robot=h.last_xy.copy(),
                    velocity_robot=h.velocity.copy(),
                    age_frames=h.age_frames,
                )
                for tid, h in self._histories.items()
            ]

    def reset_tracks(self) -> None:
        """Clear histories between episodes.

        NOTE: this clears only OUR finite-difference state. The DeepSORT
        tracker itself lives in perception_node (a separate process); its
        ids are NOT reset here. Across sim episodes that is fine because
        ids only ever grow (no correctness issue, just non-zero starting
        ids). If you ever co-locate DeepSORT in this process, reset it here
        too (README ADAPT-C: leaking ids corrupts age_frames).
        """
        with self._lock:
            self._histories.clear()


# ----------------------------------------------------------------------
# Standalone smoke test (no ROS spin): inject fake messages, check tracks.
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    """Construct the node, feed it synthetic scan + detections, verify.

    Runs without a live ROS graph: we call the callbacks directly with
    hand-built messages. This checks the fusion + tracking wiring end to
    end (id keying, velocity sign, robot-frame output) before touching
    the simulator.
    """
    import types

    rclpy.init()
    node = RealPerceptionTracker(
        image_width=640, h_fov_deg=70.0, cam_yaw_offset=0.0,
    )

    n = 720

    def make_scan(target_idx, r, t):
        msg = LaserScan()
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
        msg.angle_min = -np.pi
        msg.angle_increment = 2.0 * np.pi / n
        ranges = [10.0] * n
        for k in (-2, -1, 0, 1, 2):
            ranges[(target_idx + k) % n] = r
        msg.ranges = ranges
        return msg

    def make_det(u, track_id, t):
        from vision_msgs.msg import Detection2D
        arr = Detection2DArray()
        arr.header.stamp.sec = int(t)
        arr.header.stamp.nanosec = int((t - int(t)) * 1e9)
        d = Detection2D()
        d.id = str(track_id)
        d.bbox.center.position.x = float(u)
        arr.detections.append(d)
        return arr

    # Frame 1: person dead-ahead at 2.0 m, id 7.
    ahead = (0 - (-np.pi)) / (2 * np.pi / n)
    ahead_idx = int(round(ahead)) % n
    node._scan.update(make_scan(ahead_idx, 2.0, t=0.0))
    node._on_tracks(make_det(u=320.0, track_id=7, t=0.0))

    tr = node.get_tracks_robot_frame()
    assert len(tr) == 1, f"expected 1 track, got {len(tr)}"
    assert tr[0].track_id == 7
    p = tr[0].position_robot
    print(f"  frame1: id7 pos_robot = {p}  (expect ~[2.0, 0.0])")
    assert abs(p[0] - 2.0) < 0.05 and abs(p[1]) < 0.05

    # Frame 2: same person now at 1.5 m, 0.1 s later -> approaching,
    # vx should be negative (closing distance along +x).
    node._scan.update(make_scan(ahead_idx, 1.5, t=0.1))
    node._on_tracks(make_det(u=320.0, track_id=7, t=0.1))
    tr = node.get_tracks_robot_frame()
    v = tr[0].velocity_robot
    print(f"  frame2: id7 vel_robot = {v}  (expect vx < 0, approaching)")
    assert v[0] < 0, f"approaching target must have vx<0, got {v[0]}"
    assert tr[0].age_frames == 2

    # reset clears histories.
    node.reset_tracks()
    assert len(node.get_tracks_robot_frame()) == 0
    print("  reset -> 0 tracks  (correct)")

    node.destroy_node()
    rclpy.shutdown()
    print("[real_perception_tracker] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
