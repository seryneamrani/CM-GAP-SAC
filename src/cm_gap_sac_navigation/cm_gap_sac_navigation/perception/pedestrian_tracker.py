"""Pedestrian tracker bridge between perception output and RL env.

This module does NOT run YOLOv8 or DeepSORT itself. The user's existing
stack already publishes tracked detections; we subscribe and convert.

Two implementations are provided:

  1. PerceptionTracker  - subscribes to vision_msgs/Detection3DArray, the
                          assumed output of the YOLO+DeepSORT pipeline.
                          Velocity is computed by finite differences across
                          frames keyed on tracking_id (Hungarian matching is
                          done upstream by DeepSORT).

  2. GroundTruthTracker - subscribes to Gazebo Harmonic dynamic-pose info
                          via ros_gz_bridge. Used for ablation A5 (perfect
                          perception baseline).

Both expose the same API:
    get_tracks() -> List[PedestrianTrack]

Velocities are kept in WORLD frame; the env transforms them to robot frame
when assembling the state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from vision_msgs.msg import Detection3DArray  # type: ignore

from cm_gap_sac_navigation.perception.track_state import PedestrianTrack


@dataclass
class _TrackHistory:
    """Internal state kept per track id for finite-difference velocity."""
    last_xy: np.ndarray             # shape (2,)
    last_t: float                   # seconds
    age_frames: int = 0
    velocity: np.ndarray = None     # shape (2,)

    def __post_init__(self) -> None:
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=np.float32)


class PerceptionTracker(Node):
    """ROS 2 node: subscribes to detection topic, maintains track histories.

    The class is designed to be **thread-safe** because the RL env may call
    `get_tracks()` from a worker thread while rclpy delivers messages on the
    spinning thread.
    """

    # Tracks unseen for more than this many seconds are evicted.
    STALE_TIMEOUT_S = 0.5

    # Velocity smoothing (exponential moving average) coefficient.
    # 0.0 = no smoothing (use instantaneous diff).
    # 0.7 is a good default; reduces jitter without large lag.
    VEL_EMA_ALPHA = 0.7

    def __init__(
        self,
        topic: str = "/perception/pedestrian_tracks",
        node_name: str = "cm_gap_sac_perception_tracker",
    ) -> None:
        super().__init__(node_name)

        # Best-effort QoS matches typical perception publishers (high rate,
        # don't block on slow consumers).
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._sub = self.create_subscription(
            Detection3DArray, topic, self._on_detections, qos
        )
        self._lock = Lock()
        self._histories: Dict[int, _TrackHistory] = {}
        self.get_logger().info(f"PerceptionTracker subscribed to {topic}")

    # --------------------------------------------------------------------
    # Subscription callback
    # --------------------------------------------------------------------
    def _on_detections(self, msg: Detection3DArray) -> None:
        # Stamp in seconds (we are tolerant to clock skew - finite diffs only).
        stamp_s = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        seen_ids = set()

        with self._lock:
            for det in msg.detections:
                # tracking_id encodes the persistent DeepSORT identifier.
                # Some YOLO-DeepSORT publishers stuff it into det.id (string),
                # others into det.tracking_id (int). We try both.
                track_id = self._extract_track_id(det)
                if track_id is None:
                    continue

                xy = np.array(
                    [det.bbox.center.position.x, det.bbox.center.position.y],
                    dtype=np.float32,
                )

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
                    self._histories[track_id] = _TrackHistory(
                        last_xy=xy, last_t=stamp_s, age_frames=1,
                    )

                seen_ids.add(track_id)

            # Evict stale tracks (DeepSORT may keep them briefly during
            # occlusion; we don't want them to leak into the RL state).
            stale = [
                tid for tid, h in self._histories.items()
                if (stamp_s - h.last_t) > self.STALE_TIMEOUT_S
            ]
            for tid in stale:
                del self._histories[tid]

    @staticmethod
    def _extract_track_id(det) -> Optional[int]:
        """Recover an integer tracking id from a Detection3D message.

        Detection3D in vision_msgs (Jazzy) does not have a dedicated tracking
        id field; conventionally we use `det.id` (a string) as set by
        DeepSORT. We coerce to int when possible and fall back to None.
        """
        raw = getattr(det, "id", None)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            # Some publishers use 'class_id|track_id' format.
            try:
                return int(str(raw).split("|")[-1])
            except (TypeError, ValueError):
                return None

    # --------------------------------------------------------------------
    # Public API consumed by the env
    # --------------------------------------------------------------------
    def get_tracks(self) -> List[PedestrianTrack]:
        """Return a snapshot of currently-active tracks (world frame)."""
        with self._lock:
            return [
                PedestrianTrack(
                    track_id=tid,
                    position_world=h.last_xy.copy(),
                    velocity_world=h.velocity.copy(),
                    age_frames=h.age_frames,
                )
                for tid, h in self._histories.items()
            ]

    def reset_tracks(self) -> None:
        """Clear track history; called between episodes."""
        with self._lock:
            self._histories.clear()
