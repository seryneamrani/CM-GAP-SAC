"""Ground-truth pedestrian provider (ablation A5).

For ablation A5 we want to isolate the conceptual contribution of the
multimodal architecture from the noise of the YOLO+DeepSORT pipeline.
This module subscribes to the Gazebo Harmonic dynamic-pose stream (bridged
through ros_gz_bridge) and exposes the same `get_tracks()` API as the
real-perception tracker.

Implementation note:
    Gazebo Harmonic publishes dynamic poses on /world/<world>/dynamic_pose/info
    as gz.msgs.Pose_V. The ros_gz_bridge wraps this into tf2_msgs/TFMessage,
    which is what we subscribe to here.

    Actor names follow the convention used in our existing world files
    (pedestrian_0, pedestrian_1, ...). We filter by name prefix.
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, List

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from tf2_msgs.msg import TFMessage  # type: ignore

from cm_gap_sac_navigation.perception.pedestrian_tracker import _TrackHistory
from cm_gap_sac_navigation.perception.track_state import PedestrianTrack


class GroundTruthTracker(Node):
    """Publishes the same interface as PerceptionTracker but uses Gazebo GT."""

    PED_NAME_PREFIX = "pedestrian_"
    STALE_TIMEOUT_S = 0.5
    VEL_EMA_ALPHA = 0.7

    def __init__(
        self,
        topic: str,
        node_name: str = "cm_gap_sac_gt_tracker",
    ) -> None:
        super().__init__(node_name)

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._sub = self.create_subscription(
            TFMessage, topic, self._on_tf, qos
        )
        self._lock = Lock()
        self._histories: Dict[str, _TrackHistory] = {}   # keyed by actor name
        self._name_to_id: Dict[str, int] = {}
        self._next_id = 0

        self.get_logger().info(f"GroundTruthTracker subscribed to {topic}")

    def _on_tf(self, msg: TFMessage) -> None:
        # Take the timestamp from the first transform (they share a clock).
        if not msg.transforms:
            return
        stamp = msg.transforms[0].header.stamp
        stamp_s = stamp.sec + 1e-9 * stamp.nanosec

        with self._lock:
            for tf in msg.transforms:
                name = tf.child_frame_id
                if not name.startswith(self.PED_NAME_PREFIX):
                    continue

                xy = np.array(
                    [tf.transform.translation.x, tf.transform.translation.y],
                    dtype=np.float32,
                )

                if name in self._histories:
                    h = self._histories[name]
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
                    self._histories[name] = _TrackHistory(
                        last_xy=xy, last_t=stamp_s, age_frames=1,
                    )
                    self._name_to_id[name] = self._next_id
                    self._next_id += 1

            stale = [
                n for n, h in self._histories.items()
                if (stamp_s - h.last_t) > self.STALE_TIMEOUT_S
            ]
            for n in stale:
                del self._histories[n]

    def get_tracks(self) -> List[PedestrianTrack]:
        with self._lock:
            return [
                PedestrianTrack(
                    track_id=self._name_to_id[name],
                    position_world=h.last_xy.copy(),
                    velocity_world=h.velocity.copy(),
                    age_frames=h.age_frames,
                )
                for name, h in self._histories.items()
            ]

    def reset_tracks(self) -> None:
        with self._lock:
            self._histories.clear()
            # Keep _name_to_id stable across resets; actor names are persistent.
