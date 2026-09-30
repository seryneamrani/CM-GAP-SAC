"""Ground-truth pedestrian provider (ablation A5).

Subscribes to per-pedestrian Odometry published by pedestrian_manager
on /pedestrian/<name>/state. Position is the real Gazebo pose; velocity
is the exact commanded velocity (no finite-difference, no EMA lag).
Replaces the previous dynamic_pose/info path, which published empty
child_frame_id and could not identify pedestrians by name.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Dict, List

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry

from cm_gap_sac_navigation.perception.track_state import PedestrianTrack


class GroundTruthTracker(Node):
    """GT pedestrian tracks from pedestrian_manager Odometry topics."""

    PED_NAMES = [f"ped_{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8)]
    STALE_TIMEOUT_S = 0.5

    def __init__(self, topic: str = "", node_name: str = "cm_gap_sac_gt_tracker"):
        super().__init__(node_name)
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._lock = Lock()
        self._state: Dict[str, dict] = {}   # name -> {xy, vel, t, age}
        self._name_to_id = {n: i for i, n in enumerate(self.PED_NAMES)}

        for name in self.PED_NAMES:
            self.create_subscription(
                Odometry, f"/pedestrian/{name}/state",
                self._make_cb(name), qos)

        self.get_logger().info(
            f"GroundTruthTracker subscribed to /pedestrian/<name>/state "
            f"for {self.PED_NAMES}")

    def _make_cb(self, name: str):
        def cb(msg: Odometry):
            xy = np.array([msg.pose.pose.position.x,
                           msg.pose.pose.position.y], dtype=np.float32)
            vel = np.array([msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y], dtype=np.float32)
            with self._lock:
                s = self._state.get(name)
                age = (s["age"] + 1) if s else 1
                self._state[name] = {
                    "xy": xy, "vel": vel,
                    "t": time.monotonic(), "age": age,
                }
        return cb

    def get_tracks(self) -> List[PedestrianTrack]:
        now = time.monotonic()
        with self._lock:
            tracks = []
            for name, s in self._state.items():
                if (now - s["t"]) > self.STALE_TIMEOUT_S:
                    continue
                tracks.append(PedestrianTrack(
                    track_id=self._name_to_id[name],
                    position_world=s["xy"].copy(),
                    velocity_world=s["vel"].copy(),
                    age_frames=s["age"],
                ))
            return tracks

    def reset_tracks(self) -> None:
        with self._lock:
            self._state.clear()