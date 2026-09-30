"""Gymnasium environment wrapping Gazebo Harmonic for CM-GAP_SAC training (v0.2 stable).

The environment owns:
    - One GazeboInterface node (sensors: scan, odom, imu + cmd_vel + sim control)
    - One pedestrian tracker (PerceptionTracker or GroundTruthTracker)
    - A background thread that spins both nodes via a MultiThreadedExecutor

Observation space (Dict to keep modalities cleanly separated):
    lidar      : Box(n_beams,)         downsampled scan, clipped to range_max
    pedestrians: Box(k_max, 5)         [x_rel, y_rel, vx_rel, vy_rel, age_norm]
    ped_mask   : MultiBinary(k_max)    1 if slot is real, 0 if padded
    imu        : Box(6,)               normalized [ax, ay, az, wx, wy, wz]
    goal       : Box(4,)               [d_goal, theta_goal, v_robot, omega_robot]

Action space:
    Box([v_min, omega_min], [v_max, omega_max], dtype=float32)

Reward function (v0.2): six additive terms.
    r_t = r_goal + r_collision + r_progress + r_prox + r_smooth + r_time

Changes vs original:
    1. MultiThreadedExecutor now uses 6 threads (was 2), which was the
       main bottleneck capping callback throughput.
    2. reset() now calls reset_pedestrians() instead of the disabled
       reset_world(). This clears the accumulated physical state between
       episodes (drifted pedestrians, stale contact pairs in the ODE
       broadphase) without deleting the dynamically-spawned robot.
    3. Optional per-step timing instrumentation behind a flag.
    4. Nav2 path planning integration (waypoint-based navigation).
       At reset(), Nav2 computes a global path and obs["goal"] now points
       to the current waypoint, not the final goal.

Tracker modes (v0.3):
    - ground_truth       : GroundTruthTracker, world-frame PedestrianTrack.
    - noisy_ground_truth : GroundTruthTracker wrapped with sensor noise,
                            still world-frame PedestrianTrack.
    - real_perception    : RealPerceptionTracker, returns ROBOT-frame
                            TrackRobot. The env converts these to world
                            frame using the current GT robot pose before
                            calling assemble_pedestrian_state, so all three
                            modes feed the policy identically-shaped,
                            world-frame-derived observations.

Trajectory prediction (Phase B):
    - TrajectoryPredictor is wired into _assemble_observation().
    - prediction_horizon == 0 (default) => prediction DISABLED, behaviour
      strictly identical to the pre-patch run (safety net).
    - When enabled, self._ped_futures is populated every step and is
      available to Phase C (observation augmentation) and Phase D (CBF).

References:
    - Ng et al. 1999 (potential-based reward shaping)
    - Hall 1966 (proxemic intimate zone, 0.45 m)
    - Lee et al. 2020 (smoothness term for sim-to-real)
    - Zhang et al. 2025 (terminal magnitudes inspiration, GAP_SAC)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point


from cm_gap_sac_navigation.envs.gazebo_interface import GazeboInterface
from cm_gap_sac_navigation.envs.reward import compute_reward
from cm_gap_sac_navigation.perception.ground_truth_provider import GroundTruthTracker
from cm_gap_sac_navigation.perception.pedestrian_tracker import PerceptionTracker
from cm_gap_sac_navigation.perception.track_state import (
    assemble_pedestrian_state,
    PedestrianTrack,
)
# PATCH PHASE B — REMPLACEMENT 1: import TrajectoryPredictor
from cm_gap_sac_navigation.perception.trajectory_predictor import TrajectoryPredictor
from cm_gap_sac_navigation.utils.config_loader import Config, load_config
from cm_gap_sac_navigation.utils.geometry import (
    downsample_lidar,
    goal_polar_in_robot,
    robot_to_world,
    velocity_robot_to_world,
)


# Enable per-step timing prints via env var CMG_PROFILE=1 to chase the
# 150-280 ms / step overhead. Off by default to keep training quiet.
_PROFILE = bool(int(os.environ.get("CMG_PROFILE", "0")))


# Ground-truth static obstacle list, mirrored from train.py to keep the env
# self-contained. Used as a backup collision check that does not depend on
# LiDAR ray height or mesh <collision> correctness. Disable for sim-to-real.
STATIC_OBSTACLES_GT = [
    # === Beds and large furniture ===
    (-5.0, 5.0, 0.9),    # bed_patient_2 — patient bed (longue_2), ~1m wide × 2m long, conservative circle
    
    # === Bedside tables ===
    (6.5, 5.0, 0.4),     # bedside_table_1 (petite_2)
    (6.5, -5.0, 0.4),    # bedside_table_2 (petite_1)
    
    # === IV stands (thin poles, small radius) ===
    (-1.6, -7.0, 0.3),   # iv_stand_2 (longue_1 door area)
    (-6.5, -6.5, 0.3),   # iv_stand_extra_2 (longue_1)
    
    # === Freezer / appliances ===
    (0.19, 2.1, 0.6),    # freezer_comp_2 (corridor near door north)
    
    # === Static humanoids (Scrubs models) ===
    (-2.0, -1.0, 0.4),   # nurse_1 (corridor center)
    (6.0, 0.5, 0.4),     # static_visitor_2 (corridor east end)
]


ROBOT_COLLISION_RADIUS = 0.20  # LIMO Pro footprint half-width + small margin

class LimoGazeboEnv(gym.Env):
    """Single-robot navigation env on LIMO Pro in Gazebo Harmonic."""

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        config_path: str | Path,
        seed: Optional[int] = None,
        spawn_xy_yaw: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        goal_sampler: Optional[Callable[[], Tuple[float, float]]] = None,
        spawn_sampler: Optional[Callable[[], Tuple[float, float, float]]] = None,
    ) -> None:
        super().__init__()

        self.cfg: Config = load_config(config_path)
        self._spawn_xy_yaw = spawn_xy_yaw
        self._spawn_sampler = spawn_sampler
        self._goal_sampler = goal_sampler or (lambda: (3.0, 3.0))
        self._rng = np.random.default_rng(seed)
        self._backward_dist_acc = 0.0
        self._global_step = 0  # set externally by train.py, like the curriculum tracker

        # ---- Spaces --------------------------------------------------
        n_beams = self.cfg.observation.lidar.n_beams
        k_max = self.cfg.observation.pedestrian.k_max
        n_ped_f = self.cfg.observation.pedestrian.n_features
        n_imu = self.cfg.observation.imu.n_features
        n_goal = self.cfg.observation.goal.n_features

        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(
                low=self.cfg.observation.lidar.range_min,
                high=self.cfg.observation.lidar.range_max,
                shape=(n_beams,),
                dtype=np.float32,
            ),
            "pedestrians": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(k_max, n_ped_f),
                dtype=np.float32,
            ),
            "ped_mask": spaces.MultiBinary(k_max),
            "imu": spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_imu,),
                dtype=np.float32,
            ),
            "goal": spaces.Box(
                low=np.array([0.0, -np.pi,
                              self.cfg.robot.v_min,
                              self.cfg.robot.omega_min], dtype=np.float32),
                high=np.array([1e3, np.pi,
                               self.cfg.robot.v_max,
                               self.cfg.robot.omega_max], dtype=np.float32),
                shape=(n_goal,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([self.cfg.robot.v_min,
                          self.cfg.robot.omega_min], dtype=np.float32),
            high=np.array([self.cfg.robot.v_max,
                           self.cfg.robot.omega_max], dtype=np.float32),
            dtype=np.float32,
        )

        # ---- ROS 2 wiring -------------------------------------------
        self._init_ros()

        # ---- Episode state ------------------------------------------
        self._step_count = 0
        self._goal_xy = np.array(self._goal_sampler(), dtype=np.float32)
        self._prev_d_goal: Optional[float] = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._dt = 1.0 / self.cfg.episode.control_hz

        # IMU normalization bounds for fast normalization in step()
        self._imu_norm = np.array([
            self.cfg.observation.imu.accel_max,
            self.cfg.observation.imu.accel_max,
            self.cfg.observation.imu.accel_max,
            self.cfg.observation.imu.gyro_max,
            self.cfg.observation.imu.gyro_max,
            self.cfg.observation.imu.gyro_max,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # ROS 2 lifecycle
    # ------------------------------------------------------------------
    def _init_ros(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        # Fallback spawn poses used when the config does not supply
        # pedestrian_poses. Coordinates match the SDF world layout.
        _PED_SPAWN_POSES = {
            "ped_1": ( 5.43,  1.62, 0.0),
            "ped_2": (-3.92,  1.18, 0.0),
            "ped_4": (-6.75, -1.77, 0.0),
            "ped_5": ( 2.75, -6.50, 0.0),
            "ped_6": (-1.65, -4.20, 0.0),
            "ped_7": ( 2.75,  6.50, 0.0),
            "ped_8": (-1.76,  4.20, 0.0),
        }

        self._gz = GazeboInterface(
            world_name=self.cfg.gazebo.world_name,
            scan_topic=self.cfg.gazebo.scan_topic,
            odom_topic=self.cfg.gazebo.odom_topic,
            imu_topic=self.cfg.gazebo.imu_topic,
            cmd_vel_topic=self.cfg.gazebo.cmd_vel_topic,
            pedestrian_poses=getattr(self.cfg.gazebo, "pedestrian_poses", None) or _PED_SPAWN_POSES,
        )

        # ----- Tracker selection (3-way) ------------------------------
        # tracker_mode: ground_truth | noisy_ground_truth | real_perception
        # Retro-compat: if the YAML predates tracker_mode and only has the
        # old boolean, map it so existing runs (e.g. the 1M GT run) keep
        # working without a config edit.
        mode = getattr(self.cfg.gazebo, "tracker_mode", None)
        if mode is None:
            legacy_gt = getattr(
                self.cfg.gazebo, "use_ground_truth_pedestrians", True
            )
            mode = "ground_truth" if legacy_gt else "real_perception"
        self._tracker_mode = mode

        # real_perception returns ROBOT-frame tracks; the env converts them
        # to world before assemble_pedestrian_state. ground_truth /
        # noisy_ground_truth already produce world-frame PedestrianTrack.
        self._tracker_is_robot_frame = (mode == "real_perception")

        if mode == "ground_truth":
            gt_topic = self.cfg.gazebo.gt_pose_topic.format(
                world=self.cfg.gazebo.world_name,
            )
            self._tracker = GroundTruthTracker(topic=gt_topic)
            tracker_node = self._tracker

        elif mode == "noisy_ground_truth":
            from cm_gap_sac_navigation.perception.noisy_ground_truth_tracker import (
                NoisyGroundTruthTracker, NoiseConfig,
            )
            gt_topic = self.cfg.gazebo.gt_pose_topic.format(
                world=self.cfg.gazebo.world_name,
            )
            base = GroundTruthTracker(topic=gt_topic)
            noise_cfg = getattr(self.cfg.gazebo, "perception_noise", None) or {}
            self._tracker = NoisyGroundTruthTracker(base, NoiseConfig(**noise_cfg))
            # IMPORTANT: the wrapper is NOT a Node; add the WRAPPED node to
            # the executor. __getattr__ forwards get_tracks/reset_tracks.
            tracker_node = base

        elif mode == "real_perception":
            from limo_perception.real_perception_tracker import RealPerceptionTracker
            self._tracker = RealPerceptionTracker(
                tracks_topic=getattr(
                    self.cfg.gazebo, "ped_tracks_topic", "/tracked_obstacles"),
                scan_topic=self.cfg.gazebo.scan_topic,
                image_width=getattr(self.cfg.gazebo, "cam_image_width", 640),
                h_fov_deg=getattr(self.cfg.gazebo, "cam_hfov_deg", 70.0),
                cam_yaw_offset=getattr(self.cfg.gazebo, "cam_yaw_offset", 0.0),
            )
            tracker_node = self._tracker

        else:
            raise ValueError(
                f"unknown tracker_mode: {mode!r}. "
                f"Expected ground_truth | noisy_ground_truth | real_perception."
            )

        # 6 threads: 3 sensor callbacks on GazeboInterface + tracker
        # callbacks + service-style timers. 2 threads was the original
        # bottleneck capping observed SPS at 3-5.
        self._executor = MultiThreadedExecutor(num_threads=6)
        self._executor.add_node(self._gz)
        self._executor.add_node(tracker_node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        # ============================================================
        # PATCH 6b — Nav2 path planning client.
        # Attached to self._gz (the env's Node). The action client lives
        # inside a node that is already being spun by self._executor in
        # the background thread above, so callbacks get processed for
        # free; we only need to poll future.done() to wait for results.
        # ============================================================
        self._nav2_path_client = ActionClient(
            self._gz, ComputePathToPose, 'compute_path_to_pose')
        self._gz.get_logger().info("Waiting for Nav2 planner...")
        if not self._nav2_path_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError(
                "Nav2 planner server not available. "
                "Launch nav2_planner_only.launch.py FIRST before training.")
        self._gz.get_logger().info("Nav2 planner connected.")
        # Waypoint state
        self._path_cache = {}
        self._waypoint_reach_dist = 0.5
        self._waypoints = []
        self._current_waypoint_idx = 0
        # ============================================================

        # ============================================================
        # RViz visualization publishers (cockpit complet)
        # ============================================================
        self._viz_ped_pub = self._gz.create_publisher(
            MarkerArray, '/pedestrian_markers', 10)
        self._viz_wp_pub = self._gz.create_publisher(
            MarkerArray, '/waypoints_markers', 10)
        self._viz_goal_pub = self._gz.create_publisher(
            Marker, '/goal_marker', 10)
        self._viz_traj_pub = self._gz.create_publisher(
            Path, '/robot_trajectory', 10)
        self._viz_plan_pub = self._gz.create_publisher(
            Path, '/plan', 10)
        self._viz_traj_poses = []
        self._viz_step_counter = 0
        self._viz_publish_every = 5         # publish at ~1/5 of control rate
        self._viz_max_traj_points = 800     # cap trajectory memory
        # ============================================================

        # PATCH PHASE B — REMPLACEMENT 2: trajectory predictor setup
        # prediction_horizon == 0 => prediction DISABLED (no-op, behaviour
        # identical to the current run). Safety net.
        self._pred_horizon = int(
            getattr(self.cfg.observation.pedestrian, "prediction_horizon", 0)
        )
        self._prediction_enabled = self._pred_horizon > 0
        # dt used by the LINEAR FALLBACK only (the LSTM predicts at its own
        # step as long as it is not fine-tuned; cf. dt note).
        self._pred_dt = float(
            getattr(self.cfg.observation.pedestrian, "prediction_dt", 0.4)
        )
        # Cache of the last predictions {track_id: (H, 2) world}, populated
        # every step; consumed by Phase C (obs) and Phase D (CBF).
        self._ped_futures = {}

        self._predictor = None
        if self._prediction_enabled:
            model = self._load_lstm_model()
            obs_len = int(
                getattr(self.cfg.observation.pedestrian, "prediction_obs_len", 8)
            )
            self._predictor = TrajectoryPredictor(
                model=model,
                obs_len=obs_len,
                device=getattr(self.cfg, "device", "cpu"),
            )
            kind = "LSTM" if model is not None else "fallback-lineaire"
            self._gz.get_logger().info(
                f"TrajectoryPredictor actif (horizon={self._pred_horizon}, "
                f"obs_len={obs_len}, mode={kind})"
            )

        if not self._gz.wait_for_first_messages(timeout_s=10.0):
            raise RuntimeError(
                "Gazebo did not produce /scan, /odom or /imu within 10 s. "
                "Is the simulator running?"
            )

    # ------------------------------------------------------------------
    # PATCH PHASE B — REMPLACEMENT 3: LSTM model loader
    # ------------------------------------------------------------------
    def _load_lstm_model(self):
        """Load Social-LSTM Lite. Returns None if unavailable (-> fallback).

        The checkpoint path comes from the YAML
        (observation.pedestrian.prediction_ckpt); default = best_eth_ucy.pt.
        If torch is missing or the file is absent, logs a warning and falls
        back to the linear predictor (the pipeline stays functional, just
        without LSTM).
        """
        ckpt = getattr(
            self.cfg.observation.pedestrian, "prediction_ckpt",
            os.path.expanduser(
                "~/social_lstm_lite/checkpoints/best_eth_ucy.pt"),
        )
        if not os.path.isfile(ckpt):
            self._gz.get_logger().warn(
                f"LSTM ckpt introuvable ({ckpt}) -> fallback lineaire."
            )
            return None
        try:
            import torch
            from cm_gap_sac_navigation.perception.social_lstm_lite import (
                SocialLSTMLite,
            )
            model = SocialLSTMLite(
                embedding_dim=int(getattr(
                    self.cfg.observation.pedestrian, "prediction_embed", 16)),
                hidden_size=int(getattr(
                    self.cfg.observation.pedestrian, "prediction_hidden", 32)),
                grid_size=int(getattr(
                    self.cfg.observation.pedestrian, "prediction_grid", 4)),
                neighborhood=float(getattr(
                    self.cfg.observation.pedestrian, "prediction_neighborhood", 2.0)),
            )
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                sd = sd.get("model_state_dict", sd.get("state_dict", sd))
            model.load_state_dict(sd)
            model.eval()
            self._gz.get_logger().info(f"LSTM charge : {ckpt}")
            return model
        except Exception as e:
            self._gz.get_logger().warn(
                f"Echec chargement LSTM ({e}) -> fallback lineaire."
            )
            return None

    # ==================================================================
    # PATCH 6c — Nav2 path planning helpers.
    # _wait_for_future uses poll-based waiting (NOT spin_until_future_complete)
    # because self._gz is already being spun by self._executor in a
    # background thread. Calling spin from here would deadlock or fail
    # with "node already added to an executor".
    # ==================================================================
    def _wait_for_future(self, future, timeout_s: float) -> bool:
        """Poll until future is done (executor processes callbacks in bg thread)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.005)
        return False

    def _compute_nav2_path(self, sx, sy, gx, gy, timeout_s=3.0):
        """Query Nav2 for waypoints from (sx,sy) to (gx,gy). Cached + subsampled."""
        cache_key = ("v3", round(sx, 1), round(sy, 1), round(gx, 1), round(gy, 1))
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        goal_msg = ComputePathToPose.Goal()
        now = self._gz.get_clock().now().to_msg()

        goal_msg.start.header.frame_id = "map"
        goal_msg.start.header.stamp = now
        goal_msg.start.pose.position.x = float(sx)
        goal_msg.start.pose.position.y = float(sy)
        goal_msg.start.pose.orientation.w = 1.0

        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = now
        goal_msg.goal.pose.position.x = float(gx)
        goal_msg.goal.pose.position.y = float(gy)
        goal_msg.goal.pose.orientation.w = 1.0

        goal_msg.planner_id = "GridBased"
        goal_msg.use_start = True  # CRITIQUE: utilise start fourni, pas TF

        send_future = self._nav2_path_client.send_goal_async(goal_msg)
        if not self._wait_for_future(send_future, timeout_s):
            self._gz.get_logger().warn("Nav2 send_goal timeout")
            return [(gx, gy)]

        handle = send_future.result()
        if not handle.accepted:
            self._gz.get_logger().warn("Nav2 goal rejected")
            return [(gx, gy)]

        result_future = handle.get_result_async()
        if not self._wait_for_future(result_future, timeout_s):
            self._gz.get_logger().warn("Nav2 result timeout")
            return [(gx, gy)]

        result = result_future.result().result
        poses = result.path.poses
        if len(poses) == 0:
            self._gz.get_logger().warn(
                f"Nav2 empty path ({sx:.1f},{sy:.1f})->({gx:.1f},{gy:.1f})")
            return [(gx, gy)]

        waypoints = [(p.pose.position.x, p.pose.position.y) for p in poses]
        waypoints = self._subsample_waypoints(waypoints, min_dist=1.0)
        self._path_cache[cache_key] = waypoints
        return waypoints

    def _subsample_waypoints(self, waypoints, min_dist=0.3):
        """Subsample waypoints with uniform 30cm spacing.
        
        Simpler and more effective than direction-change detection because Nav2
        paths are often straight lines with few natural direction changes.
        Uniform 30cm ensures the SAC policy always has a nearby waypoint,
        including through doorways.
        """
        if len(waypoints) <= 2:
            return waypoints
        subsampled = [waypoints[0]]
        for wp in waypoints[1:-1]:
            last = subsampled[-1]
            d = ((wp[0] - last[0])**2 + (wp[1] - last[1])**2) ** 0.5
            if d >= min_dist:
                subsampled.append(wp)
        subsampled.append(waypoints[-1])
        return subsampled
    # ==================================================================


    # ==================================================================
    # RViz visualization publishers
    # ==================================================================
    def _publish_plan(self):
        """Publish full Nav2 path on /plan (called once per episode)."""
        if not self._waypoints:
            return
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self._gz.get_clock().now().to_msg()
        for wp in self._waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._viz_plan_pub.publish(path)

    def _publish_waypoint_markers(self):
        """Publish waypoints with color coding: past/current/future/final."""
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        if not self._waypoints:
            self._viz_wp_pub.publish(arr)
            return
        stamp = self._gz.get_clock().now().to_msg()
        # Line connecting all waypoints
        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = stamp
        line.ns = "wp_line"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.04
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.5, 0.0, 0.7
        for wp in self._waypoints:
            p = Point(); p.x = float(wp[0]); p.y = float(wp[1]); p.z = 0.05
            line.points.append(p)
        arr.markers.append(line)
        # Sphere at each waypoint
        for i, wp in enumerate(self._waypoints):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(wp[0])
            m.pose.position.y = float(wp[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            if i == len(self._waypoints) - 1:                # final goal
                m.scale.x = m.scale.y = m.scale.z = 0.45
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            elif i == self._current_waypoint_idx:            # current target
                m.scale.x = m.scale.y = m.scale.z = 0.35
                m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.0
            elif i < self._current_waypoint_idx:             # passed
                m.scale.x = m.scale.y = m.scale.z = 0.15
                m.color.r, m.color.g, m.color.b = 0.4, 0.4, 0.4
            else:                                             # future
                m.scale.x = m.scale.y = m.scale.z = 0.25
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0
            m.color.a = 0.9
            arr.markers.append(m)
        self._viz_wp_pub.publish(arr)

    def _publish_goal_marker(self):
        """Publish final goal as transparent green cylinder."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self._gz.get_clock().now().to_msg()
        m.ns = "goal"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(self._goal_xy[0])
        m.pose.position.y = float(self._goal_xy[1])
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.6
        m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.9, 0.0, 0.35
        self._viz_goal_pub.publish(m)

    def _publish_pedestrians(self):
        """Publish pedestrians as blue cylinders + yellow velocity arrows."""
        # Same world-frame conversion logic as _assemble_observation
        if self._tracker_is_robot_frame:
            rx, ry, yaw, _, _ = self._gz.get_robot_state()
            robot_xy = np.array([rx, ry], dtype=np.float32)
            tracks = [
                PedestrianTrack(
                    track_id=t.track_id,
                    position_world=robot_to_world(t.position_robot, robot_xy, yaw),
                    velocity_world=velocity_robot_to_world(t.velocity_robot, yaw),
                    age_frames=t.age_frames,
                )
                for t in self._tracker.get_tracks_robot_frame()
            ]
        else:
            tracks = self._tracker.get_tracks()
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self._gz.get_clock().now().to_msg()
        for i, t in enumerate(tracks):
            body = Marker()
            body.header.frame_id = "map"
            body.header.stamp = stamp
            body.ns = "pedestrians"
            body.id = i
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = float(t.position_world[0])
            body.pose.position.y = float(t.position_world[1])
            body.pose.position.z = 0.9
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 0.5
            body.scale.z = 1.8
            body.color.r, body.color.g, body.color.b, body.color.a = 0.2, 0.6, 1.0, 0.75
            arr.markers.append(body)
            # Velocity arrow if moving
            vx, vy = float(t.velocity_world[0]), float(t.velocity_world[1])
            if vx*vx + vy*vy > 0.0025:  # > 0.05 m/s
                arrow = Marker()
                arrow.header = body.header
                arrow.ns = "ped_velocity"
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.04   # shaft diameter
                arrow.scale.y = 0.10   # head diameter
                arrow.scale.z = 0.0
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 1.0, 0.0, 0.9
                start = Point()
                start.x = float(t.position_world[0]); start.y = float(t.position_world[1]); start.z = 1.0
                end = Point()
                end.x = float(t.position_world[0] + vx)
                end.y = float(t.position_world[1] + vy)
                end.z = 1.0
                arrow.points = [start, end]
                arr.markers.append(arrow)
        self._viz_ped_pub.publish(arr)

    def _publish_robot_trajectory(self, rx, ry):
        """Append current pose + publish trail."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self._gz.get_clock().now().to_msg()
        pose.pose.position.x = float(rx)
        pose.pose.position.y = float(ry)
        pose.pose.orientation.w = 1.0
        self._viz_traj_poses.append(pose)
        if len(self._viz_traj_poses) > self._viz_max_traj_points:
            self._viz_traj_poses = self._viz_traj_poses[-self._viz_max_traj_points:]
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = pose.header.stamp
        path.poses = list(self._viz_traj_poses)
        self._viz_traj_pub.publish(path)

    def _publish_rviz_step(self):
        """Per-step publish. TF runs every step; markers throttled."""
        # TF every step (RViz needs 20 Hz to render robot smoothly)
       
        # Heavy markers throttled
        self._viz_step_counter += 1
        if self._viz_step_counter < self._viz_publish_every:
            return
        self._viz_step_counter = 0
        try:
            rx, ry, _, _, _ = self._gz.get_robot_state()
            self._publish_robot_trajectory(rx, ry)
            self._publish_waypoint_markers()
            self._publish_pedestrians()
            self._publish_goal_marker()
        except Exception as e:
            self._gz.get_logger().debug(f"viz publish failed: {e}")


   
    # ==================================================================    

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------
    def _normalize_imu(self, raw: np.ndarray) -> np.ndarray:
        """Normalize IMU into [-1, 1] using configured accel/gyro bounds.

        Saturating clip rather than scaling: an IMU spike beyond the bound
        is rare but informative; we keep the saturated value rather than
        rescaling the whole signal.
        """
        normalized = raw / self._imu_norm
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _assemble_observation(self) -> Dict[str, np.ndarray]:
        # 1. LiDAR
        raw = self._gz.get_scan()
        if raw is None:
            raw = np.full(self.cfg.observation.lidar.raw_n,
                          self.cfg.observation.lidar.range_max,
                          dtype=np.float32)
        lidar = downsample_lidar(
            raw,
            n_beams=self.cfg.observation.lidar.n_beams,
            range_min=self.cfg.observation.lidar.range_min,
            range_max=self.cfg.observation.lidar.range_max,
        )

        # 2. Robot pose + twist
        x, y, yaw, v, omega = self._gz.get_robot_state()
        robot_xy = np.array([x, y], dtype=np.float32)

        # 3. Goal in robot frame
        # PATCH 6e — obs["goal"] now points to the CURRENT WAYPOINT,
        # not the final goal. Fallback to final goal if no waypoints
        # are available (e.g. first call before reset finished).
        if self._waypoints and self._current_waypoint_idx < len(self._waypoints):
            target_xy = np.array(self._waypoints[self._current_waypoint_idx],
                                 dtype=np.float32)
        else:
            target_xy = self._goal_xy
        d_goal, theta_goal = goal_polar_in_robot(robot_xy, yaw, target_xy)

        # 4. Pedestrians
        #    ground_truth / noisy: tracker already returns world-frame
        #    PedestrianTrack via get_tracks().
        #    real_perception: tracker returns ROBOT-frame TrackRobot via
        #    get_tracks_robot_frame(); convert to world HERE using the GT
        #    robot pose (x, y, yaw) we just read. This is the single
        #    sim-to-real seam: in sim (x, y, yaw) is the Gazebo GT pose; on
        #    hardware it will be the AMCL/SLAM pose feeding get_robot_state.
        if self._tracker_is_robot_frame:
            tracks_robot = self._tracker.get_tracks_robot_frame()
            tracks = [
                PedestrianTrack(
                    track_id=t.track_id,
                    position_world=robot_to_world(
                        t.position_robot, robot_xy, yaw),
                    velocity_world=velocity_robot_to_world(
                        t.velocity_robot, yaw),
                    age_frames=t.age_frames,
                )
                for t in tracks_robot
            ]
        else:
            tracks = self._tracker.get_tracks()

        # PATCH PHASE B — REMPLACEMENT 4: feed predictor and cache futures.
        # Must happen AFTER tracks are resolved (world frame) and BEFORE
        # assemble_pedestrian_state. When disabled this is a strict no-op.
        if self._prediction_enabled and self._predictor is not None:
            self._predictor.update(tracks)
            self._ped_futures = self._predictor.predict(
                horizon=self._pred_horizon, dt=self._pred_dt,
            )
        else:
            self._ped_futures = {}

        ped_features, ped_mask = assemble_pedestrian_state(
            tracks=tracks,
            robot_xy=robot_xy,
            robot_yaw=yaw,
            k_max=self.cfg.observation.pedestrian.k_max,
            relevance_radius=self.cfg.observation.pedestrian.relevance_radius,
            track_age_norm=self.cfg.observation.pedestrian.track_age_norm,
        )
        # 5. IMU
        imu_raw = self._gz.get_imu()
        imu_normalized = self._normalize_imu(imu_raw)

        return {
            "lidar": lidar,
            "pedestrians": ped_features,
            "ped_mask": ped_mask,
            "imu": imu_normalized,
            "goal": np.array([d_goal, theta_goal, v, omega], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _check_termination(self, obs):
        info = {}

        # FIX Nav2: success uses distance to FINAL goal (self._goal_xy),
        # NOT to current waypoint (obs["goal"][0]). With waypoint-based
        # obs, the in-obs d_goal would falsely trigger success when the
        # robot reaches any intermediate waypoint.
        rx, ry, _, _, _ = self._gz.get_robot_state()
        d_to_final_goal = float(np.hypot(rx - self._goal_xy[0], ry - self._goal_xy[1]))
        if d_to_final_goal < self.cfg.episode.goal_radius:
            info["outcome"] = "success"
            return True, False, info

        # === Ground-truth static collision (mesh-independent) ===
        # The LiDAR check below misses obstacles whose mesh <collision> is
        # degenerate or whose visible silhouette sits above/below the scan
        # plane. This GT cylinder overlap catches them.
        rx, ry, _, _, _ = self._gz.get_robot_state()
        for ox, oy, oradius in STATIC_OBSTACLES_GT:
            if (rx - ox)**2 + (ry - oy)**2 < (oradius + ROBOT_COLLISION_RADIUS)**2:
                info["outcome"] = "collision"
                info["collision_type"] = "static"
                info["collision_source"] = "ground_truth"
                return True, False, info

        # === LiDAR static collision (catches walls + visible obstacles) ===
        min_range = float(obs["lidar"].min())
        if min_range < self.cfg.episode.collision_radius_static:
            info["outcome"] = "collision"
            info["collision_type"] = "static"
            info["collision_source"] = "lidar"
            return True, False, info

        # === Pedestrian collision (unchanged) ===
        mask = obs["ped_mask"]
        if mask.sum() > 0:
            active = obs["pedestrians"][mask.astype(bool)]
            d_min_ped = float(np.linalg.norm(active[:, :2], axis=1).min())
            if d_min_ped < self.cfg.episode.collision_radius_ped:
                info["outcome"] = "collision"
                info["collision_type"] = "pedestrian"
                return True, False, info

        if self._step_count >= self.cfg.episode.max_steps:
            info["outcome"] = "timeout"
            return False, True, info

        return False, False, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            terminated: bool,
            info: Dict[str, Any],
            r_reflex: float = 0.0,
        ) -> Dict[str, float]:
            """Delegate to the pure reward function (no ROS).

            r_reflex is computed by env.step (it needs obs and global_step),
            passed in as a scalar so that reward.py stays a pure function of
            MDP dynamics + scalar augmentations.
            """
            return compute_reward(
                obs=obs,
                action=action,
                prev_action=self._prev_action,
                prev_d_goal=self._prev_d_goal,
                backward_dist_acc=self._backward_dist_acc,
                r_reflex=r_reflex,
                terminated=terminated,
                info=info,
                cfg=self.cfg.reward,
            )

    def set_global_step(self, step: int) -> None:
        """Set the global training step, used by the reflex shaping decay.

        Called once per env step from train.py before env.step(), so that
        the reflex shaping term sees a coherent global counter that survives
        episode resets (the episodic _step_count cannot serve this role).
        """
        self._global_step = int(step)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _wait_for_fresh_pose(
        self,
        target_xy: Tuple[float, float],
        tol: float = 0.30,
        timeout_s: float = 1.0,
    ) -> bool:
        """Block until the ground-truth robot pose matches target_xy.

        Polls the GazeboInterface robot pose (fed by dynamic_pose/info)
        until it is within `tol` metres of the requested spawn, or until
        timeout. Guarantees the first observation of the new episode is
        computed with the post-teleport pose, not the stale one.

        Returns True if the pose converged, False on timeout (we still
        proceed; one slightly-stale first obs is better than hanging).
        """
        tx, ty = float(target_xy[0]), float(target_xy[1])
        deadline = time.monotonic() + timeout_s
        # Minimum settle so physics applies the teleport at least once.
        time.sleep(self._dt)
        while time.monotonic() < deadline:
            gx, gy, _, _, _ = self._gz.get_robot_state()
            if abs(gx - tx) < tol and abs(gy - ty) < tol:
                return True
            time.sleep(0.01)
        self._gz.get_logger().warn(
            f"_wait_for_fresh_pose: GT pose did not reach "
            f"({tx:.2f}, {ty:.2f}) within {timeout_s:.1f}s; "
            f"proceeding with latest pose."
        )
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick spawn pose
        if self._spawn_sampler is not None:
            sx, sy, syaw = self._spawn_sampler()
        else:
            sx, sy, syaw = self._spawn_xy_yaw

        # Single batched reset: robot + all pedestrians in ONE service call.
        # With gz-transport13 this is ~50-100ms vs ~3-5s with the CLI path.
        robot_ok, n_ped_ok = self._gz.reset_all_entities(
            robot_xy_yaw=(sx, sy, syaw),
            confirm_timeout_s=0.5,
        )
        if not robot_ok:
            self._gz.get_logger().error("reset_all_entities: robot pose failed")
        expected_peds = len(self._gz._pedestrian_poses)
        if n_ped_ok < expected_peds:
            self._gz.get_logger().warn(
                f"reset_all_entities: only {n_ped_ok}/{expected_peds} peds OK"
            )

        # Tracker + goal
        self._tracker.reset_tracks()
        self._goal_xy = np.array(self._goal_sampler(), dtype=np.float32)

        # PATCH PHASE B — REMPLACEMENT 5: reset predictor buffers between episodes.
        if self._predictor is not None:
            self._predictor.reset()
        self._ped_futures = {}

        # Wait for the ground-truth pose bridge to reflect the teleport,
        # instead of a blind one-tick sleep. The gz service already
        # confirmed the set, but dynamic_pose/info (gz -> ROS) can lag a
        # few tens of ms. If we assemble the first observation before the
        # bridge updates, goal_polar_in_robot uses the STALE pre-teleport
        # pose and the robot spins at episode start to "find" the goal.
        self._wait_for_fresh_pose(
            target_xy=(sx, sy), tol=0.30, timeout_s=1.0,
        )

        # ============================================================
        # PATCH 6d — Compute Nav2 path for this episode.
        # Must run AFTER goal is sampled AND pose has settled, BEFORE
        # the first _assemble_observation (so obs["goal"] points to the
        # current waypoint).
        # ============================================================
        gx, gy = float(self._goal_xy[0]), float(self._goal_xy[1])
        self._waypoints = self._compute_nav2_path(sx, sy, gx, gy)
        self._current_waypoint_idx = 0
        self._gz.get_logger().info(
            f"Nav2 path: {len(self._waypoints)} waypoints "
            f"from ({sx:.2f},{sy:.2f}) to ({gx:.2f},{gy:.2f})"
        )
        # ============================================================

        # === RViz: reset trajectory and publish episode-static markers ===
        self._viz_traj_poses = []
        self._viz_step_counter = 0
        try:
            self._publish_plan()
            self._publish_waypoint_markers()
            self._publish_goal_marker()
        except Exception as e:
            self._gz.get_logger().debug(f"viz reset publish failed: {e}")

        self._backward_dist_acc = 0.0
        self._step_count = 0
        self._prev_d_goal = None
        self._prev_action = np.zeros(2, dtype=np.float32)

        obs = self._assemble_observation()
        self._prev_d_goal = float(obs["goal"][0])
        return obs, {"goal": self._goal_xy.tolist()}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (2,), f"action must be (2,), got {action.shape}"
        action = np.clip(action, self.action_space.low, self.action_space.high)

        t_start = time.perf_counter() if _PROFILE else 0.0

        self._backward_dist_acc += max(0.0, -float(action[0])) * self._dt
        self._gz.publish_cmd(action[0], action[1])
        time.sleep(self._dt)

        if _PROFILE:
            t_after_sleep = time.perf_counter()

        self._step_count += 1
        obs = self._assemble_observation()

        if _PROFILE:
            t_after_obs = time.perf_counter()

        terminated, truncated, info = self._check_termination(obs)

        if self.cfg.reflex.enabled:
            from cm_gap_sac_navigation.envs.reflex import reflex_action, reflex_shaping
            a_reflex = reflex_action(
                obs,
                v_max=self.cfg.reflex.v_max,
                omega_max=self.cfg.reflex.omega_max,
                k_omega=self.cfg.reflex.k_omega,
                r_slowdown=self.cfg.reflex.r_slowdown,
                n_front=self.cfg.reflex.n_front_beams,
            )
            r_reflex = reflex_shaping(
                action_sac=action, action_reflex=a_reflex,
                global_step=self._global_step,
                beta_0=self.cfg.reflex.beta_0,
                t_anneal=self.cfg.reflex.t_anneal,
            )
        else:
            r_reflex = 0.0

        reward_breakdown = self._compute_reward(obs, action, terminated, info, r_reflex=r_reflex)
        reward = reward_breakdown.pop("total")
        
        info.update(reward_breakdown)
        info["step"] = self._step_count
        info["d_goal"] = float(obs["goal"][0])
        info["min_lidar"] = float(obs["lidar"].min())

        self._prev_action = action
        self._prev_d_goal = float(obs["goal"][0])

        # ============================================================
        # PATCH 6f — Advance waypoint index if current waypoint reached.
        # The success termination still triggers on obs["goal"][0] (which
        # now is the distance to the current waypoint); at the LAST
        # waypoint that distance equals the distance to the final goal,
        # so the existing termination logic naturally handles success.
        # ============================================================
        if self._waypoints and self._current_waypoint_idx < len(self._waypoints) - 1:
            rx, ry, _, _, _ = self._gz.get_robot_state()
            tx, ty = self._waypoints[self._current_waypoint_idx]
            d_wp = ((rx - tx)**2 + (ry - ty)**2) ** 0.5
            if d_wp < self._waypoint_reach_dist:
                self._current_waypoint_idx += 1
                # FIX Nav2: snap prev_d_goal to distance to NEW waypoint so that
                # the next step's r_progress = c * (prev - d_new) ≈ 0 instead of
                # a large negative jump.
                new_tx, new_ty = self._waypoints[self._current_waypoint_idx]
                self._prev_d_goal = float(np.hypot(rx - new_tx, ry - new_ty))
        # ============================================================

        if terminated or truncated:
            self._gz.publish_zero_cmd()
            # === RViz: throttled visualization update ===
            self._publish_rviz_step()
        if _PROFILE and self._step_count % 20 == 0:
            t_end = time.perf_counter()
            print(
                f"[step {self._step_count}] "
                f"sleep={1000*(t_after_sleep-t_start):.1f}ms "
                f"obs={1000*(t_after_obs-t_after_sleep):.1f}ms "
                f"rest={1000*(t_end-t_after_obs):.1f}ms "
                f"total={1000*(t_end-t_start):.1f}ms"
            )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self._gz.publish_zero_cmd()
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._gz.destroy_node()
            # NoisyGroundTruthTracker is not itself a Node; destroy the
            # wrapped node if present, otherwise destroy the tracker
            # directly (GroundTruthTracker / RealPerceptionTracker).
            tracker_to_destroy = getattr(self._tracker, "_base", self._tracker)
            tracker_to_destroy.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


# ----------------------------------------------------------------------
# CLI smoke test entry point (registered in setup.py)
# ----------------------------------------------------------------------
def smoke_test() -> None:
    """Run 50 random-action steps in the env. Useful to verify wiring."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[2] / "config" / "cm_gap_sac.yaml"),
    )
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    env = LimoGazeboEnv(config_path=args.config, seed=0)
    obs, info = env.reset()
    print(f"[smoke] reset OK. obs keys: {list(obs.keys())}")
    print(f"[smoke] lidar shape: {obs['lidar'].shape}, "
          f"ped shape: {obs['pedestrians'].shape}, "
          f"imu shape: {obs['imu'].shape}, "
          f"goal: {obs['goal']}")

    total_r = 0.0
    for t in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            outcome = info.get("outcome", "?")
            print(f"[smoke] episode ended at t={t} ({outcome}); resetting")
            obs, info = env.reset()

    print(f"[smoke] total reward: {total_r:.2f}")
    env.close()


if __name__ == "__main__":
    smoke_test()