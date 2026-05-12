"""Gymnasium environment wrapping Gazebo Harmonic for CM-GAP_SAC training (v0.2).

The environment owns:
    - One GazeboInterface node (sensors: scan, odom, imu + cmd_vel + sim control)
    - One pedestrian tracker (PerceptionTracker or GroundTruthTracker)
    - A background thread that spins both nodes via a MultiThreadedExecutor

Observation space (Dict to keep modalities cleanly separated):
    lidar      : Box(n_beams,)         downsampled scan, clipped to range_max
    pedestrians: Box(k_max, 5)         [x_rel, y_rel, vx_rel, vy_rel, age_norm]
    ped_mask   : MultiBinary(k_max)    1 if slot is real, 0 if padded
    imu        : Box(6,)               normalized [ax, ay, az, wx, wy, wz]    NEW v0.2
    goal       : Box(4,)               [d_goal, theta_goal, v_robot, omega_robot]

Action space:
    Box([v_min, omega_min], [v_max, omega_max], dtype=float32)

Reward function (v0.2): six additive terms.
    r_t = r_goal + r_collision + r_progress + r_prox + r_smooth + r_time

References:
    - Ng et al. 1999 (potential-based reward shaping)
    - Hall 1966 (proxemic intimate zone, 0.45 m)
    - Lee et al. 2020 (smoothness term for sim-to-real)
    - Zhang et al. 2025 (terminal magnitudes inspiration, GAP_SAC)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor

from cm_gap_sac_navigation.envs.gazebo_interface import GazeboInterface
from cm_gap_sac_navigation.envs.reward import compute_reward
from cm_gap_sac_navigation.perception.ground_truth_provider import GroundTruthTracker
from cm_gap_sac_navigation.perception.pedestrian_tracker import PerceptionTracker
from cm_gap_sac_navigation.perception.track_state import (
    assemble_pedestrian_state,
)
from cm_gap_sac_navigation.utils.config_loader import Config, load_config
from cm_gap_sac_navigation.utils.geometry import (
    downsample_lidar,
    goal_polar_in_robot,
)


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
    ) -> None:
        super().__init__()

        self.cfg: Config = load_config(config_path)
        self._spawn_xy_yaw = spawn_xy_yaw
        self._goal_sampler = goal_sampler or (lambda: (3.0, 3.0))
        self._rng = np.random.default_rng(seed)

        # ---- Spaces --------------------------------------------------
        n_beams = self.cfg.observation.lidar.n_beams
        k_max   = self.cfg.observation.pedestrian.k_max
        n_ped_f = self.cfg.observation.pedestrian.n_features
        n_imu   = self.cfg.observation.imu.n_features
        n_goal  = self.cfg.observation.goal.n_features

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
            "imu": spaces.Box(                                         # NEW v0.2
                low=-1.0, high=1.0,    # normalized
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

        # ---- Episode state -------------------------------------------
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

        self._gz = GazeboInterface(
            world_name=self.cfg.gazebo.world_name,
            scan_topic=self.cfg.gazebo.scan_topic,
            odom_topic=self.cfg.gazebo.odom_topic,
            imu_topic=self.cfg.gazebo.imu_topic,                       # NEW v0.2
            cmd_vel_topic=self.cfg.gazebo.cmd_vel_topic,
            reset_service_template=self.cfg.gazebo.reset_service,
            set_pose_service_template=self.cfg.gazebo.set_pose_service,
        )

        if self.cfg.gazebo.use_ground_truth_pedestrians:
            gt_topic = self.cfg.gazebo.gt_pose_topic.format(
                world=self.cfg.gazebo.world_name,
            )
            self._tracker = GroundTruthTracker(topic=gt_topic)
        else:
            self._tracker = PerceptionTracker(
                topic=self.cfg.gazebo.ped_tracks_topic,
            )

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._gz)
        self._executor.add_node(self._tracker)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._spin_thread.start()

        if not self._gz.wait_for_first_messages(timeout_s=10.0):
            raise RuntimeError(
                "Gazebo did not produce /scan, /odom or /imu within 10 s. "
                "Is the simulator running?"
            )

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
        d_goal, theta_goal = goal_polar_in_robot(robot_xy, yaw, self._goal_xy)

        # 4. Pedestrians
        tracks = self._tracker.get_tracks()
        ped_features, ped_mask = assemble_pedestrian_state(
            tracks=tracks,
            robot_xy=robot_xy,
            robot_yaw=yaw,
            k_max=self.cfg.observation.pedestrian.k_max,
            relevance_radius=self.cfg.observation.pedestrian.relevance_radius,
            track_age_norm=self.cfg.observation.pedestrian.track_age_norm,
        )

        # 5. IMU                                                       NEW v0.2
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
    def _check_termination(self, obs: Dict[str, np.ndarray]
                           ) -> Tuple[bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}

        d_goal = float(obs["goal"][0])
        if d_goal < self.cfg.episode.goal_radius:
            info["outcome"] = "success"
            return True, False, info

        min_range = float(obs["lidar"].min())
        if min_range < self.cfg.episode.collision_radius:
            info["outcome"] = "collision"
            return True, False, info

        if self._step_count >= self.cfg.episode.max_steps:
            info["outcome"] = "timeout"
            return False, True, info

        return False, False, info

    # ------------------------------------------------------------------
    # Reward (v0.2: full 6-term implementation)
    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        terminated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Delegate to the pure reward function (no ROS)."""
        return compute_reward(
            obs=obs,
            action=action,
            prev_action=self._prev_action,
            prev_d_goal=self._prev_d_goal,
            terminated=terminated,
            info=info,
            cfg=self.cfg.reward,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._gz.publish_zero_cmd()
        time.sleep(0.05)

        # DEBUG: reset_world disabled (robot disappears)
        # ok = self._gz.reset_world(timeout_s=2.0)
        # if not ok:
        #     self._gz.get_logger().warn("World reset failed; continuing anyway.")

        # DEBUG: set_robot_pose disabled (robot disappears)
        if hasattr(self, '_spawn_sampler') and self._spawn_sampler is not None:
            sx, sy, syaw = self._spawn_sampler()
        else:
            sx, sy, syaw = self._spawn_xy_yaw
        self._gz.set_robot_pose(self.cfg.robot.name, sx, sy, syaw, timeout_s=2.0)

        self._tracker.reset_tracks()
        self._goal_xy = np.array(self._goal_sampler(), dtype=np.float32)

        time.sleep(self._dt)

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

        self._gz.publish_cmd(action[0], action[1])
        time.sleep(self._dt)

        self._step_count += 1
        obs = self._assemble_observation()
        terminated, truncated, info = self._check_termination(obs)
        reward_breakdown = self._compute_reward(obs, action, terminated, info)
        reward = reward_breakdown.pop("total")

        info.update(reward_breakdown)
        info["step"] = self._step_count
        info["d_goal"] = float(obs["goal"][0])
        info["min_lidar"] = float(obs["lidar"].min())

        self._prev_action = action
        self._prev_d_goal = float(obs["goal"][0])

        if terminated or truncated:
            self._gz.publish_zero_cmd()

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
            self._tracker.destroy_node()
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
