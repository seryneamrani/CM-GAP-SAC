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


# Enable per-step timing prints via env var CMG_PROFILE=1 to chase the
# 150-280 ms / step overhead. Off by default to keep training quiet.
_PROFILE = bool(int(os.environ.get("CMG_PROFILE", "0")))


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
            "ped_5": ( 3.21, -6.50, 0.0),
            "ped_6": (-1.65, -4.20, 0.0),
            "ped_7": ( 3.19,  6.50, 0.0),
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

        if self.cfg.gazebo.use_ground_truth_pedestrians:
            gt_topic = self.cfg.gazebo.gt_pose_topic.format(
                world=self.cfg.gazebo.world_name,
            )
            self._tracker = GroundTruthTracker(topic=gt_topic)
        else:
            self._tracker = PerceptionTracker(
                topic=self.cfg.gazebo.ped_tracks_topic,
            )

        # 6 threads: 3 sensor callbacks on GazeboInterface + tracker
        # callbacks + service-style timers. 2 threads was the original
        # bottleneck capping observed SPS at 3-5.
        self._executor = MultiThreadedExecutor(num_threads=6)
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
    def _check_termination(self, obs: Dict[str, np.ndarray]
                       ) -> Tuple[bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}

        d_goal = float(obs["goal"][0])
        if d_goal < self.cfg.episode.goal_radius:
            info["outcome"] = "success"
            return True, False, info

        # Static obstacle collision (walls, furniture) via LiDAR.
        min_range = float(obs["lidar"].min())
        if min_range < self.cfg.episode.collision_radius:
            info["outcome"] = "collision"
            return True, False, info

        # Dynamic obstacle collision (pedestrians) via ground-truth positions.
        # Necessary because Gazebo Harmonic actors do not have a collision mesh
        # by default, so the LiDAR passes through them without detection.
        mask = obs["ped_mask"]
        if mask.sum() > 0:
            active = obs["pedestrians"][mask.astype(bool)]
            d_min_ped = float(np.linalg.norm(active[:, :2], axis=1).min())
            if d_min_ped < self.cfg.episode.collision_radius:
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

        # Let physics settle one control tick
        time.sleep(self._dt)

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