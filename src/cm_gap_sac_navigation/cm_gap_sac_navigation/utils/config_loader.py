"""Tiny config loader that parses cm_gap_sac.yaml into a nested dataclass.

v0.2 changes:
    - ImuCfg added under ObservationCfg
    - reward fields kept as raw dict for compatibility, plus a typed
      RewardCfg helper for the env.
    - HeadsCfg added (actor/critic dims).

v0.3 changes:
    - GazeboCfg extended with tracker_mode (ground_truth |
      noisy_ground_truth | real_perception), perception_noise, and
      camera-projection fields (cam_image_width, cam_hfov_deg,
      cam_yaw_offset) used by RealPerceptionTracker / NoisyGroundTruthTracker.
      All new fields have defaults, so existing YAMLs (e.g. the ones using
      only use_ground_truth_pedestrians) keep loading unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RobotCfg:
    name: str
    wheel_radius: float
    wheel_separation: float
    base_frame: str
    odom_frame: str
    v_max: float
    v_min: float
    omega_max: float
    omega_min: float


@dataclass
class LidarCfg:
    raw_n: int
    n_beams: int
    range_min: float
    range_max: float
    fov_deg: float


@dataclass
class PedestrianCfg:
    k_max: int
    n_features: int
    track_age_norm: int
    relevance_radius: float


@dataclass
class ImuCfg:                            # NEW in v0.2
    n_features: int
    accel_max: float
    gyro_max: float


@dataclass
class GoalCfg:
    n_features: int


@dataclass
class ObservationCfg:
    lidar: LidarCfg
    pedestrian: PedestrianCfg
    imu: ImuCfg                          # NEW in v0.2
    goal: GoalCfg


@dataclass
class EpisodeCfg:
    control_hz: float
    max_steps: int
    goal_radius: float
    collision_radius_static: float
    collision_radius_ped: float


@dataclass
class GazeboCfg:
    world_name: str
    use_synchronous_stepping: bool
    reset_service: str
    set_pose_service: str
    spawn_service: str
    scan_topic: str
    odom_topic: str
    imu_topic: str
    cmd_vel_topic: str
    ped_tracks_topic: str
    use_ground_truth_pedestrians: bool
    gt_pose_topic: str
    use_reset_world: bool = True
    use_set_pose: bool = True

    # ---- NEW in v0.3: 3-way tracker selection -----------------------
    # ground_truth | noisy_ground_truth | real_perception
    # If absent, LimoGazeboEnv derives it from use_ground_truth_pedestrians.
    tracker_mode: str = "ground_truth"
    perception_noise: dict = field(default_factory=dict)

    # ---- NEW in v0.3: camera projection for real_perception ---------
    cam_image_width: int = 640
    cam_hfov_deg: float = 70.0
    cam_yaw_offset: float = 0.0


@dataclass
class RewardCfg:
    r_goal: float
    r_collision: float
    c_progress: float
    d_intimate: float
    alpha_prox: float
    alpha_smooth: float
    r_time: float
    alpha_shield: float = 0.0
    alpha_reverse: float = 0.05
    d_reverse_free: float = 0.5
    d_reverse_sat: float = 0.5
    d_social: float = 2.5
    alpha_social: float = 0.5
    social_gate_blend: float = 1.0
    collision_radius_ped: float = 0.46
    # NEW: static-obstacle proximity (mirrors r_prox for pedestrians)
    alpha_static_prox: float = 0.0       # default 0 → backward compatible
    d_static_safe: float = 0.5
    collision_radius_static: float = 0.20
@dataclass
class ReflexCfg:
    """Hand-crafted geometric reflex used as a decaying prior shaping term."""
    enabled: bool = False
    v_max: float = 0.5
    omega_max: float = 1.0
    k_omega: float = 1.0
    k_repulse: float = 1.5
    r_influence: float = 1.0
    r_slowdown: float = 1.5
    n_front_beams: int = 30
    beta_0: float = 0.5
    t_anneal: int = 50000


@dataclass
class Config:
    """Top-level config; sub-sections used by current modules are typed."""
    robot: RobotCfg
    observation: ObservationCfg
    episode: EpisodeCfg
    gazebo: GazeboCfg
    reward: RewardCfg                    # promoted to typed in v0.2
    reflex: ReflexCfg = field(default_factory=ReflexCfg)   # ← NEW
    encoders: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, Any] = field(default_factory=dict)
    heads: Dict[str, Any] = field(default_factory=dict)        # NEW
    sac: Dict[str, Any] = field(default_factory=dict)
    per: Dict[str, Any] = field(default_factory=dict)
    cbf: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        robot=RobotCfg(**raw["robot"]),
        observation=ObservationCfg(
            lidar=LidarCfg(**raw["observation"]["lidar"]),
            pedestrian=PedestrianCfg(**raw["observation"]["pedestrian"]),
            imu=ImuCfg(**raw["observation"]["imu"]),
            goal=GoalCfg(**raw["observation"]["goal"]),
        ),
        episode=EpisodeCfg(**raw["episode"]),
        gazebo=GazeboCfg(**raw["gazebo"]),
        reward=RewardCfg(**raw["reward"]),
        reflex=ReflexCfg(**raw.get("reflex", {})),
        encoders=raw.get("encoders", {}),
        attention=raw.get("attention", {}),
        heads=raw.get("heads", {}),
        sac=raw.get("sac", {}),
        per=raw.get("per", {}),
        cbf=raw.get("cbf", {}),
        training=raw.get("training", {}),
        action=raw.get("action", {}),
    )