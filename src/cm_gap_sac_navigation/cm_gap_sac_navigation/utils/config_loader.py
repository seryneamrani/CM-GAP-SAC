"""Tiny config loader that parses cm_gap_sac.yaml into a nested dataclass.

v0.2 changes:
    - ImuCfg added under ObservationCfg
    - reward fields kept as raw dict for compatibility, plus a typed
      RewardCfg helper for the env.
    - HeadsCfg added (actor/critic dims).
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
    collision_radius: float


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

@dataclass
class RewardCfg:                         # typed view of the reward terms
    r_goal: float
    r_collision: float
    c_progress: float
    d_intimate: float
    alpha_prox: float
    alpha_smooth: float
    r_time: float
    alpha_shield: float = 0.0            # NEW: shield activation penalty
                                          # (default 0 for backward compatibility)
    alpha_reverse: float = 0.05
@dataclass
class Config:
    """Top-level config; sub-sections used by current modules are typed."""
    robot: RobotCfg
    observation: ObservationCfg
    episode: EpisodeCfg
    gazebo: GazeboCfg
    reward: RewardCfg                    # promoted to typed in v0.2
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
        encoders=raw.get("encoders", {}),
        attention=raw.get("attention", {}),
        heads=raw.get("heads", {}),
        sac=raw.get("sac", {}),
        per=raw.get("per", {}),
        cbf=raw.get("cbf", {}),
        training=raw.get("training", {}),
        action=raw.get("action", {}),
    )
