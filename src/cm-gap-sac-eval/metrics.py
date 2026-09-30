"""
CM-GAP-SAC — Per-episode metric computation.

Pure Python + numpy. No ROS/Gazebo dependency — testable on synthetic
trajectories without the simulation stack.

Usage:
    from metrics import EpisodeTrace, compute_metrics

    trace = EpisodeTrace(dt=0.1, positions=..., velocities=..., ...)
    result = compute_metrics(trace, config)
    # result is JSON-serializable, matches the results.jsonl schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ==============================================================================
# Data container — populate during episode rollout
# ==============================================================================
@dataclass
class EpisodeTrace:
    """Raw per-step data recorded during one episode.

    Populate incrementally in run_eval.py's step loop; pass to
    compute_metrics() at episode end.
    """
    dt: float                                # sim step (s)

    # Kinematics — arrays of shape (T, ...)
    positions: np.ndarray                    # (T, 2) — robot xy
    velocities: np.ndarray                   # (T, 2) — linear velocity xy
    actions: np.ndarray                      # (T, action_dim)

    # Reward
    rewards: np.ndarray                      # (T,) — total reward per step
    reward_components: dict[str, np.ndarray] = field(default_factory=dict)
    # e.g. {"goal": (T,), "progress": (T,), "collision": (T,), "cbf_safety": (T,), ...}

    # Distances per step
    min_dist_static: np.ndarray = field(default_factory=lambda: np.array([]))       # (T,)
    min_dist_pedestrian: np.ndarray = field(default_factory=lambda: np.array([]))   # (T,) inf if no ped

    # CBF (None if CBF disabled for this run)
    cbf_active: Optional[np.ndarray] = None          # (T,) bool
    cbf_correction: Optional[np.ndarray] = None      # (T,) ||a_shielded - a_policy||
    cbf_infeasible: Optional[np.ndarray] = None      # (T,) bool — QP infeasible

    # Policy diagnostics (optional — SAC stochastic policy)
    policy_entropy_per_step: Optional[np.ndarray] = None

    # Termination
    reached_goal: bool = False
    collided_pedestrian: bool = False
    collided_static: bool = False
    timed_out: bool = False

    # Environmental
    optimal_path_length: float = 0.0         # Nav2 global planner path length — SPL/PLR denominator
    goal_xy: Optional[tuple[float, float]] = None    # for GPE (goal-progress efficiency)


# ==============================================================================
# Individual metric functions (unit-testable)
# ==============================================================================
def check_freeze_at_start(velocities: np.ndarray, dt: float,
                          threshold_v: float = 0.05,
                          window_s: float = 3.0) -> bool:
    """True if robot never exceeded threshold_v within the first window_s seconds."""
    n_steps = max(1, int(window_s / dt))
    n_steps = min(n_steps, len(velocities))
    if n_steps == 0:
        return True
    speeds = np.linalg.norm(velocities[:n_steps], axis=-1)
    return bool(np.max(speeds) < threshold_v)


def compute_path_length(positions: np.ndarray) -> float:
    """Total traveled Euclidean distance."""
    if len(positions) < 2:
        return 0.0
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=-1)))


def compute_planner_path_length(planner_path: Optional[np.ndarray]) -> float:
    """Longueur du plan global (Nav2) — somme des distances entre waypoints
    consécutifs. Utilisée comme référence optimale pour SPL et PLR à la
    place du straight-line euclidien.

    Args:
        planner_path: (N, 2) waypoints xy retournés par Nav2 au reset.
                      None ou < 2 points → planning failed / goal trivial.

    Returns:
        Longueur (m) du plan, ou 0.0 si pas de plan valide.
    """
    if planner_path is None or len(planner_path) < 2:
        return 0.0
    diffs = np.diff(planner_path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=-1)))


def compute_spl(success: bool, path_length: float, optimal_length: float) -> float:
    """Success weighted by Path Length (Anderson et al. 2018)."""
    if not success or path_length <= 0 or optimal_length <= 0:
        return 0.0
    return float(optimal_length / max(path_length, optimal_length))


def compute_action_smoothness(actions: np.ndarray) -> float:
    """Mean ||a_t - a_{t-1}|| — jerk proxy."""
    if len(actions) < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=-1)))


def compute_personal_space_violation_rate(min_dist_pedestrian: np.ndarray,
                                          threshold_m: float = 0.5) -> float:
    """Fraction of steps within threshold_m of nearest pedestrian."""
    if len(min_dist_pedestrian) == 0:
        return 0.0
    finite = min_dist_pedestrian[np.isfinite(min_dist_pedestrian)]
    if len(finite) == 0:
        return 0.0
    return float(np.mean(finite < threshold_m))


def compute_social_discomfort_drl_vo(min_dist_pedestrian: np.ndarray,
                                     robot_speeds: np.ndarray,
                                     d_min: float = 0.2,
                                     d_max: float = 1.0) -> float:
    """DRL-VO social discomfort — proximity weighted by robot speed.

    score_t = speed_t * (1 - clip((d_t - d_min) / (d_max - d_min), 0, 1))
    """
    if len(min_dist_pedestrian) == 0:
        return 0.0
    d = np.where(np.isfinite(min_dist_pedestrian), min_dist_pedestrian, d_max + 1.0)
    proximity = 1.0 - np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)
    return float(np.mean(robot_speeds * proximity))


# ==============================================================================
# Additional metrics (5) — added to complement the reference set
# ==============================================================================
def compute_time_to_first_motion(velocities: np.ndarray, dt: float,
                                 v_thresh: float = 0.05) -> float:
    """Time (seconds) until first step exceeding v_thresh. Returns full episode
    duration if no motion ever detected."""
    if len(velocities) == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=-1)
    above = np.where(speeds >= v_thresh)[0]
    if len(above) == 0:
        return float(len(velocities) * dt)
    return float(above[0] * dt)


def compute_cbf_infeasibility_rate(cbf_infeasible: Optional[np.ndarray]) -> float:
    """Fraction of steps where the CBF-QP was infeasible.

    Requires per-step logging of shield_result.infeasible (already tracked in
    train.py — plumb it through env.step's info dict as info['cbf_infeasible'])."""
    if cbf_infeasible is None or len(cbf_infeasible) == 0:
        return 0.0
    return float(np.mean(cbf_infeasible.astype(bool)))


def compute_path_smoothness(velocities: np.ndarray, path_length: float,
                            eps: float = 1e-6) -> float:
    """Total heading change per unit distance (rad/m).

    Uses velocity direction as heading proxy; robust to instantaneous stops
    (skips steps with |v| below noise floor).
    """
    if len(velocities) < 2 or path_length < eps:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=-1)
    valid = speeds > 0.02  # ignore near-stationary steps for heading estimation
    if valid.sum() < 2:
        return 0.0
    headings = np.arctan2(velocities[valid, 1], velocities[valid, 0])
    dtheta = np.diff(headings)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap to [-pi, pi]
    total_turn = float(np.sum(np.abs(dtheta)))
    return total_turn / (path_length + eps)


def compute_goal_progress_efficiency(positions: np.ndarray,
                                     goal_xy: Optional[tuple[float, float]],
                                     path_length: float,
                                     eps: float = 1e-6) -> float:
    """Net goal-approach distance divided by total path length.

    Range: [-1, 1]. Value 1 = every meter traveled brought robot 1m closer
    to goal (rectilinear pursuit). Value 0 = ended at same distance from
    goal as start (wandering). Value < 0 = ended further from goal.
    """
    if goal_xy is None or len(positions) < 2 or path_length < eps:
        return 0.0
    gx, gy = goal_xy
    d0 = float(np.hypot(positions[0, 0] - gx, positions[0, 1] - gy))
    dT = float(np.hypot(positions[-1, 0] - gx, positions[-1, 1] - gy))
    return (d0 - dT) / (path_length + eps)


def compute_initial_goal_distance(positions: np.ndarray,
                                  goal_xy: Optional[tuple[float, float]]) -> float:
    """Straight-line initial distance to goal — used offline in analysis to
    stratify SR by distance quartile (SR-D metric)."""
    if goal_xy is None or len(positions) == 0:
        return 0.0
    gx, gy = goal_xy
    return float(np.hypot(positions[0, 0] - gx, positions[0, 1] - gy))


# ==============================================================================
# Full metrics computation — called once per episode
# ==============================================================================
def compute_metrics(trace: EpisodeTrace, config: dict) -> dict:
    """Compute the full metrics dict for one episode.

    Returns a JSON-serializable dict with keys `outcome`, `metrics`,
    `path_length_m` — matches the results.jsonl schema.
    """
    m = config["metrics"]
    speeds = np.linalg.norm(trace.velocities, axis=-1) if len(trace.velocities) else np.array([])
    path_length = compute_path_length(trace.positions)

    success = trace.reached_goal and not (trace.collided_pedestrian or trace.collided_static)

    outcome = {
        "success": bool(success),
        "collision_pedestrian": bool(trace.collided_pedestrian),
        "collision_static": bool(trace.collided_static),
        "timeout": bool(trace.timed_out),
        "freeze_at_start": check_freeze_at_start(
            trace.velocities, trace.dt,
            threshold_v=m["navigation_core"]["freeze_threshold_v"],
            window_s=m["navigation_core"]["freeze_window_s"],
        ),
    }

    metrics_out: dict = {}

    # ---- Task performance ----
    if m["navigation_core"]["spl"]:
        metrics_out["spl"] = compute_spl(success, path_length, trace.optimal_path_length)

    # ---- RL diagnostics ----
    rl = m.get("rl_diagnostics", {})
    if rl.get("episode_return"):
        metrics_out["episode_return"] = float(np.sum(trace.rewards))
    if rl.get("episode_return_components") and trace.reward_components:
        metrics_out["episode_return_components"] = {
            name: float(np.sum(vals)) for name, vals in trace.reward_components.items()
        }
    if rl.get("episode_length_steps"):
        metrics_out["episode_length_steps"] = int(len(trace.velocities))
    if rl.get("policy_entropy_mean") and trace.policy_entropy_per_step is not None \
            and len(trace.policy_entropy_per_step) > 0:
        metrics_out["policy_entropy_mean"] = float(np.mean(trace.policy_entropy_per_step))
    if rl.get("action_smoothness"):
        metrics_out["action_smoothness"] = compute_action_smoothness(trace.actions)

    # ---- Safety / CBF ----
    if m["safety_cbf"]["min_clearance_static"]:
        metrics_out["min_clearance_static_m"] = (
            float(np.min(trace.min_dist_static)) if len(trace.min_dist_static) else 0.0
        )
    if m["safety_cbf"]["min_clearance_pedestrian"]:
        finite_ped = trace.min_dist_pedestrian[np.isfinite(trace.min_dist_pedestrian)] \
            if len(trace.min_dist_pedestrian) else np.array([])
        metrics_out["min_clearance_pedestrian_m"] = (
            float(np.min(finite_ped)) if len(finite_ped) else float("inf")
        )
    if m["safety_cbf"]["intervention_rate"] and trace.cbf_active is not None:
        metrics_out["cbf_intervention_rate"] = float(np.mean(trace.cbf_active))
    if m["safety_cbf"]["correction_magnitude"] and trace.cbf_correction is not None \
            and trace.cbf_active is not None:
        active_mask = trace.cbf_active.astype(bool)
        if active_mask.any():
            metrics_out["cbf_correction_magnitude_mean"] = float(
                np.mean(trace.cbf_correction[active_mask])
            )
        else:
            metrics_out["cbf_correction_magnitude_mean"] = 0.0

    # ---- Social ----
    if m["social"]["personal_space_violation_rate"]:
        metrics_out["personal_space_violation_rate"] = compute_personal_space_violation_rate(
            trace.min_dist_pedestrian,
            threshold_m=m["social"]["personal_space_threshold_m"],
        )
    if m["social"]["social_discomfort_score"]:
        metrics_out["social_discomfort_score"] = compute_social_discomfort_drl_vo(
            trace.min_dist_pedestrian, speeds
        )

    # ---- Efficiency ----
    # metrics.py, dans compute_metrics()
    if m["efficiency"]["path_length_ratio"]:
        metrics_out["path_length_ratio"] = (
            float(path_length / trace.optimal_path_length)
            if success and trace.optimal_path_length > 0 else None
        )
    if m["efficiency"]["time_to_goal"]:
        metrics_out["time_to_goal_s"] = (
            float(len(trace.velocities) * trace.dt) if success else None
        )

    # ---- Additional metrics (always computed — cheap post-processing) ----
    metrics_out["time_to_first_motion_s"] = compute_time_to_first_motion(
        trace.velocities, trace.dt,
        v_thresh=m["navigation_core"]["freeze_threshold_v"],
    )
    metrics_out["cbf_infeasibility_rate"] = compute_cbf_infeasibility_rate(
        getattr(trace, "cbf_infeasible", None)
    )
    metrics_out["path_smoothness_rad_per_m"] = compute_path_smoothness(
        trace.velocities, path_length
    )
    metrics_out["goal_progress_efficiency"] = compute_goal_progress_efficiency(
        trace.positions, trace.goal_xy, path_length
    )
    metrics_out["initial_goal_distance_m"] = compute_initial_goal_distance(
        trace.positions, trace.goal_xy
    )

    return {
        "outcome": outcome,
        "metrics": metrics_out,
        "path_length_m": float(path_length),
    }