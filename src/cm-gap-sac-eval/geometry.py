"""
CM-GAP-SAC — Geometry & sampling helpers for eval.

Constants and validators COPIED from train.py. Keep in sync if the training
world changes. Ideally, refactor these into a shared module inside
cm_gap_sac_navigation.utils and import from both train.py and run_eval.py.

Public API:
    ZONE_BOXES              — dict of (xmin, xmax, ymin, ymax) per zone name
    sample_spawn_in_zone    — validated spawn point in a zone
    sample_goal_in_zone     — validated goal point in a zone with min-distance
    compute_optimal_length  — straight-line SPL denominator (upgrade to A* later)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ==============================================================================
# World geometry — MUST match train.py values
# ==============================================================================
HOSPITAL_X_MIN, HOSPITAL_X_MAX = -7.3, 7.3
HOSPITAL_Y_MIN, HOSPITAL_Y_MAX = -7.3, 7.3

MIN_SPAWN_GOAL_DIST = 2.0
ROBOT_MARGIN = 0.25
GOAL_CLEARANCE = 1.00
GOAL_OBSTACLE_CLEARANCE = 0.70
SPAWN_OBSTACLE_CLEARANCE = 0.8
PED_SPAWN_CLEARANCE = 0.8

WALLS = [
    (-8.1, -7.9, -8.0, 8.0),      # wall_south
    ( 7.9,  8.1, -8.0, 8.0),      # wall_north
    (-8.0,  8.0, -8.1, -7.9),     # wall_west
    (-8.0,  8.0,  7.9,  8.1),     # wall_east
    (-5.5,  5.5,  2.9, 3.1),      # corridor_wall_1
    (-5.5,  5.5, -3.1, -2.9),     # corridor_wall_2
    ( 1.9,  2.1,  3.15,  4.85),   # div_left_top_1
    ( 1.9,  2.1,  6.15,  7.85),   # div_left_top_2
    ( 1.9,  2.1, -4.85, -3.15),   # div_left_bot_1
    ( 1.9,  2.1, -7.85, -6.15),   # div_left_bot_2
]

STATIC_OBSTACLES = [
    (-5.0, 5.0, 0.9),    # bed_patient_2
    ( 6.5, 5.0, 0.4),    # bedside_table_1
    ( 6.5, -5.0, 0.4),   # bedside_table_2
    (-1.6, -7.0, 0.3),   # iv_stand_2
    (-6.5, -6.5, 0.3),   # iv_stand_extra_2
    ( 0.19, 2.1, 0.6),   # freezer_comp_2
    (-2.0, -1.0, 0.4),   # nurse_1
    ( 6.0, 0.5, 0.4),    # static_visitor_2
]

PED_STARTS = [
    (5.43, 1.62), (-3.92, 1.18), (-6.75, -1.77),
    (2.75, -6.50), (-1.65, -4.20), (2.75, 6.50), (-1.76, 4.20),
]

# ZONE_BOXES: (xmin, xmax, ymin, ymax) — matches train.py exactly
ZONE_BOXES = {
    "corridor": (-6.5, 6.5, -1.5, 1.5),
    "longue_1": (-7.0, -4.0, -7.0, -4.0),
    "petite_1": (4.5, 6.5, -6.5, -4.0),
    "petite_2": (4.5, 6.5,  4.0,  6.5),
    "longue_2": (-7.0, -4.0, 4.0,  7.0),
}


# ==============================================================================
# Validators — bitwise identical to train.py
# ==============================================================================
def _in_wall(x: float, y: float, margin: float) -> bool:
    for xmin, xmax, ymin, ymax in WALLS:
        if (xmin - margin < x < xmax + margin
                and ymin - margin < y < ymax + margin):
            return True
    return False


def is_navigable(x: float, y: float) -> bool:
    """True if (x, y) is a valid ROBOT spawn location."""
    if not (-7.8 < x < 7.8 and -7.8 < y < 7.8):
        return False
    if _in_wall(x, y, ROBOT_MARGIN):
        return False
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < oradius ** 2:
            return False
    return True


def is_valid_goal(x: float, y: float) -> bool:
    """True if (x, y) is a valid GOAL location (stricter clearance)."""
    if not (-7.8 < x < 7.8 and -7.8 < y < 7.8):
        return False
    if _in_wall(x, y, GOAL_CLEARANCE):
        return False
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < (oradius + GOAL_OBSTACLE_CLEARANCE) ** 2:
            return False
    return True


def has_obstacle_clearance(x: float, y: float, min_clearance: float) -> bool:
    for ox, oy, oradius in STATIC_OBSTACLES:
        if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < (oradius + min_clearance):
            return False
    return True


def clear_of_pedestrians(x: float, y: float) -> bool:
    for sx, sy in PED_STARTS:
        if (x - sx) ** 2 + (y - sy) ** 2 < PED_SPAWN_CLEARANCE ** 2:
            return False
    return True


def _snap_door_spawn(x: float, y: float) -> tuple[float, float]:
    """Snap spawn near door gaps to the gap center (matches train.py)."""
    if 4.85 < y < 6.15:
        y = 5.50
    elif -6.15 < y < -4.85:
        y = -5.50
    return x, y


# ==============================================================================
# Sampling
# ==============================================================================
def sample_spawn_in_zone(zone_name: str, rng: np.random.Generator,
                         max_tries: int = 500) -> Optional[tuple[float, float, float]]:
    """Sample a validated (x, y, yaw) spawn point in a zone.

    Returns None if no valid point found in max_tries (should not happen for
    zones with normal size + geometry).

    Yaw is uniform — the training code sets yaw = bearing_to_goal + noise,
    which will be applied at the call site once the goal is known.
    """
    if zone_name not in ZONE_BOXES:
        raise ValueError(f"Unknown zone: {zone_name}. Known: {list(ZONE_BOXES)}")

    xmin, xmax, ymin, ymax = ZONE_BOXES[zone_name]

    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))

        if not is_navigable(x, y):
            continue
        if zone_name == "corridor" and not clear_of_pedestrians(x, y):
            continue
        if not has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            continue

        x, y = _snap_door_spawn(x, y)
        if not is_navigable(x, y) or not has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            continue

        yaw = float(rng.uniform(-np.pi, np.pi))
        return (x, y, yaw)

    return None


def sample_goal_in_zone(zone_name: str, spawn_xy: tuple[float, float],
                        rng: np.random.Generator,
                        max_tries: int = 200,
                        max_distance: float = 8.0) -> Optional[tuple[float, float]]:
    """Sample a validated goal in a zone, at least MIN_SPAWN_GOAL_DIST from spawn."""
    if zone_name not in ZONE_BOXES:
        raise ValueError(f"Unknown zone: {zone_name}")

    xmin, xmax, ymin, ymax = ZONE_BOXES[zone_name]
    sx, sy = spawn_xy

    for _ in range(max_tries):
        gx = float(rng.uniform(xmin, xmax))
        gy = float(rng.uniform(ymin, ymax))
        d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
        if MIN_SPAWN_GOAL_DIST < d <= max_distance and is_valid_goal(gx, gy):
            return (gx, gy)

    return None


def bearing_yaw(spawn_xy: tuple[float, float], goal_xy: tuple[float, float],
                rng: np.random.Generator, noise_rad: float = np.pi / 6) -> float:
    """Yaw = bearing to goal + uniform noise ±noise_rad — same as train.py."""
    sx, sy = spawn_xy
    gx, gy = goal_xy
    bearing = float(np.arctan2(gy - sy, gx - sx))
    return bearing + float(rng.uniform(-noise_rad, noise_rad))


def compute_optimal_length(spawn_xy: tuple[float, float],
                           goal_xy: tuple[float, float]) -> float:
    """Straight-line optimal length (SPL denominator)."""
    return float(np.hypot(goal_xy[0] - spawn_xy[0], goal_xy[1] - spawn_xy[1]))


# ==============================================================================
# World-wide sampling — for scenario-based eval (no zone restrictions)
# ==============================================================================
def which_zone(x: float, y: float) -> Optional[str]:
    """Return zone name containing (x, y), or None if outside all zones."""
    for name, (xmin, xmax, ymin, ymax) in ZONE_BOXES.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return name
    return None


def sample_spawn_world_wide(rng: np.random.Generator,
                            max_tries: int = 500
                            ) -> Optional[tuple[float, float, float]]:
    """Sample a valid spawn ANYWHERE in the hospital, fully uniform.

    No zone snapping: petite_1/petite_2 are sampled uniformly like every
    other zone, using plain rejection sampling against the same validators
    as everywhere else. Yaw is uniform; caller can override with
    bearing-to-goal.
    """
    for _ in range(max_tries):
        x = float(rng.uniform(HOSPITAL_X_MIN, HOSPITAL_X_MAX))
        y = float(rng.uniform(HOSPITAL_Y_MIN, HOSPITAL_Y_MAX))

        if not is_navigable(x, y):
            continue
        if not has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            continue
        if not clear_of_pedestrians(x, y):
            continue

        yaw = float(rng.uniform(-np.pi, np.pi))
        return (x, y, yaw)

    return None


# ==============================================================================
# Zone-controlled sampling — for 70/30 intra vs cross-zone eval split
# ==============================================================================
# All zones are eligible for both intra- and cross-zone sampling, including
# the petites. There's no more snapping to a fixed room center: spawn/goal
# points are drawn uniformly inside the zone box via rejection sampling
# (sample_spawn_in_zone / sample_goal_in_zone), exactly like corridor and
# longue zones. Petite boxes are ~2.0 x 2.5 m (diagonal ~3.2 m), so a valid
# pair respecting MIN_SPAWN_GOAL_DIST (2.0 m) exists but is rarer — max_tries
# on sample_goal_in_zone (200) comfortably covers the rejection rate.
ALL_ZONES_LIST = ["corridor", "longue_1", "longue_2", "petite_1", "petite_2"]
INTRA_ZONE_CANDIDATES = ALL_ZONES_LIST


def sample_intra_zone_pair(rng: np.random.Generator, max_tries: int = 100):
    """Sample (spawn, goal) both in the same zone, uniformly (any zone).

    Returns (spawn_xyyaw, goal_xy, spawn_zone, goal_zone) or None on failure.
    """
    for _ in range(max_tries):
        zone = INTRA_ZONE_CANDIDATES[int(rng.integers(len(INTRA_ZONE_CANDIDATES)))]
        spawn = sample_spawn_in_zone(zone, rng)
        if spawn is None:
            continue
        goal = sample_goal_in_zone(zone, spawn[:2], rng)
        if goal is None:
            continue
        return spawn, goal, zone, zone
    return None


def sample_cross_zone_pair(rng: np.random.Generator, max_tries: int = 100):
    """Sample (spawn, goal) in different zones, uniformly (any of the 5 zones).

    Returns (spawn_xyyaw, goal_xy, spawn_zone, goal_zone) or None on failure.
    """
    for _ in range(max_tries):
        spawn_zone = ALL_ZONES_LIST[int(rng.integers(len(ALL_ZONES_LIST)))]
        others = [z for z in ALL_ZONES_LIST if z != spawn_zone]
        goal_zone = others[int(rng.integers(len(others)))]

        spawn = sample_spawn_in_zone(spawn_zone, rng)
        if spawn is None:
            continue
        goal = sample_goal_in_zone(goal_zone, spawn[:2], rng)
        if goal is None:
            continue
        return spawn, goal, spawn_zone, goal_zone
    return None