"""End-to-end training loop for CM-GAP_SAC (v0.8).

Curriculum (task + goal-distance combined):
    The hospital is split into spawn zones. A task curriculum unlocks zones
    progressively; within each zone a goal-distance curriculum grows the
    target range. A weighted rehearsal keeps old zones alive (60% on the
    newest zone, 40% on previously-mastered zones) to avoid catastrophic
    forgetting.

    Phases (by global step):
      P1  [0,150k)   : corridor only
      P2  [150k,350k): + longue_1   (x∈[-8,-3], y∈[-7.8,-3.2], div_bot, wall_west)
      P3  [350k,600k): + petite_1, petite_2 (x∈[3.5,8], both y bands)
      P4  [600k,...) : + longue_2   (x∈[-8,-3], y∈[3.2,7.8], div_top, wall_east)

    SDF axes: x = north(+8)/south(-8) vertical, y = east(+8)/west(-8) horizontal.

Other features (carried from v0.7):
    - Door spawn snap to gap centers (y=±5.50).
    - Spawn grace: first SPAWN_GRACE_STEPS steps ignore collisions.
    - Shield debug log (u_sac vs u_safe, nactive, omega_decay).
    - True rolling-mean metrics + full append-only CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cm_gap_sac_navigation.utils.config_loader import load_config


# ======================================================================
# Hospital geometry — matches hospital_v05b.sdf
# ======================================================================
HOSPITAL_X_MIN, HOSPITAL_X_MAX = -7.3, 7.3
HOSPITAL_Y_MIN, HOSPITAL_Y_MAX = -7.3, 7.3
MIN_SPAWN_GOAL_DIST = 2.0
SPAWN_GRACE_STEPS = 5
SPAWN_OBSTACLE_CLEARANCE = 0.8     # 80cm, > active_radius (0.40)

# Network reset configuration (Nikishin et al. 2022, ICML).
# Resets the LAST TWO Linear layers of each Q-head + LayerNorms, keeping
# the first Linear that encodes (xi, action) → hidden. Also resets the
# target critic in sync and clears the critic optimizer state to avoid
# stale Adam momentum on freshly-initialized weights.
RESET_CRITIC_EVERY = 10**9     # apply reset every N global steps
SOCIAL_GATE_RAMP_STEPS = 20000   # gate B ramps 0→1 after resume
RESET_AFTER_STEP = 90000        # don't reset before this step (let warmup learn)
RESET_BEFORE_STEP = 500000      # NEW: stop resets after this step
CROSS_ZONE_START_STEP = 150000

# Consolidation phase: bias toward intra-zone goals for final training
CONSOLIDATION_START_STEP = 1400000        # activate consolidation from this step
CONSOLIDATION_INTRA_ZONE_BIAS = 0.80      # 70% intra-zone during consolidation
ROBOT_MARGIN = 0.25
# Goal clearance: minimum distance from goal CENTER to nearest wall/obstacle.
# Must be > goal_radius (0.50) + collision_radius_static (0.20) + safety margin
# so the robot's success zone doesn't intersect the wall's safety zone.
# Otherwise the robot has to graze the wall to reach the goal — guaranteed
# collisions or chronic shield interventions.
GOAL_CLEARANCE = 1.00              # ← passé de 0.55 à 1.00 (vrai dégagement mur)
GOAL_OBSTACLE_CLEARANCE = 0.70     # ← NEW: marge au-delà du rayon d'obstacle

# Corridor sub-bands for stratified spawn rotation (Problem 1).
CORRIDOR_SUBZONES_X = [(-6.5, -2.0), (-2.0, 2.0), (2.0, 6.5)]  # W / M / E

# Corridor waypoint zones used as proxy target for cross-room trips (Problem 2).
CORRIDOR_EAST_X = (3.5, 5.8)
CORRIDOR_WEST_X = (-5.8, -3.5)


WALLS = [
    (-8.1, -7.9, -8.0, 8.0),       # wall_south
    ( 7.9,  8.1, -8.0, 8.0),       # wall_north
    (-8.0,  8.0, -8.1, -7.9),      # wall_west
    (-8.0,  8.0,  7.9,  8.1),      # wall_east
    (-5.5, 5.5,  2.9, 3.1),      # corridor_wall_1
    (-5.5, 5.5, -3.1, -2.9),     # corridor_wall_2
    (1.9, 2.1,  3.15,  4.85),      # div_left_top_1
    (1.9, 2.1,  6.15,  7.85),      # div_left_top_2
    (1.9, 2.1, -4.85, -3.15),      # div_left_bot_1
    (1.9, 2.1, -7.85, -6.15),      # div_left_bot_2
]

STATIC_OBSTACLES = [
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

PED_SPAWN_CLEARANCE = 0.8
PED_STARTS = [
    (5.43, 1.62), (-3.92, 1.18), (0.55, -1.96), (-6.75, -1.77),
    (2.75, -6.50), (-1.65, -4.20), (2.75, 6.50), (-1.76, 4.20),
]

# Spawn zone boxes (xmin, xmax, ymin, ymax).
# Constraint: robot spawn must be at least active_radius=1.0m from any wall
# so the CBF shield doesn't activate immediately at reset. Otherwise the
# shield clamps v=0 from the first step and the robot can never explore.
ZONE_BOXES = {
    "corridor": (-6.5, 6.5, -1.5, 1.5),
    "longue_1": (-7.0, -4.0, -7.0, -4.0),
    "petite_1": (4.5, 6.5, -6.5, -4.0),     # x_max: 7.0 → 6.5, y_min: -7.0 → -6.5
    "petite_2": (4.5, 6.5,  4.0,  6.5),     # idem symétrique
    "longue_2": (-7.0, -4.0, 4.0,  7.0),
}

# Safe fallback goals: pre-validated points roughly at zone centers,
# used when the random goal sampler exhausts its 100 attempts. Each
# zone has multiple candidates so the goal_sampler can pick one that
# satisfies MIN_SPAWN_GOAL_DIST from the current spawn.
#
# These candidates are validated at startup against _is_valid_goal()
# to ensure they don't fall inside walls or obstacles. Invalid ones
# are silently filtered out.
ZONE_SAFE_GOALS = {
    "corridor": [
        (-5.0,  0.0),
        ( 0.0, -1.0),
        ( 4.5, -1.5),
        (-4.5, -1.5),
    ],
    "longue_1": [
        (-4.0, -6.5),
        (-4.0, -6.8),
    ],
    "longue_2": [
        (-4.0,  6.5),
        (-4.0,  6.8),
    ],
    "petite_1": [
        ( 5.0, -5.8),   # NEW: éloigné de bedside_table_2 (6.5, -5.0)
        ( 5.5, -6.5),   # NEW: déjà dans corridor d'approche
        ( 4.8, -6.0),   # NEW: bas-gauche du coin sud-est
    ],
    "petite_2": [
        ( 5.0,  5.8),   # NEW: éloigné de bedside_table_1 (6.5, 5.0)
        ( 5.5,  6.5),   # NEW: déjà dans corridor d'approche
        ( 4.8,  6.0),   # NEW: haut-gauche du coin nord-est
        ],
    }

# Curriculum-distance: per-zone max distance grows linearly from
# CURRICULUM_START_MAX to the zone's full diameter over CURRICULUM_RAMP_STEPS,
# starting from the moment the zone is unlocked. No discrete stages, no jumps.
# Designed to avoid the catastrophic-forgetting spikes observed in earlier runs
# when the max_distance jumped from 5.0 to "unlimited" at step 100k.
CURRICULUM_START_MAX = 4.0
CURRICULUM_RAMP_STEPS = 300000

# Zone-specific final max distance (matches actual ZONE_BOXES geometry).
ZONE_FINAL_MAX = {
    "corridor": 10.0,   # corridor diagonal ≈ 13.8m
    "longue_1": 6.0,    # longue zones ≈ 6.8m diagonal
    "longue_2": 4.5,
    "petite_1": 5.0,    # petite zones ≈ 5.8m diagonal
    "petite_2": 5.0,
}

# Zone unlock steps (used as offset for curriculum age).
ZONE_UNLOCK_STEP = {
    "corridor": 0,
    "longue_1": 150000,
    "petite_1": 350000,
    "petite_2": 350000,
    "longue_2": 600000,
}

# Mixed-distance rehearsal: 40% of goals stay within half the current ceiling
# so the policy keeps practicing shorter goals while the ceiling rises.
SHORT_GOAL_MIX_FRAC = 0.40

# Task curriculum phases: (step_threshold, current_zone_label, unlocked_zones_list).
# Used by _phase_for_step() and _weighted_spawn_zone().
TASK_PHASES = [
    (150000,  "corridor", ["corridor"]),
    (350000,  "longue_1", ["corridor", "longue_1"]),
    (600000,  "petite",   ["corridor", "longue_1", "petite_1", "petite_2"]),
    (10**9,   "longue_2", ["corridor", "longue_1", "petite_1", "petite_2", "longue_2"]),
]

# ======================================================================
# Fixed evaluation scenarios — DECOUPLED from training curriculum.
# Same scenarios at every eval call, so success/collision curves are
# directly comparable across the entire training run. Covers all 4 zones
# even when locked (failure rate on locked zones is expected to decline
# after their unlock step, providing a clear catastrophic-forgetting signal).
#
# Format: (spawn_x, spawn_y, spawn_yaw, goal_x, goal_y, label)
# All points validated as free-space against current STATIC_OBSTACLES and WALLS.
# ======================================================================
EVAL_SCENARIOS = [
    # --- Corridor (always available) ---
    (-5.0,  0.0,  0.0,        4.5, -1.5, "corridor_long_east"),
    ( 4.5, -1.5,  np.pi,     -5.0,  0.0, "corridor_long_west"),
    (-4.5, -1.5,  0.0,        4.5,  0.0, "corridor_mid_long"),
    ( 0.0,  0.0,  np.pi,     -4.5,  1.5, "corridor_short_west"),
    (-5.0,  0.0, -np.pi/2,    4.5,  1.5, "corridor_yaw_offset"),

    # --- longue_1 (unlocks at 150k) ---
    (-5.5, -5.5,  np.pi/4,   -4.0, -4.5, "longue_1_intra"),         # CORRIGÉ
    (-5.0, -6.5,  np.pi/2,   -5.0,  0.0, "longue_1_to_corridor"),

    # --- longue_2 (unlocks at 600k) ---
    (-5.5,  5.5, -np.pi/4,   -4.0,  4.5, "longue_2_intra"),         # CORRIGÉ
    (-5.0,  6.5, -np.pi/2,   -5.0,  0.0, "longue_2_to_corridor"),

    # --- petite_1 (unlocks at 350k) ---
    ( 5.5, -6.5,  np.pi,      4.5, -1.5, "petite_1_to_corridor"),

    # --- petite_2 (unlocks at 350k) ---
    ( 5.5,  6.5,  np.pi,      4.5, -1.5, "petite_2_to_corridor"),
]
# Rehearsal weighting: fraction of spawns on the newest (current) zone.
# The remaining (1 - REHEARSAL_CURRENT_FRAC) goes to previously-mastered zones.
REHEARSAL_CURRENT_FRAC = 0.60
UNIFORM_REHEARSAL_STEP = 1_100_000

# === Cross-zone goal injection (curriculum-gated) ===
CROSS_ZONE_GOAL_FRAC_MAX = 0.15
CROSS_ZONE_RAMP_STEPS = 300000
CROSS_ZONE_START_STEP = 150000   # début ramp = unlock de longue_1
CROSS_ZONE_SIMPLE_RATIO = 0.7    # 70% room↔corridor, 30% room↔room
CROSS_ROOM_MIN_STEP = 400000     # cross-room (2 portes) attend 400k
MAX_CROSS_DIST = 7.0

# ======================================================================
# Geometry helpers
# ======================================================================
def _in_wall(x: float, y: float, margin: float) -> bool:
    for xmin, xmax, ymin, ymax in WALLS:
        if (xmin - margin < x < xmax + margin and
                ymin - margin < y < ymax + margin):
            return True
    return False


def _is_navigable(x: float, y: float) -> bool:
    if not (-7.8 < x < 7.8 and -7.8 < y < 7.8):
        return False
    if _in_wall(x, y, ROBOT_MARGIN):
        return False
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < oradius ** 2:
            return False
    return True


def _is_valid_goal(x: float, y: float) -> bool:
    if not (-7.8 < x < 7.8 and -7.8 < y < 7.8):
        return False
    if _in_wall(x, y, GOAL_CLEARANCE):
        return False
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < (oradius + GOAL_OBSTACLE_CLEARANCE) ** 2:
            return False
    return True


def _clear_of_pedestrians(x: float, y: float) -> bool:
    for sx, sy in PED_STARTS:
        if (x - sx) ** 2 + (y - sy) ** 2 < PED_SPAWN_CLEARANCE ** 2:
            return False
    return True



def _sample_in_corridor_subband(rng, xmin, xmax, max_tries=300):
    """Like _sample_in_zone('corridor') but constrained to a sub-band in x."""
    ymin, ymax = -1.5, 1.5
    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if not _is_navigable(x, y):
            continue
        if not _clear_of_pedestrians(x, y):
            continue
        if not _has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            continue
        return (x, y)
    return None


def _corridor_waypoint_toward(target_zone: str, rng, max_tries: int = 30):
    """Pick a corridor waypoint biased toward `target_zone`'s corridor entry.

    Used for room→room cross-zone trips: instead of the (timeout-bound) target
    room itself, we use a corridor point on the target's side. The robot
    practices the exit-and-traverse-in-direction subsegment within the time
    budget. Subsequent corridor→room episodes (existing code path) cover the
    other half.
    """
    if target_zone in ("petite_1", "petite_2"):
        x_range = CORRIDOR_EAST_X
    elif target_zone in ("longue_1", "longue_2"):
        x_range = CORRIDOR_WEST_X
    else:
        x_range = (-2.0, 2.0)

    if target_zone.endswith("_2"):       # north rooms → bias north of corridor
        y_range = (0.3, 1.3)
    elif target_zone.endswith("_1"):     # south rooms → bias south of corridor
        y_range = (-1.3, -0.3)
    else:
        y_range = (-1.0, 1.0)

    for _ in range(max_tries):
        gx = float(rng.uniform(*x_range))
        gy = float(rng.uniform(*y_range))
        if _is_valid_goal(gx, gy) and _has_obstacle_clearance(gx, gy, 0.5):
            return (gx, gy)
    # Fallback: a known-valid corridor safe goal on the target side
    if target_zone in ("petite_1", "petite_2"):
        return (4.5, -1.5 if target_zone.endswith("_1") else 1.5) if False else (4.5, -1.5)
    return (-4.5, -1.5)

def _has_obstacle_clearance(x: float, y: float, min_clearance: float) -> bool:
    """Check that point (x,y) is at least min_clearance meters from all static obstacles."""
    for ox, oy, oradius in STATIC_OBSTACLES:
        if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < (oradius + min_clearance):
            return False
    return True

def _snap_door_spawn(x: float, y: float) -> tuple:
    if 4.85 < y < 6.15:
        y = 5.50
    elif -6.15 < y < -4.85:
        y = -5.50
    return x, y


def _phase_for_step(step: int):
    for thr, cur, unlocked in TASK_PHASES:
        if step < thr:
            return cur, unlocked
    return TASK_PHASES[-1][1], TASK_PHASES[-1][2]


def _local_goal_max(zone: str, step: int) -> tuple:
    """Smooth per-zone curriculum-distance, gated by zone LOCAL age.
    Returns (max_d_long, max_d_short):
      - max_d_long: current ceiling, linearly interpolated from
        CURRICULUM_START_MAX to ZONE_FINAL_MAX[zone] over CURRICULUM_RAMP_STEPS
      - max_d_short: rehearsal bound = half the current ceiling, used by the
        goal_sampler to keep practicing shorter goals.
    """
    final_max = ZONE_FINAL_MAX[zone]
    local_age = step - ZONE_UNLOCK_STEP[zone]
    if local_age <= 0:
        return CURRICULUM_START_MAX, CURRICULUM_START_MAX
    progress = min(1.0, local_age / CURRICULUM_RAMP_STEPS)
    max_d_long = CURRICULUM_START_MAX + progress * (final_max - CURRICULUM_START_MAX)
    max_d_short = max(CURRICULUM_START_MAX, max_d_long * 0.5)
    return max_d_long, max_d_short



def _cross_zone_frac(step: int) -> float:
    """Curriculum fraction of cross-zone goals at given step.
    Ramps from 0 at CROSS_ZONE_START_STEP to CROSS_ZONE_GOAL_FRAC_MAX
    over CROSS_ZONE_RAMP_STEPS.
    """
    if step < CROSS_ZONE_START_STEP:
        return 0.0
    progress = min(1.0, (step - CROSS_ZONE_START_STEP) / CROSS_ZONE_RAMP_STEPS)
    return CROSS_ZONE_GOAL_FRAC_MAX * progress

def _sample_in_zone(rng, zone: str, max_tries: int = 300):
    """Sample a navigable SPAWN point inside a zone box. Returns None on fail.
    Pedestrian clearance is enforced only in the corridor (peds live there).
    Spawn must also be at least SPAWN_OBSTACLE_CLEARANCE meters from any
    static obstacle, so the shield doesn't activate at episode start."""
    xmin, xmax, ymin, ymax = ZONE_BOXES[zone]
    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if not _is_navigable(x, y):
            continue
        if zone == "corridor" and not _clear_of_pedestrians(x, y):
            continue
        # NEW: enforce minimum clearance from static obstacles
        if not _has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            continue
        x, y = _snap_door_spawn(x, y)
        if _is_navigable(x, y) and _has_obstacle_clearance(x, y, SPAWN_OBSTACLE_CLEARANCE):
            return (x, y)
    return None


def _sample_intra_zone_goal(rng, zone: str, max_tries: int = 100):
    """Sample a goal within the spawn zone's box, respecting validity."""
    if zone not in ZONE_BOXES:
        return None
    xmin, xmax, ymin, ymax = ZONE_BOXES[zone]
    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if _is_valid_goal(x, y):
            return (x, y)
    return None


def _weighted_spawn_zone(rng, step: int) -> str:
    """Pick a spawn zone with rehearsal weighting.

    - Before UNIFORM_REHEARSAL_STEP: 60% newest zone, 40% mastered zones.
    - After UNIFORM_REHEARSAL_STEP: uniform across all unlocked zones.
    """
    cur, unlocked = _phase_for_step(step)
    if len(unlocked) == 1:
        return unlocked[0]

    # Uniform mode: once all zones are mature, drop the newest-zone bias.
    if step >= UNIFORM_REHEARSAL_STEP:
        return unlocked[int(rng.integers(len(unlocked)))]

    # Weighted mode (unchanged): 60% newest, 40% mastered.
    cur_zones = ["petite_1", "petite_2"] if cur == "petite" else [cur]
    old = [z for z in unlocked if z not in cur_zones]
    if rng.random() < REHEARSAL_CURRENT_FRAC or not old:
        return cur_zones[int(rng.integers(len(cur_zones)))]
    return old[int(rng.integers(len(old)))]

def _sample_goal_point(rng, max_tries: int = 200):
    for _ in range(max_tries):
        x = float(rng.uniform(HOSPITAL_X_MIN, HOSPITAL_X_MAX))
        y = float(rng.uniform(HOSPITAL_Y_MIN, HOSPITAL_Y_MAX))
        if _is_valid_goal(x, y):
            return (x, y)
    return None





# ======================================================================
class _GracefulShutdown:
    def __init__(self) -> None:
        self.requested = False
        self._count = 0
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self._count += 1
        self.requested = True
        if self._count >= 2:
            print("\n[train] Second Ctrl-C: force exit.", flush=True)
            sys.exit(130)
        print("\n[train] Shutdown requested. Ctrl-C again to force.", flush=True)


# ======================================================================
class MetricsLogger:
    FIELDS = [
    "episode", "step", "outcome", "collision_type", "spawn_zone",
    "spawn_x", "spawn_y", "goal_x", "goal_y", "d_goal_init","n_waypoints", "waypoints_reached",  # NEW
    "reward", "steps",
    "success_roll", "collision_roll", "collision_ped_roll",
    "collision_static_roll", "shield_frozen_roll", "mean_reward_roll",
    "cbf_omega_decay_ep", "sps", "d_min_ped_terminal", "min_lidar_terminal", "collision_source",   
]

    def __init__(self, tb_writer, csv_path: Path, window: int = 100):
        self.tb = tb_writer
        self.window = window
        self.hist = deque(maxlen=window)
        self.csv_path = csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        self._fh = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def _roll(self, key_fn) -> float:
        if not self.hist:
            return 0.0
        return sum(key_fn(e) for e in self.hist) / len(self.hist)

    def log_episode(self, *, episode, step, outcome, collision_type, spawn_zone,
                spawn_x, spawn_y, goal_x, goal_y, d_goal_init, n_waypoints, waypoints_reached,   # NEW
                reward, steps, cbf_omega_decay_ep, sps, d_min_ped_terminal=float("nan"),      # NEW
                min_lidar_terminal=float("nan"),      # NEW
                collision_source=""):
        ep = {"outcome": outcome, "collision_type": collision_type, "reward": reward}
        self.hist.append(ep)
        succ = self._roll(lambda e: 1.0 if e["outcome"] == "success" else 0.0)
        coll = self._roll(lambda e: 1.0 if e["outcome"] == "collision" else 0.0)
        coll_p = self._roll(lambda e: 1.0 if e["collision_type"] == "pedestrian" else 0.0)
        coll_s = self._roll(lambda e: 1.0 if e["collision_type"] == "static" else 0.0)
        frozen = self._roll(lambda e: 1.0 if e["outcome"] == "shield_frozen" else 0.0)
        mean_r = self._roll(lambda e: e["reward"])
        if self.tb:
            self.tb.add_scalar("episode/reward", reward, step)
            self.tb.add_scalar("episode/length", steps, step)
            self.tb.add_scalar("episode/success", 1.0 if outcome == "success" else 0.0, step)
            self.tb.add_scalar("episode/collision", 1.0 if outcome == "collision" else 0.0, step)
            if cbf_omega_decay_ep is not None:
                self.tb.add_scalar("episode/cbf_omega_decay", cbf_omega_decay_ep, step)
            self.tb.add_scalar(f"rolling/success_rate_{self.window}", succ, step)
            self.tb.add_scalar(f"rolling/collision_rate_{self.window}", coll, step)
            self.tb.add_scalar(f"rolling/collision_ped_{self.window}", coll_p, step)
            self.tb.add_scalar(f"rolling/collision_static_{self.window}", coll_s, step)
            self.tb.add_scalar(f"rolling/shield_frozen_{self.window}", frozen, step)
            self.tb.add_scalar(f"rolling/mean_reward_{self.window}", mean_r, step)
            self.tb.add_scalar("perf/steps_per_second", sps, step)
        self._writer.writerow({
            "episode": episode, "step": step, "outcome": outcome,
            "collision_type": collision_type if collision_type else "",
            "spawn_zone": spawn_zone,
            "reward": f"{reward:.4f}", "steps": steps,
            "success_roll": f"{succ:.4f}", "collision_roll": f"{coll:.4f}",
            "collision_ped_roll": f"{coll_p:.4f}",
            "spawn_x": f"{spawn_x:.2f}",      # NEW
            "spawn_y": f"{spawn_y:.2f}",      # NEW
            "goal_x":  f"{goal_x:.2f}",       # NEW
            "goal_y":  f"{goal_y:.2f}",       # NEW
            "n_waypoints": n_waypoints,
            "waypoints_reached": waypoints_reached,
            "d_goal_init": f"{d_goal_init:.2f}",  # NEW
            "collision_static_roll": f"{coll_s:.4f}",
            "shield_frozen_roll": f"{frozen:.4f}",
            "mean_reward_roll": f"{mean_r:.4f}",
            "cbf_omega_decay_ep": (f"{cbf_omega_decay_ep:.4f}"
                                   if cbf_omega_decay_ep is not None else ""),
            "sps": f"{sps:.2f}",
            "d_min_ped_terminal": (f"{d_min_ped_terminal:.3f}"
                       if np.isfinite(d_min_ped_terminal) else ""),
            "min_lidar_terminal": (f"{min_lidar_terminal:.3f}"
                                if np.isfinite(min_lidar_terminal) else ""),
            "collision_source":   collision_source,
        })
        self._fh.flush()
        return {"success": succ, "collision": coll, "collision_ped": coll_p,
                "collision_static": coll_s, "shield_frozen": frozen, "mean_reward": mean_r}

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def make_optimizer_log(info, alpha: float) -> dict:
    return {
        "train/critic_loss": info.critic_loss, "train/actor_loss": info.actor_loss,
        "train/alpha_loss": info.alpha_loss, "train/alpha": alpha,
        "train/log_prob_mean": info.log_prob_mean, "train/q_min_mean": info.q_min_mean,
        "train/td_error_mean": info.td_error_mean, "train/td_error_max": info.td_error_max,
    }


def _validate_safe_goals() -> dict:
    """Filter ZONE_SAFE_GOALS to only the points that pass _is_valid_goal.

    Called once at startup. Raises if a zone has NO valid safe goal,
    which would mean the fallback is unusable for that zone.
    """
    validated = {}
    for zone, candidates in ZONE_SAFE_GOALS.items():
        valid = [(x, y) for (x, y) in candidates if _is_valid_goal(x, y)]
        if not valid:
            raise RuntimeError(
                f"Zone '{zone}' has no valid safe-goal fallback. "
                f"Tested candidates: {candidates}. Either widen "
                f"GOAL_CLEARANCE / GOAL_OBSTACLE_CLEARANCE or add new "
                f"candidates to ZONE_SAFE_GOALS."
            )
        validated[zone] = valid
        if len(valid) < len(candidates):
            invalid = [pt for pt in candidates if pt not in valid]
            print(f"[init] Zone '{zone}': filtered out invalid fallback "
                  f"goals {invalid}, kept {len(valid)}/{len(candidates)}",
                  flush=True)
    return validated


def evaluate_fixed_set(env, policy, shield, device, tb_writer, step,
                       scenarios=EVAL_SCENARIOS):
    """Run a fixed set of (spawn, goal) scenarios. Independent of curriculum.

    Returns aggregate metrics dict. Logs per-scenario and aggregate scalars to TB
    with prefix 'eval/'. Restores the training samplers before returning.
    """
    # Save the training samplers
    saved_spawn = env._spawn_sampler
    saved_goal = env._goal_sampler

    results = []
    for sx, sy, syaw, gx, gy, label in scenarios:
        env._spawn_sampler = lambda sx=sx, sy=sy, syaw=syaw: (sx, sy, syaw)
        env._goal_sampler = lambda gx=gx, gy=gy: (gx, gy)
        obs, info = env.reset()
           # DEBUG: verify spawn teleport actually worked
        rx, ry, _, _, _ = env._gz.get_robot_state()
        err = ((rx - sx)**2 + (ry - sy)**2) ** 0.5
        print(f"[eval-dbg] {label:32s}  req=({sx:+.1f},{sy:+.1f}) "
            f"got=({rx:+.1f},{ry:+.1f}) err={err:.2f}m", flush=True)
        
        ep_reward = 0.0
        ep_steps = 0
        outcome = "timeout"
        coll_type = ""
        max_steps = env.cfg.episode.max_steps
        for _ in range(max_steps):
            with torch.no_grad():
                action_t, _, _ = policy.act_from_obs(
                    obs, device=device, deterministic=True)
            action = action_t.cpu().numpy().astype(np.float32)
            if shield is not None:
                rx, ry, ryaw, _, _ = env._gz.get_robot_state()
                shield_res = shield.filter(
                    u_sac=action, robot_xy_yaw=(rx, ry, ryaw),
                    lidar_scan=obs["lidar"], pedestrians_rel=obs["pedestrians"],
                    ped_mask=obs["ped_mask"])
                action = shield_res.safe_action
            obs, r, term, trunc, info = env.step(action)
            ep_reward += float(r)
            ep_steps += 1
            if term or trunc:
                outcome = info.get("outcome", "timeout")
                coll_type = info.get("collision_type", "") or ""
                break
        results.append({
            "label": label, "outcome": outcome, "collision_type": coll_type,
            "reward": ep_reward, "steps": ep_steps,
        })

    # Restore training samplers
    env._spawn_sampler = saved_spawn
    env._goal_sampler = saved_goal

    # Aggregate
    n = len(results)
    succ = sum(1 for r in results if r["outcome"] == "success") / n
    coll = sum(1 for r in results if r["outcome"] == "collision") / n
    coll_p = sum(1 for r in results if r["collision_type"] == "pedestrian") / n
    coll_s = sum(1 for r in results if r["collision_type"] == "static") / n
    timeout = sum(1 for r in results if r["outcome"] == "timeout") / n
    mean_r = sum(r["reward"] for r in results) / n
    mean_s = sum(r["steps"] for r in results) / n

    # Per-zone aggregate (zone = label prefix before first '_')
    by_zone = {}
    for r in results:
        zone = r["label"].split("_")[0] + (
            "_" + r["label"].split("_")[1] if r["label"].split("_")[0] in ("longue", "petite") else ""
        )
        by_zone.setdefault(zone, []).append(1.0 if r["outcome"] == "success" else 0.0)

    print(f"\n[eval] step={step} | succ={succ*100:.1f}% coll={coll*100:.1f}% "
          f"(ped={coll_p*100:.0f}% stat={coll_s*100:.0f}%) "
          f"timeout={timeout*100:.0f}% meanR={mean_r:+.2f} meanS={mean_s:.0f}",
          flush=True)
    for zone, vals in by_zone.items():
        print(f"[eval]   {zone}: {sum(vals)/len(vals)*100:.0f}% ({len(vals)} eps)", flush=True)

    if tb_writer:
        tb_writer.add_scalar("eval/success_rate", succ, step)
        tb_writer.add_scalar("eval/collision_rate", coll, step)
        tb_writer.add_scalar("eval/collision_ped", coll_p, step)
        tb_writer.add_scalar("eval/collision_static", coll_s, step)
        tb_writer.add_scalar("eval/timeout_rate", timeout, step)
        tb_writer.add_scalar("eval/mean_reward", mean_r, step)
        tb_writer.add_scalar("eval/mean_steps", mean_s, step)
        for zone, vals in by_zone.items():
            tb_writer.add_scalar(f"eval/success_{zone}", sum(vals)/len(vals), step)

    return {"success": succ, "collision": coll, "mean_reward": mean_r}



# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train CM-GAP_SAC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--roll-window", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=None,
                    help="TB run subdir name (default: timestamp). "
                         "Pass the existing run name when resuming to continue curves.")
    args = parser.parse_args()

    from cm_gap_sac_navigation.envs.limo_gazebo_env import LimoGazeboEnv
    from cm_gap_sac_navigation.models.policy import build_policy_from_config
    from cm_gap_sac_navigation.rl.per_buffer import build_per_from_config
    from cm_gap_sac_navigation.rl.sac import build_sac_agent
    from cm_gap_sac_navigation.rl.safety_shield import build_shield_from_config
    from cm_gap_sac_navigation.training.eval import evaluate

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}", flush=True)

    # Validate safe-goal fallback points against current GOAL_CLEARANCE.
    # Raises if any zone has zero valid fallbacks (config error).
    safe_goals = _validate_safe_goals()
    print(f"[train] Validated safe-goal fallbacks: "
          f"{ {z: len(g) for z, g in safe_goals.items()} }", flush=True)

    total_steps = args.total_steps or cfg.training["total_steps"]
    eval_every  = args.eval_every  or cfg.training["eval_every"]
    save_every  = args.save_every  or cfg.training["save_every"]
    log_dir     = cfg.training["log_dir"]
    seed        = cfg.training["seed"]
    batch_size  = cfg.sac["batch_size"]
    warmup      = cfg.sac["warmup_steps"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tb_writer = None
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(log_dir) / run_name
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(run_dir))
            print(f"[train] TensorBoard: {run_dir}", flush=True)
        except ImportError:
            print("[train] tensorboard not installed.", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = MetricsLogger(
        tb_writer=tb_writer, csv_path=run_dir / "episodes.csv", window=args.roll_window)
    print(f"[train] CSV: {run_dir / 'episodes.csv'}", flush=True)

    print("[train] Building policy + critic...", flush=True)
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)
    print(f"[train] Building PER buffer (cap={cfg.per['capacity']})...", flush=True)
    buffer = build_per_from_config(cfg, action_dim=cfg.action.get("dim", 2), seed=seed)
    shield = build_shield_from_config(cfg)
    print(f"[train] CBF shield {'ENABLED' if shield else 'DISABLED'}.", flush=True)

    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(sd["agent"])
        if "buffer" in sd:
            buffer.load_state_dict(sd["buffer"])
            print(f"[train] Buffer restored: {len(buffer)} transitions, "
                  f"buffer step counter at {buffer._step}", flush=True)
        else:
            print(f"[train] WARNING: checkpoint has no buffer; "
                  f"will refill from scratch (1000 warmup steps will execute)", flush=True)
        start_step = int(sd.get("step", 0))
        print(f"[train] Resumed from {args.resume} at step {start_step}", flush=True)

    # ---- One-shot buffer reward patch: add r_social to stored rewards ----
    # r_social is a pure function of the POST-action obs (compute_reward
    # receives next_obs), so we read the _nxt_* tensors, not _obs_*.
    if args.resume and os.environ.get("PATCH_BUFFER_SOCIAL", "0") == "1":
        rcfg = cfg.reward
        n = len(buffer)
        with torch.no_grad():
            P = buffer._nxt_ped[:n]                       # (N, K, F)
            M = buffer._nxt_pmask[:n].bool()              # (N, K)
            px, py = P[..., 0], P[..., 1]
            vx, vy = P[..., 2], P[..., 3]
            d = torch.sqrt(px * px + py * py)             # (N, K)
            d = torch.where(M, d, torch.full_like(d, float("inf")))
            d_min, k_idx = d.min(dim=1)                   # (N,)
            near = torch.isfinite(d_min) & (d_min < rcfg.d_social)
            d_use = torch.where(near, d_min, torch.ones_like(d_min))  # no inf/nan
            ar = torch.arange(n, device=P.device)
            pxn, pyn = px[ar, k_idx], py[ar, k_idx]
            vxn, vyn = vx[ar, k_idx], vy[ar, k_idx]
            v_close = (-(pxn * vxn + pyn * vyn)
                       / d_use.clamp(min=1e-3)).clamp(min=0.0)
            r_soc = (-rcfg.alpha_social * v_close
                     * (1.0 - d_use / rcfg.d_social).pow(2))
            r_soc = torch.where(near & (v_close > 0), r_soc,
                                torch.zeros_like(r_soc))
            buffer._reward[:n] += r_soc
            n_patched = int((r_soc < 0).sum().item())
            mean_pen = float(r_soc[r_soc < 0].mean().item()) if n_patched else 0.0
        print(f"[patch] r_social applied to {n_patched}/{n} transitions "
              f"(mean penalty {mean_pen:+.4f})", flush=True)
    # ---- Environment --------------------------------------------------
    print("[train] Building env...", flush=True)
    rng = np.random.default_rng(seed)
    _step_tracker = [0]
    _last_spawn = [(0.0, 0.0)]
    _last_zone = ["corridor"]
    _cached_goal = [None]  # v0.9: goal pre-sampled by spawn_sampler

    _corridor_strat_idx = [0]   # rotates 0,1,2,0,1,2,... across W/M/E sub-bands

    def _sample_goal_for(sx: float, sy: float, zone: str, step: int) -> tuple:
        """Pure function: (spawn, zone, step) -> (gx, gy).
        Extracted from goal_sampler so spawn_sampler can call it and orient yaw."""
        max_d_long, max_d_short = _local_goal_max(zone, step)
        max_d = max_d_short if rng.random() < SHORT_GOAL_MIX_FRAC else max_d_long

        # === Consolidation phase: bias toward intra-zone for final training ===
        # Applied only after CONSOLIDATION_START_STEP. During this phase,
        # CONSOLIDATION_INTRA_ZONE_BIAS fraction of goals are sampled within
        # the spawn zone. Rebalances the artifact where uniform world sampling
        # produced ~87% cross-zone episodes.
        if step >= CONSOLIDATION_START_STEP:
            if rng.random() < CONSOLIDATION_INTRA_ZONE_BIAS:
                if zone in ZONE_BOXES:
                    xmin, xmax, ymin, ymax = ZONE_BOXES[zone]
                    for _ in range(50):
                        gx = float(rng.uniform(xmin, xmax))
                        gy = float(rng.uniform(ymin, ymax))
                        d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
                        if (_is_valid_goal(gx, gy)
                                and MIN_SPAWN_GOAL_DIST < d <= max_d):
                            return (gx, gy)
                    # If failed 50 tries, fall through to normal sampling below

        cur_phase, unlocked = _phase_for_step(step)
        other_zones = [z for z in unlocked if z != zone]
        rooms_unlocked = [z for z in unlocked if z != "corridor"]
        cross_frac = _cross_zone_frac(step)

        # === Cross-zone goal injection ===
        if other_zones and rng.random() < cross_frac:
            if zone == "corridor":
                reachable = []
                for rz in rooms_unlocked:
                    for (gx, gy) in safe_goals.get(rz, []):
                        d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
                        if MIN_SPAWN_GOAL_DIST < d <= MAX_CROSS_DIST:
                            reachable.append((gx, gy))
                if reachable:
                    return reachable[int(rng.integers(len(reachable)))]
            else:
                simple_only = step < CROSS_ROOM_MIN_STEP
                if simple_only or rng.random() < CROSS_ZONE_SIMPLE_RATIO:
                    if zone in ("petite_1", "petite_2"):
                        x_range = CORRIDOR_EAST_X
                    elif zone in ("longue_1", "longue_2"):
                        x_range = CORRIDOR_WEST_X
                    else:
                        x_range = (-5.5, 5.5)
                    for _ in range(30):
                        gx = float(rng.uniform(*x_range))
                        gy = float(rng.uniform(-1.3, 1.3))
                        d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
                        if (_is_valid_goal(gx, gy)
                                and _has_obstacle_clearance(gx, gy, 0.5)
                                and MIN_SPAWN_GOAL_DIST < d <= MAX_CROSS_DIST):
                            return (gx, gy)
                else:
                    target_pool = [z for z in rooms_unlocked if z != zone]
                    if target_pool:
                        tz = target_pool[int(rng.integers(len(target_pool)))]
                        gx, gy = _corridor_waypoint_toward(tz, rng)
                        d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
                        if MIN_SPAWN_GOAL_DIST < d <= MAX_CROSS_DIST:
                            return (gx, gy)

        # === Primary: respects max_d ===
        for _ in range(100):
            pt = _sample_goal_point(rng)
            if pt is None:
                continue
            gx, gy = pt
            d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
            if MIN_SPAWN_GOAL_DIST < d <= max_d:
                return (gx, gy)

        # === Secondary: CAPPED at 1.5x current ceiling (v0.9 bugfix) ===
        max_d_secondary = max_d_long * 1.5
        for _ in range(100):
            pt = _sample_goal_point(rng)
            if pt is not None:
                gx, gy = pt
                d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
                if MIN_SPAWN_GOAL_DIST < d <= max_d_secondary:
                    return (gx, gy)

        # === Tertiary: safe_goal fallback ===
        candidates = safe_goals.get(zone, safe_goals["corridor"])
        far_enough = [(gx, gy) for (gx, gy) in candidates
                    if ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5 > MIN_SPAWN_GOAL_DIST]
        pool = far_enough if far_enough else candidates
        return pool[int(rng.integers(len(pool)))]

    def spawn_sampler():
        step = _step_tracker[0]
        for _ in range(10):
            zone = _weighted_spawn_zone(rng, step)
            if zone == "corridor":
                sub_xmin, sub_xmax = CORRIDOR_SUBZONES_X[_corridor_strat_idx[0] % 3]
                _corridor_strat_idx[0] += 1
                pt = _sample_in_corridor_subband(rng, sub_xmin, sub_xmax)
                if pt is None:
                    pt = _sample_in_zone(rng, "corridor")
            else:
                pt = _sample_in_zone(rng, zone)
            if pt is not None:
                x, y = pt
                _last_spawn[0] = (x, y)
                _last_zone[0] = zone

                # v0.9: sample goal NOW so we can orient the robot toward it
                gx, gy = _sample_goal_for(x, y, zone, step)
                _cached_goal[0] = (gx, gy)

                # Yaw = bearing to goal + ±30° noise (recovery learning)
                # Previous behavior: hardcoded corridor-facing yaw → up to
                # 225° error on intra-zone goals, breaking Nav2 tracking.
                bearing = float(np.arctan2(gy - y, gx - x))
                yaw = bearing + float(rng.uniform(-np.pi / 6, np.pi / 6))
                return (x, y, yaw)
        _last_spawn[0] = (0.0, 0.0)
        _last_zone[0] = "corridor"
        _cached_goal[0] = None
        return (0.0, 0.0, 0.0)

    def goal_sampler():
        # v0.9: consume the goal pre-sampled by spawn_sampler
        if _cached_goal[0] is not None:
            g = _cached_goal[0]
            _cached_goal[0] = None
            return g
        # Fallback path (should rarely trigger — only if goal_sampler
        # is called out of the normal reset cycle, e.g. eval override)
        sx, sy = _last_spawn[0]
        return _sample_goal_for(sx, sy, _last_zone[0], _step_tracker[0])
    def update_curriculum_step(step):
        _step_tracker[0] = step

    env = LimoGazeboEnv(
        config_path=args.config, seed=seed,
        spawn_xy_yaw=spawn_sampler(), goal_sampler=goal_sampler)
    env._spawn_sampler = spawn_sampler

    # ---- Training loop ------------------------------------------------
    shutdown = _GracefulShutdown()
    print("[train] Initial reset...", flush=True)
    obs, info = env.reset()

    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    ep_omega_decay_sum = 0.0
    ep_omega_decay_n = 0
    consec_infeasible = 0
    max_consec_infeasible = cfg.cbf.get("max_consecutive_infeasible", 20)
    _prev_phase = None

    t_start = time.time()
    print(f"[train] Starting: {total_steps} steps, warmup={warmup}, "
          f"batch={batch_size}, roll={args.roll_window}", flush=True)

    for step in range(start_step, total_steps):
        update_curriculum_step(step)
        env.set_global_step(step)

        # Announce phase transitions.
        cur_phase, _ = _phase_for_step(step)
        if cur_phase != _prev_phase:
            print(f"[curriculum] step={step} -> phase '{cur_phase}'", flush=True)
            _prev_phase = cur_phase

        # ---- Sample action --------------------------------------------
        if step < warmup:
            action = env.action_space.sample().astype(np.float32)
            attn_entropy = 0.0
        else:
            action_t, _, ent_t = policy.act_from_obs(
                obs, device=device, deterministic=False)
            action = action_t.cpu().numpy().astype(np.float32)
            attn_entropy = float(ent_t.cpu().item())

        # ---- CBF safety shield ----------------------------------------
        cbf_active = cbf_modified = cbf_infeasible = False
        if shield is not None and step >= warmup:
            action_before = action.copy()
            x, y, yaw, _, _ = env._gz.get_robot_state()
            shield_result = shield.filter(
                u_sac=action, robot_xy_yaw=(x, y, yaw),
                lidar_scan=obs["lidar"], pedestrians_rel=obs["pedestrians"],
                ped_mask=obs["ped_mask"])
            action = shield_result.safe_action
            cbf_active = shield_result.n_active_constraints > 0
            cbf_modified = shield_result.was_modified
            cbf_infeasible = shield_result.infeasible
            if cbf_active:
                ep_omega_decay_sum += float(shield_result.omega_decay)
                ep_omega_decay_n += 1
            consec_infeasible = consec_infeasible + 1 if cbf_infeasible else 0
            if step % 20 == 0:
                print(
                    f"[shield] step={step} nactive={shield_result.n_active_constraints} "
                    f"u_sac=({action_before[0]:+.3f},{action_before[1]:+.3f}) "
                    f"u_safe=({action[0]:+.3f},{action[1]:+.3f}) "
                    f"mod={cbf_modified} infeas={cbf_infeasible} "
                    f"odec={shield_result.omega_decay:.3f}", flush=True)

        # ---- Step env -------------------------------------------------
        # in the training loop, before env.step / or once per episode:
        # Gate B ramp (A is constant; buffer is patched at full alpha)
        env.cfg.reward.social_gate_blend = min(
            1.0, max(0.0, (step - start_step) / SOCIAL_GATE_RAMP_STEPS))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_flag = float(terminated)
        info["cbf_active"] = cbf_active
        info["cbf_modified"] = cbf_modified
        info["cbf_infeasible"] = cbf_infeasible

        # Spawn grace: ignore collisions in the first N steps after reset.
        if episode_steps < SPAWN_GRACE_STEPS:
            if terminated and info.get("outcome") == "collision":
                terminated = False
                done_flag = 0.0
                reward = float(reward) - float(cfg.reward.r_collision)
                info.pop("outcome", None)
                info.pop("collision_type", None)

        emergency_stop = cfg.cbf.get("emergency_stop_on_infeasible", True)
        if emergency_stop and consec_infeasible >= max_consec_infeasible:
            truncated = True
            info["outcome"] = "shield_frozen"
            consec_infeasible = 0

        if cbf_modified and step >= warmup:
            shield_pen = -cfg.reward.alpha_shield
            reward = float(reward) + shield_pen
            info["r_shield"] = shield_pen
        else:
            info["r_shield"] = 0.0

        buffer.add(obs=obs, action=action, reward=float(reward),
                   next_obs=next_obs, done=done_flag, attention_entropy=attn_entropy)
        episode_reward += float(reward)
        episode_steps += 1

        # ---- Episode end ----------------------------------------------
        if terminated or truncated:
            episode_count += 1
            outcome = info.get("outcome", "timeout")
            ep_omega_decay = (ep_omega_decay_sum / ep_omega_decay_n
                              if ep_omega_decay_n > 0 else None)
            elapsed = time.time() - t_start
            sps = (step - start_step + 1) / max(1.0, elapsed)
            rolls = metrics_logger.log_episode(
                episode=episode_count, step=step, outcome=outcome,
                collision_type=info.get("collision_type"),
                spawn_zone=_last_zone[0], reward=episode_reward,
                spawn_x=float(_last_spawn[0][0]),
                spawn_y=float(_last_spawn[0][1]),    # NEW
                goal_x=float(env._goal_xy[0]),       # NEW
                goal_y=float(env._goal_xy[1]),       # NEW
                d_goal_init=float(np.linalg.norm(np.array(env._goal_xy) - np.array(_last_spawn[0]))),
                n_waypoints=len(env._waypoints) if hasattr(env, '_waypoints') else 0,  # NEW
                waypoints_reached=env._current_waypoint_idx if hasattr(env, '_current_waypoint_idx') else 0,  # NEW
                steps=episode_steps, cbf_omega_decay_ep=ep_omega_decay, sps=sps, d_min_ped_terminal=float(info.get("d_min_ped", float("nan"))),
                min_lidar_terminal=float(info.get("min_lidar", float("nan"))),
                collision_source=info.get("collision_source", ""),
                )
            if episode_count % 20 == 0:
                od = f"{ep_omega_decay:.2f}" if ep_omega_decay is not None else "n/a"
                print(
                    f"[train] step={step:7d} ep={episode_count:5d} "
                    f"succ={rolls['success']*100:5.1f}% coll={rolls['collision']*100:5.1f}% "
                    f"(ped={rolls['collision_ped']*100:.0f}% stat={rolls['collision_static']*100:.0f}%) "
                    f"frozen={rolls['shield_frozen']*100:4.0f}% meanR={rolls['mean_reward']:+7.2f} "
                    f"odec={od} alpha={agent.alpha.item():.4f} sps={sps:.1f}", flush=True)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            ep_omega_decay_sum = 0.0
            ep_omega_decay_n = 0
        else:
            obs = next_obs

        # ---- Gradient update ------------------------------------------
        if step >= warmup and buffer.can_sample(batch_size):
            batch = buffer.sample(batch_size)
            update_info, td_errors, entropies = agent.update(batch)
            buffer.update_priorities(indices=batch["indices"],
                                     td_errors=td_errors, entropies=entropies)
            if tb_writer and step % 100 == 0:
                logs = make_optimizer_log(update_info, agent.alpha.item())
                logs["per/beta"] = batch["beta"]
                for k, v in logs.items():
                    tb_writer.add_scalar(k, v, step)

        # ---- Periodic checkpoint (BEFORE reset to save trained weights) ----
        # IMPORTANT: save must happen BEFORE the critic reset, otherwise
        # the checkpoint contains a half-randomized critic and resume looks
        # like a fresh start. This is the v0.9 ordering fix.
        if step > 0 and step % save_every == 0:
            ckpt_path = Path(log_dir) / f"checkpoint_step{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "agent":  agent.state_dict(),
                "buffer": buffer.state_dict(),
                "step":   step,
            }, ckpt_path)
            print(f"[train] Saved: {ckpt_path} (with buffer, ~1.3GB)", flush=True)

        # ---- Periodic critic reset (Nikishin 2022) --------------------
        # Comes AFTER save: the saved checkpoint contains the
        # fully-trained critic, while the live agent moves on with the
        # reset applied. Resume restores the pre-reset trained critic.
        if (step > RESET_AFTER_STEP
                and step < RESET_BEFORE_STEP
                and step % RESET_CRITIC_EVERY == 0
                and step >= warmup):
            n_reset = agent.reset_critic_late_layers()
            print(f"[reset] step={step} reset {n_reset} critic layers + "
                  f"target sync + optimizer cleared", flush=True)
            if tb_writer:
                tb_writer.add_scalar("train/critic_reset", 1.0, step)

        # ---- Periodic evaluation (AFTER reset to measure post-reset) ----
        if ((not args.no_eval) and step > 0
                and step % eval_every == 0 and step >= warmup):
            print(f"\n[train] Evaluation at step {step}...", flush=True)
            metrics = evaluate_fixed_set(env, policy, shield, device, tb_writer, step)
            print(f"[train] Eval result: succ={metrics['success']*100:.1f}% "
                f"coll={metrics['collision']*100:.1f}% meanR={metrics['mean_reward']:+.2f}",
                flush=True)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            ep_omega_decay_sum = 0.0
            ep_omega_decay_n = 0

        if shutdown.requested:
            ckpt_path = Path(log_dir) / f"checkpoint_interrupted_step{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "agent":  agent.state_dict(),
                "buffer": buffer.state_dict(),
                "step":   step,
            }, ckpt_path)
            print(f"[train] Emergency save: {ckpt_path}", flush=True)
            break

    final_ckpt = Path(log_dir) / "checkpoint_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "agent":  agent.state_dict(),
        "buffer": buffer.state_dict(),
        "step":   step,
    }, final_ckpt)
    print(f"[train] Final: {final_ckpt}", flush=True)
    metrics_logger.close()
    if tb_writer:
        tb_writer.close()
    env.close()
    print(f"[train] Done. {(time.time() - t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()