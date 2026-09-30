"""
CM-GAP-SAC Eval — Scenario Manifest Expander (v3 — scenario-based)

Expands eval_config.yaml into a per-episode JSONL manifest. New structure:
    scenario × method × ep_idx  (no more zones/cross_zones)

Spawn/goal are sampled world-wide, snapped to petite-room centers if they
land inside a petite room.

Paired seeding: cell = (scenario_id, ep_idx). All methods on the same
(scenario, ep_idx) share the same spawn/goal — enables paired McNemar.

Usage:
    python expand_scenarios.py --config eval_config.yaml --output manifest.jsonl
    python expand_scenarios.py --config eval_config.yaml --smoke
    python expand_scenarios.py --config eval_config.yaml --filter-method nav2_dwb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency — install with:  pip install pyyaml")


# ==============================================================================
# Data model
# ==============================================================================
@dataclass(frozen=True)
class EpisodeSpec:
    """One concrete evaluation episode."""
    episode_id: str
    scenario_id: str
    n_pedestrians: int
    pedestrian_speed_mps: float
    method: str
    checkpoint: Optional[str]
    episode_seed: int              # cell-scoped: (scenario_id, ep_idx)
    episode_idx_within_scenario: int
    # Precomputed geometry (shared across methods on same (scenario, ep_idx))
    spawn_xy_yaw: tuple[float, float, float]
    goal_xy: tuple[float, float]
    optimal_path_length: float
    # Zone assignment (70% intra / 30% cross split per scenario)
    zone_type: str                 # "intra" or "cross"
    spawn_zone: str
    goal_zone: str
    


# ==============================================================================
# Deterministic seeding
# ==============================================================================
def derive_seed(master_seed: int, *parts) -> int:
    key = f"{master_seed}::" + "::".join(str(p) for p in parts)
    h = hashlib.blake2b(key.encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ==============================================================================
# Config validation
# ==============================================================================
def validate_config(cfg: dict) -> None:
    required = ["experiment", "global", "worlds", "scenarios", "robot",
                "methods", "metrics", "statistics"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required top-level key: {k}")

    scenario_ids = [s["id"] for s in cfg["scenarios"]]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate scenario id detected")

    for s in cfg["scenarios"]:
        if "n_pedestrians" not in s or "pedestrian_speed_mps" not in s:
            raise ValueError(f"Scenario {s.get('id')} missing n_pedestrians or speed")
        if s["n_pedestrians"] < 0:
            raise ValueError(f"Scenario {s['id']} has negative n_pedestrians")
        if s["pedestrian_speed_mps"] < 0:
            raise ValueError(f"Scenario {s['id']} has negative pedestrian_speed")

    method_names = set(cfg["methods"].keys())
    for ck_method in cfg.get("checkpoints", {}):
        if ck_method not in method_names:
            raise ValueError(f"checkpoints references unknown method '{ck_method}'")


# ==============================================================================
# Expansion — with precomputed geometry
# ==============================================================================
def expand(cfg: dict, smoke: bool = False,
           filter_methods: Optional[set] = None,
           filter_checkpoints: Optional[set] = None) -> list[EpisodeSpec]:
    # Lazy imports — geometry only needed at manifest gen time
    import numpy as np
    from geometry import (
        sample_intra_zone_pair, sample_cross_zone_pair, compute_optimal_length,
    )

    # 70% intra-zone / 30% cross-zone split per scenario (prof directive)
    INTRA_ZONE_FRAC = 0.70

    g = cfg["global"]
    n_eps = g["episodes_per_scenario_smoke"] if smoke else g["episodes_per_scenario"]
    master_seed = g["master_seed"]

    scenarios = cfg["scenarios"]
    methods = cfg["methods"]
    checkpoints_cfg = cfg.get("checkpoints", {})

    episodes: list[EpisodeSpec] = []

    # Precompute geometry per (scenario_id, ep_idx) — shared across methods
    # Cache is keyed by (scenario_id, ep_idx) so all methods get identical
    # spawn/goal for paired analysis.
    geom_cache: dict[tuple, Optional[tuple]] = {}
    n_sample_fail = 0

    def precompute(scenario_id: str, ep_idx: int) -> Optional[tuple]:
        key = (scenario_id, ep_idx)
        if key in geom_cache:
            return geom_cache[key]

        # 70/30 split: first N*0.7 episodes are intra-zone, remainder cross-zone.
        # Deterministic by ep_idx so all methods on same (scenario, ep_idx) get
        # the same zone_type — preserves paired McNemar analysis.
        intra_cutoff = int(INTRA_ZONE_FRAC * n_eps)
        zone_type = "intra" if ep_idx < intra_cutoff else "cross"

        for retry in range(20):
            seed = derive_seed(master_seed, scenario_id, ep_idx, retry)
            rng = np.random.default_rng(seed)

            if zone_type == "intra":
                result = sample_intra_zone_pair(rng)
            else:
                result = sample_cross_zone_pair(rng)

            if result is None:
                continue
            spawn, goal, spawn_zone, goal_zone = result

            # Yaw = bearing to goal + noise (matches train.py convention)
            bearing = float(np.arctan2(goal[1] - spawn[1], goal[0] - spawn[0]))
            yaw_noise = float(rng.uniform(-np.pi / 6, np.pi / 6))
            yaw = bearing + yaw_noise
            opt_len = compute_optimal_length(spawn[:2], goal)

            geom_cache[key] = (
                (spawn[0], spawn[1], yaw), goal, opt_len,
                zone_type, spawn_zone, goal_zone,
            )
            return geom_cache[key]

        geom_cache[key] = None
        return None

    for scenario in scenarios:
        scenario_id = scenario["id"]

        for method_name in methods:
            if filter_methods and method_name not in filter_methods:
                continue

            if method_name in checkpoints_cfg:
                ckpts = list(checkpoints_cfg[method_name].keys())
            else:
                ckpts = [None]

            for ckpt in ckpts:
                if filter_checkpoints and ckpt not in filter_checkpoints:
                    continue

                for ep_idx in range(n_eps):
                    geom = precompute(scenario_id, ep_idx)
                    if geom is None:
                        n_sample_fail += 1
                        continue
                    spawn_xyyaw, goal_xy, opt_len, zone_type, spawn_zone, goal_zone = geom
                    seed = derive_seed(master_seed, scenario_id, ep_idx)

                    parts = [scenario_id, method_name]
                    if ckpt:
                        parts.append(ckpt)
                    parts.append(f"ep{ep_idx:04d}")
                    episode_id = "__".join(parts)

                    episodes.append(EpisodeSpec(
                        episode_id=episode_id,
                        scenario_id=scenario_id,
                        n_pedestrians=int(scenario["n_pedestrians"]),
                        pedestrian_speed_mps=float(scenario["pedestrian_speed_mps"]),
                        method=method_name,
                        checkpoint=ckpt,
                        episode_seed=seed,
                        episode_idx_within_scenario=ep_idx,
                        spawn_xy_yaw=(round(spawn_xyyaw[0], 4),
                                      round(spawn_xyyaw[1], 4),
                                      round(spawn_xyyaw[2], 4)),
                        goal_xy=(round(goal_xy[0], 4), round(goal_xy[1], 4)),
                        optimal_path_length=round(opt_len, 4),
                        zone_type=zone_type,
                        spawn_zone=spawn_zone,
                        goal_zone=goal_zone,
                    ))

    if n_sample_fail > 0:
        print(f"[warn] {n_sample_fail} (scenario, ep_idx, method) skipped due to sampling failures",
              file=sys.stderr)

    return episodes


# ==============================================================================
# Output
# ==============================================================================
def write_manifest(episodes: list[EpisodeSpec], cfg: dict, output: Path) -> None:
    header = {
        "_type": "manifest_header",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "experiment": cfg["experiment"],
        "master_seed": cfg["global"]["master_seed"],
        "episodes_per_scenario": cfg["global"]["episodes_per_scenario"],
        "total_episodes": len(episodes),
        "scenarios": [s["id"] for s in cfg["scenarios"]],
    }
    with output.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")


def summarize(episodes: list[EpisodeSpec]) -> None:
    if not episodes:
        print("Manifest is empty.")
        return

    print(f"Total episodes: {len(episodes):,}")
    print("\nBy method:")
    for m, c in Counter(e.method for e in episodes).most_common():
        print(f"  {m:<30} {c:>7,}")
    print("\nBy scenario:")
    for s, c in Counter(e.scenario_id for e in episodes).most_common():
        print(f"  {s:<30} {c:>7,}")


# ==============================================================================
# CLI
# ==============================================================================
def parse_csv_set(s: Optional[str]) -> Optional[set]:
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("manifest.jsonl"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--filter-method", type=str, help="Comma-separated method names")
    p.add_argument("--filter-checkpoint", type=str, help="Comma-separated checkpoint names")
    p.add_argument("--filter-scenario", type=str, help="Comma-separated scenario IDs")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    # Optionally filter scenarios BEFORE expansion
    filter_scenarios = parse_csv_set(args.filter_scenario)
    if filter_scenarios:
        cfg["scenarios"] = [s for s in cfg["scenarios"] if s["id"] in filter_scenarios]
        if not cfg["scenarios"]:
            sys.exit(f"No scenarios matched filter: {filter_scenarios}")

    episodes = expand(
        cfg,
        smoke=args.smoke,
        filter_methods=parse_csv_set(args.filter_method),
        filter_checkpoints=parse_csv_set(args.filter_checkpoint),
    )
    summarize(episodes)

    if not args.summary_only and episodes:
        write_manifest(episodes, cfg, args.output)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
