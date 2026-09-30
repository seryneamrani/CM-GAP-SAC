"""
CM-GAP-SAC — Evaluation runner.

Uses your existing LimoGazeboEnv + SAC agent. Consumes manifest.jsonl,
per-episode: samples deterministic (spawn, goal) from the manifest seed,
overrides env._spawn_sampler / env._goal_sampler (same trick as your
evaluate_fixed_set), rolls out deterministic policy, logs metrics.

Resumable — restart skips episodes already in results.jsonl.

Usage:
    # From your ROS 2 workspace (after source install/setup.bash):
    python run_eval.py \\
        --eval-config config/eval_config.yaml \\
        --train-config path/to/your/training_config.yaml \\
        --manifest outputs/manifest.jsonl \\
        --output outputs/results.jsonl

    # Dev: limit episodes for smoke test
    python run_eval.py ... --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Your training code — MUST be importable (source your ROS 2 workspace first)
from cm_gap_sac_navigation.envs.limo_gazebo_env import LimoGazeboEnv
from cm_gap_sac_navigation.models.policy import build_policy_from_config
from cm_gap_sac_navigation.rl.sac import build_sac_agent
from cm_gap_sac_navigation.rl.safety_shield import build_shield_from_config
from cm_gap_sac_navigation.utils.config_loader import load_config

from geometry import (
    sample_spawn_in_zone, sample_goal_in_zone,
    bearing_yaw, compute_optimal_length,
)
from metrics import EpisodeTrace, compute_metrics


# ==============================================================================
# Manifest / results IO
# ==============================================================================
def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    episodes = [json.loads(l) for l in lines[1:] if l.strip()]
    return header, episodes


def already_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["episode_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ==============================================================================
# Method loader
# ==============================================================================
class RLMethod:
    """SAC policy + optional CBF shield, ready for deterministic eval."""

    def __init__(self, policy, shield, device, use_shield: bool):
        self.policy = policy
        self.shield = shield if use_shield else None
        self.device = device

    def act(self, obs) -> tuple[np.ndarray, dict, dict]:
        """Return (action, cbf_info_dict, aux_info).

        cbf_info_dict keys: cbf_active (bool), cbf_correction (float)
                            — both None if shield disabled
        aux_info keys: policy_entropy (float, if available)
        """
        with torch.no_grad():
            action_t, _, ent_t = self.policy.act_from_obs(
                obs, device=self.device, deterministic=True
            )
        action = action_t.cpu().numpy().astype(np.float32)
        entropy = float(ent_t.cpu().item()) if ent_t is not None else 0.0

        cbf_info = {"cbf_active": None, "cbf_correction": None}
        return action, cbf_info, {"policy_entropy": entropy}

    def apply_shield(self, action, robot_state, obs) -> tuple[np.ndarray, dict]:
        """Post-process action through CBF shield. Called after act()."""
        if self.shield is None:
            return action, {"cbf_active": None, "cbf_correction": None}

        rx, ry, ryaw = robot_state[0], robot_state[1], robot_state[2]
        result = self.shield.filter(
            u_sac=action, robot_xy_yaw=(rx, ry, ryaw),
            lidar_scan=obs["lidar"],
            pedestrians_rel=obs["pedestrians"],
            ped_mask=obs["ped_mask"],
        )
        correction = float(np.linalg.norm(result.safe_action - action))
        return result.safe_action, {
            "cbf_active": bool(result.n_active_constraints > 0),
            "cbf_correction": correction,
        }


class Nav2ClassicalMethod:
    """Nav2 + DWB classical baseline — TODO stub.

    You need to wire this to your Nav2 action client. Interface required:
        act(obs) → (action, cbf_info={None,None}, aux_info={})
    """
    def __init__(self):
        pass

    def act(self, obs):
        # --- TODO: query Nav2 action server, extract cmd_vel from it
        raise NotImplementedError(
            "Wire your Nav2-DWB action client here. Should return a (v, omega) "
            "action matching your SAC action space."
        )

    def apply_shield(self, action, robot_state, obs):
        return action, {"cbf_active": None, "cbf_correction": None}


def load_method(method_name: str, checkpoint_name: Optional[str],
                eval_config: dict, train_config, device):
    """Instantiate policy + shield for a manifest method entry.

    Returns a Method object with .act(obs) and .apply_shield(action, state, obs).
    """
    method_cfg = eval_config["methods"][method_name]
    method_type = method_cfg["type"]

    if method_type == "rl_policy":
        components = method_cfg["components"]

        # Resolve checkpoint path (same logic as before)
        if "checkpoint_override" in method_cfg:
            ckpt_path = method_cfg["checkpoint_override"]
        elif "inference_from_checkpoint" in method_cfg:
            ref = method_cfg["inference_from_checkpoint"]  # "cm_gap_sac_full.final_1_5M"
            ref_method, ref_ckpt = ref.split(".")
            ckpt_path = eval_config["checkpoints"][ref_method][ref_ckpt]["path"]
        elif checkpoint_name is not None:
            ckpt_path = eval_config["checkpoints"][method_name][checkpoint_name]["path"]
        else:
            raise ValueError(f"No checkpoint resolvable for method {method_name}")

        # Build policy + agent using training config (matches train.py)
        policy = build_policy_from_config(train_config)
        agent = build_sac_agent(train_config, policy=policy, device=device)

        # Load checkpoint — your format: {"agent": ..., "buffer": ..., "step": ...}
        print(f"  loading {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(sd["agent"])
        print(f"  checkpoint step: {sd.get('step', '?')}")

        # Build shield if this method uses one
        use_shield = components.get("cbf_shield", False)
        shield = build_shield_from_config(train_config) if use_shield else None

        return RLMethod(policy=policy, shield=shield, device=device, use_shield=use_shield)

    elif method_type == "classical":
        return Nav2ClassicalMethod()

    else:
        raise ValueError(f"Unknown method type: {method_type}")


# ==============================================================================
# Deterministic spawn/goal sampling from manifest
# ==============================================================================
def resolve_zones(episode_spec: dict, eval_config: dict) -> tuple[str, str]:
    """Extract (spawn_zone, goal_zone) from episode spec.

    Handles both intra-zone (zone == spawn == goal) and cross-zone
    (spawn_zone / goal_zone set in manifest by expand_scenarios.py).
    """
    spawn_zone = episode_spec.get("spawn_zone") or episode_spec["zone"]
    goal_zone = episode_spec.get("goal_zone") or episode_spec["zone"]
    return spawn_zone, goal_zone


def sample_episode_geometry(episode_spec: dict, eval_config: dict
                           ) -> Optional[tuple[tuple[float, float, float], tuple[float, float]]]:
    """From manifest spec → deterministic (spawn_xyyaw, goal_xy).

    Uses episode_seed to seed RNG. Retries with a fresh sub-seed if the
    initial sample fails validation. Returns None if all retries fail.
    """
    spawn_zone, goal_zone = resolve_zones(episode_spec, eval_config)
    base_seed = episode_spec["episode_seed"]

    # A few retries with derived seeds if the first draw fails validation
    for retry_offset in range(10):
        rng = np.random.default_rng(base_seed + retry_offset * 1_000_003)

        spawn = sample_spawn_in_zone(spawn_zone, rng)
        if spawn is None:
            continue

        goal = sample_goal_in_zone(goal_zone, spawn[:2], rng)
        if goal is None:
            continue

        # Reorient yaw to bearing-to-goal (matches train.py's spawn_sampler)
        yaw = bearing_yaw(spawn[:2], goal, rng)
        return (spawn[0], spawn[1], yaw), goal

    return None


# ==============================================================================
# Episode runner
# ==============================================================================
def run_episode(episode_spec: dict, env: LimoGazeboEnv, method,
                eval_config: dict) -> Optional[dict]:
    """Run one episode. Returns result dict, or None if sampling failed."""
    dt = eval_config["global"]["sim_step_s"]
    timeout_s = eval_config["global"]["episode_timeout_s"]
    max_steps = int(timeout_s / dt)

    # 1. Determine spawn + goal deterministically from seed
    geom = sample_episode_geometry(episode_spec, eval_config)
    if geom is None:
        print(f"  SAMPLE_FAIL {episode_spec['episode_id']}", file=sys.stderr)
        return None
    spawn_xyyaw, goal_xy = geom

    # 2. Override env samplers — the trick from evaluate_fixed_set()
    env._spawn_sampler = lambda s=spawn_xyyaw: s
    env._goal_sampler = lambda g=goal_xy: g

    # 3. Reset (env consumes the overridden samplers)
    obs, info = env.reset()

    # 4. Rollout — deterministic policy, optional shield
    positions, velocities, actions, rewards = [], [], [], []
    reward_components: dict[str, list] = {}
    min_dist_static, min_dist_pedestrian = [], []
    cbf_active, cbf_correction, policy_entropy = [], [], []

    final_info: dict = {}
    for step in range(max_steps):
        # Robot state — needed for CBF shield
        rx, ry, ryaw, _, _ = env._gz.get_robot_state()
        robot_state = (rx, ry, ryaw)

        # Policy action (deterministic)
        action, _, aux = method.act(obs)

        # Optional shield
        action, cbf_info = method.apply_shield(action, robot_state, obs)

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info

        # Log per-step
        positions.append([rx, ry])
        # velocity: finite-difference over dt (positions are pre-step, so shift by 1)
        if len(positions) >= 2:
            vx = (positions[-1][0] - positions[-2][0]) / dt
            vy = (positions[-1][1] - positions[-2][1]) / dt
        else:
            vx, vy = 0.0, 0.0
        velocities.append([vx, vy])
        actions.append(action)
        rewards.append(float(reward))

        # Reward decomposition from info (train.py records r_shield; extend if you
        # want the other components — hook them in your env's info dict)
        if "r_shield" in info:
            reward_components.setdefault("shield", []).append(float(info["r_shield"]))
        rest = float(reward) - float(info.get("r_shield", 0.0))
        reward_components.setdefault("policy", []).append(rest)

        # Clearances from env info
        min_dist_static.append(float(info.get("min_lidar", np.inf)))
        min_dist_pedestrian.append(float(info.get("d_min_ped", np.inf)))

        # CBF telemetry
        if cbf_info["cbf_active"] is not None:
            cbf_active.append(cbf_info["cbf_active"])
            cbf_correction.append(cbf_info["cbf_correction"])

        # Policy entropy (from stochastic policy — may be 0 in deterministic mode)
        if "policy_entropy" in aux:
            policy_entropy.append(aux["policy_entropy"])

        if terminated or truncated:
            break

    # 5. Assemble trace + compute metrics
    outcome = final_info.get("outcome", "timeout")
    collision_type = final_info.get("collision_type", "")

    trace = EpisodeTrace(
        dt=dt,
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        reward_components={k: np.asarray(v, dtype=float) for k, v in reward_components.items()},
        min_dist_static=np.asarray(min_dist_static, dtype=float),
        min_dist_pedestrian=np.asarray(min_dist_pedestrian, dtype=float),
        cbf_active=np.asarray(cbf_active, dtype=bool) if cbf_active else None,
        cbf_correction=np.asarray(cbf_correction, dtype=float) if cbf_correction else None,
        policy_entropy_per_step=np.asarray(policy_entropy, dtype=float) if policy_entropy else None,
        reached_goal=(outcome == "success"),
        collided_pedestrian=(outcome == "collision" and collision_type == "pedestrian"),
        collided_static=(outcome == "collision" and collision_type == "static"),
        timed_out=(outcome == "timeout"),
        optimal_path_length=compute_optimal_length(spawn_xyyaw[:2], goal_xy),
    )

    metrics_dict = compute_metrics(trace, eval_config)

    return {
        "episode_id": episode_spec["episode_id"],
        "scenario_group": episode_spec.get("scenario_group", episode_spec["scenario_id"]),
        "world": episode_spec.get("world", "hospital"),
        "zone": episode_spec.get(
            "zone", episode_spec.get("spawn_zone", episode_spec.get("zone_type"))
        ),
        "pedestrian_config": episode_spec.get("pedestrian_config", episode_spec["scenario_id"]),
        "method": episode_spec["method"],
        "checkpoint": episode_spec["checkpoint"],
        "episode_seed": episode_spec["episode_seed"],
        "spawn_xy": [round(spawn_xyyaw[0], 3), round(spawn_xyyaw[1], 3)],
        "goal_xy": [round(goal_xy[0], 3), round(goal_xy[1], 3)],
        "shield_frozen": (outcome == "shield_frozen"),
        "scenario_id": episode_spec["scenario_id"],
        "n_pedestrians": episode_spec["n_pedestrians"],
        "pedestrian_speed_mps": episode_spec["pedestrian_speed_mps"],
        "zone_type": episode_spec.get("zone_type"),
        "spawn_zone": episode_spec.get("spawn_zone"),
        "goal_zone": episode_spec.get("goal_zone"),
        **metrics_dict,
    }

# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", type=Path, required=True,
                    help="Path to eval_config.yaml (this framework)")
    ap.add_argument("--train-config", type=Path, required=True,
                    help="Path to your training config YAML (for LimoGazeboEnv)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    eval_config = yaml.safe_load(args.eval_config.read_text())
    train_config = load_config(args.train_config)

    header, episodes = load_manifest(args.manifest)
    done = already_done(args.output)
    print(f"[eval] Manifest: {header['total_episodes']} eps. Already done: {len(done)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build env ONCE with a placeholder sampler (overridden per episode)
    print("[eval] Building LimoGazeboEnv...")
    env = LimoGazeboEnv(
        config_path=str(args.train_config), seed=42,
        spawn_xy_yaw=(0.0, 0.0, 0.0),
        goal_sampler=lambda: (0.0, 0.0),
    )

    method_cache: dict[tuple, object] = {}
    n_run = n_ok = n_fail = 0

    with args.output.open("a") as f:
        for spec in episodes:
            if spec["episode_id"] in done:
                continue
            if args.limit and n_run >= args.limit:
                break

            key = (spec["method"], spec["checkpoint"])
            if key not in method_cache:
                print(f"[eval] Loading method: {key}")
                method_cache[key] = load_method(
                    spec["method"], spec["checkpoint"],
                    eval_config, train_config, device
                )

            t0 = time.time()
            try:
                result = run_episode(spec, env, method_cache[key], eval_config)
                if result is None:
                    n_fail += 1
                    continue
                result["wall_time_s"] = round(time.time() - t0, 2)
                f.write(json.dumps(result) + "\n")
                f.flush()
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"  [{n_ok}] {spec['episode_id']} → "
                          f"success={result['outcome']['success']} "
                          f"({result['wall_time_s']}s)")
            except Exception as e:
                n_fail += 1
                print(f"[FAIL] {spec['episode_id']}: {e}", file=sys.stderr)
                traceback.print_exc()
                with Path("eval_failures.log").open("a") as ff:
                    ff.write(f"{spec['episode_id']}\t{type(e).__name__}\t{e}\n")

            n_run += 1

    print(f"\n[eval] Done. Ran {n_run} new episodes ({n_ok} ok, {n_fail} failed).")
    env.close()


if __name__ == "__main__":
    main()
