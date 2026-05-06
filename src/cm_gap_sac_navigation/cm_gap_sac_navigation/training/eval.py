"""Deterministic evaluation of a trained CM-GAP_SAC policy.

Runs N episodes with `deterministic=True` (no exploration noise) and
collects the seven thesis evaluation metrics:
    - Success rate                        (% over N episodes)
    - Mean path length on success         (m)
    - Min pedestrian distance per episode (m)
    - Intrusion rate                      (% steps with d_min_ped < 0.45)
    - Mean time to goal                   (s)
    - Mean jerk                           (m/s^3 proxy via |Δv|+|Δω|)
    - CBF activation rate                 (% steps where shield acts;
                                            zero if shield disabled)

Designed to be called periodically from the training loop (eval_every steps)
and at the end for full reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class EvalMetrics:
    """Aggregated metrics from one evaluation run."""
    n_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_path_length: float        # m, on successes only
    mean_time_to_goal: float       # s, on successes only
    mean_min_ped_distance: float   # m, over all episodes (only when peds present)
    intrusion_rate: float          # % steps with d_min_ped < 0.45 m
    mean_jerk: float               # m/s + rad/s per step (proxy)
    mean_episode_length: float     # steps
    mean_episode_reward: float
    cbf_activation_rate: float     # 0.0 if shield disabled

    def to_dict(self) -> Dict[str, float]:
        return {
            "eval/success_rate":         self.success_rate,
            "eval/collision_rate":       self.collision_rate,
            "eval/timeout_rate":         self.timeout_rate,
            "eval/mean_path_length":     self.mean_path_length,
            "eval/mean_time_to_goal":    self.mean_time_to_goal,
            "eval/mean_min_ped_dist":    self.mean_min_ped_distance,
            "eval/intrusion_rate":       self.intrusion_rate,
            "eval/mean_jerk":            self.mean_jerk,
            "eval/mean_episode_length":  self.mean_episode_length,
            "eval/mean_episode_reward":  self.mean_episode_reward,
            "eval/cbf_activation_rate":  self.cbf_activation_rate,
        }

    def pretty(self) -> str:
        lines = [
            "─" * 60,
            f"Evaluation over {self.n_episodes} episodes",
            "─" * 60,
            f"  Success rate            : {self.success_rate*100:6.2f}%",
            f"  Collision rate          : {self.collision_rate*100:6.2f}%",
            f"  Timeout rate            : {self.timeout_rate*100:6.2f}%",
            f"  Mean path length        : {self.mean_path_length:6.2f} m",
            f"  Mean time to goal       : {self.mean_time_to_goal:6.2f} s",
            f"  Mean min pedestrian d   : {self.mean_min_ped_distance:6.2f} m",
            f"  Intrusion rate (<0.45m) : {self.intrusion_rate*100:6.2f}%",
            f"  Mean jerk (proxy)       : {self.mean_jerk:6.4f}",
            f"  Mean episode length     : {self.mean_episode_length:6.1f} steps",
            f"  Mean episode reward     : {self.mean_episode_reward:6.2f}",
            f"  CBF activation rate     : {self.cbf_activation_rate*100:6.2f}%",
            "─" * 60,
        ]
        return "\n".join(lines)


# ======================================================================
def evaluate(
    env,                        # LimoGazeboEnv (or compatible)
    policy,                     # CmGapSacPolicy
    n_episodes: int = 20,
    max_steps_per_episode: Optional[int] = None,
    device: Optional[torch.device] = None,
    intimate_zone: float = 0.45,
    verbose: bool = True,
) -> EvalMetrics:
    """Evaluate `policy` deterministically on `env` for `n_episodes` episodes."""
    if device is None:
        device = next(policy.parameters()).device
    policy.eval()

    successes = 0
    collisions = 0
    timeouts = 0
    path_lengths: List[float] = []
    times_to_goal: List[float] = []
    min_ped_dists: List[float] = []
    intrusion_steps = 0
    total_intrusion_eligible = 0
    jerks: List[float] = []
    episode_lengths: List[int] = []
    episode_rewards: List[float] = []
    cbf_acts = 0
    total_steps = 0

    dt = 1.0 / env.cfg.episode.control_hz

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_steps = 0
        ep_reward = 0.0
        prev_xy: Optional[np.ndarray] = None
        path_len = 0.0
        ep_min_ped = float("inf")
        prev_action = np.zeros(2, dtype=np.float32)

        while not (done or truncated):
            with torch.no_grad():
                action_t, _, _ = policy.act_from_obs(
                    obs, device=device, deterministic=True,
                )
            action = action_t.cpu().numpy().astype(np.float32)

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            # Path length (use d_goal differences as proxy if no x,y from env;
            # since the env has odom, we can use obs["goal"][0] difference.)
            d_now = float(obs["goal"][0])
            if prev_xy is None:
                prev_xy = np.array([d_now])
            else:
                # Approximate with abs delta along d_goal — gives a lower
                # bound on actual path length but is monotonic with it.
                path_len += abs(d_now - float(prev_xy[0]))
                prev_xy[0] = d_now

            # Pedestrian distance tracking.
            d_min = float(info.get("d_min_ped", float("inf")))
            if np.isfinite(d_min):
                ep_min_ped = min(ep_min_ped, d_min)
                total_intrusion_eligible += 1
                if d_min < intimate_zone:
                    intrusion_steps += 1

            # Jerk proxy: |Δv| + |Δω|.
            jerks.append(abs(action[0] - prev_action[0]) + abs(action[1] - prev_action[1]))
            prev_action = action

            # CBF activation tracking (env may set this in info if shield wired).
            if info.get("cbf_active", False):
                cbf_acts += 1

            if max_steps_per_episode is not None and ep_steps >= max_steps_per_episode:
                truncated = True

        # Outcome bookkeeping.
        outcome = info.get("outcome", "timeout" if truncated else "unknown")
        if outcome == "success":
            successes += 1
            path_lengths.append(path_len)
            times_to_goal.append(ep_steps * dt)
        elif outcome == "collision":
            collisions += 1
        else:
            timeouts += 1

        if np.isfinite(ep_min_ped):
            min_ped_dists.append(ep_min_ped)
        episode_lengths.append(ep_steps)
        episode_rewards.append(ep_reward)

        if verbose:
            print(
                f"  ep {ep+1:3d}/{n_episodes}  "
                f"outcome={outcome:9s}  "
                f"steps={ep_steps:4d}  "
                f"reward={ep_reward:+8.2f}  "
                f"min_d_ped={ep_min_ped if np.isfinite(ep_min_ped) else float('nan'):.2f}"
            )

    policy.train()

    n = max(1, n_episodes)
    metrics = EvalMetrics(
        n_episodes=n_episodes,
        success_rate=successes / n,
        collision_rate=collisions / n,
        timeout_rate=timeouts / n,
        mean_path_length=float(np.mean(path_lengths)) if path_lengths else 0.0,
        mean_time_to_goal=float(np.mean(times_to_goal)) if times_to_goal else 0.0,
        mean_min_ped_distance=float(np.mean(min_ped_dists)) if min_ped_dists else float("nan"),
        intrusion_rate=intrusion_steps / max(1, total_intrusion_eligible),
        mean_jerk=float(np.mean(jerks)) if jerks else 0.0,
        mean_episode_length=float(np.mean(episode_lengths)),
        mean_episode_reward=float(np.mean(episode_rewards)),
        cbf_activation_rate=cbf_acts / max(1, total_steps),
    )
    return metrics


# ======================================================================
def main() -> None:
    """CLI entry point: load checkpoint and evaluate."""
    import argparse
    from pathlib import Path
    from cm_gap_sac_navigation.utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="Evaluate a CM-GAP_SAC checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from cm_gap_sac_navigation.envs.limo_gazebo_env import LimoGazeboEnv
    from cm_gap_sac_navigation.models.policy import build_policy_from_config

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = LimoGazeboEnv(config_path=args.config, seed=0)
    policy = build_policy_from_config(cfg).to(device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(sd["policy"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    metrics = evaluate(env, policy, n_episodes=args.n_episodes, device=device)
    print(metrics.pretty())
    env.close()


if __name__ == "__main__":
    main()
