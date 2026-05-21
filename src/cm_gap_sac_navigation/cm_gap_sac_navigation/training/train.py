"""End-to-end training loop for CM-GAP_SAC.

Standard SAC training with PER replay, periodic deterministic evaluation,
and TensorBoard logging.

Usage:
    ros2 run cm_gap_sac_navigation train --config config/cm_gap_sac.yaml

    # or directly:
    python -m cm_gap_sac_navigation.training.train \\
        --config config/cm_gap_sac.yaml \\
        --total-steps 1500000

What it does, per training step:
    1. Sample action from policy (warmup phase: random actions to fill
       the buffer before training starts).
    2. Step the environment.
    3. Push transition to PER buffer with current pedestrian attention
       entropy (for the entropy-weighted priority).
    4. After warmup, perform one SAC gradient update per env step.
    5. Update PER priorities with the resulting TD errors.
    6. Periodically: log to TensorBoard, run eval, save checkpoint.

CLI handles:
    --resume          path to a checkpoint to continue from
    --no-eval         skip periodic eval (faster training, less observability)
    --no-tensorboard  disable TB logging
    --total-steps     override training.total_steps in YAML
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cm_gap_sac_navigation.utils.config_loader import load_config


# ======================================================================
# Utilities
# ======================================================================

# Hospital navigable bounds — entire navigable area (corridor + 4 chambres)
HOSPITAL_X_MIN, HOSPITAL_X_MAX = -6.5, 6.5
HOSPITAL_Y_MIN, HOSPITAL_Y_MAX = -2.0, 2.0
MIN_SPAWN_GOAL_DIST = 2.0

# Static obstacles to avoid (x, y, safety_radius in meters)
# Centered on actual SDF coordinates with conservative margin
STATIC_OBSTACLES = [
    # Beds + IV stands + bedside tables (grouped)
    (5.4, 5.0, 1.2),     # bed_1 + iv_stand_1 + bedside_table_1
    (5.4, -5.0, 1.2),    # bed_patient_1 + iv_stand_2 + bedside_table_2
    (-5.0, 5.0, 0.9),    # bed_patient_2
    (-5.0, -5.0, 0.9),   # wheelchair_1
    # Cabinets
    (7.0, 3.0, 0.7),     # cabinet_1
    (7.0, -3.0, 0.7),    # cabinet_2
    (7.0, 6.5, 0.7),     # cabinet_extra_1
    # Wheelchairs + IV stands extras
    (-6.0, 6.0, 0.8),    # wheelchair_extra_2
    (0.19, 2.1, 0.7),    # wheelchair_extra_1 (in corridor!)
    (-5.5, 6.8, 0.6),    # iv_stand_extra_1
    (-6.5, -6.5, 0.6),   # iv_stand_extra_2
    # Static people
    (3.0, 0.0, 0.7),     # person_standing_1
    (-3.0, 1.0, 0.7),    # nurse_1
    (-2.0, -1.0, 0.7),   # visitor_1
    (-2.0, 2.6, 0.7),    # static_nurse_2
    (6.0, 0.5, 0.7),     # static_visitor_2
    # Dynamic pedestrians (initial positions — buffer for spawn)
    (5.43, 1.62, 1.5),   # ped_1
    (-3.92, 1.18, 1.5),  # ped_2
    (-6.75, -1.77, 1.5), # ped_4
    (3.21, -6.50, 1.5),  # ped_5
    (-1.65, -4.20, 1.5), # ped_6
    (3.19, 6.50, 1.5),   # ped_7
    (-1.76, 4.20, 1.5),  # ped_8
]


def _is_navigable(x: float, y: float) -> bool:
    """Check if (x, y) is in free space, avoiding walls and obstacles."""
    # Outside outer walls (with margin)
    if abs(x) > 7.3 or abs(y) > 7.3:
        return False
    # Too close to corridor walls (y = ±3, extending x ∈ [-7, +7])
    if abs(abs(y) - 3.0) < 0.6 and abs(x) < 7.0:
        return False
    # Too close to internal vertical dividers (x = 2)
    # div_left_top_1 (2, 4): y ∈ [3, 5], div_left_top_2 (2, 7): y ∈ [6, 8]
    # div_left_bot_1 (2, -4): y ∈ [-5, -3], div_left_bot_2 (2, -7): y ∈ [-8, -6]
    if abs(x - 2.0) < 0.6:
        if 3.0 < y < 5.0 or 6.0 < y < 8.0:
            return False
        if -5.0 < y < -3.0 or -8.0 < y < -6.0:
            return False
    # Too close to any static obstacle
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < oradius ** 2:
            return False
    return True


def _sample_navigable_point(rng, max_tries: int = 100):
    """Rejection sampling: return a free point in the hospital."""
    for _ in range(max_tries):
        x = float(rng.uniform(HOSPITAL_X_MIN, HOSPITAL_X_MAX))
        y = float(rng.uniform(HOSPITAL_Y_MIN, HOSPITAL_Y_MAX))
        if _is_navigable(x, y):
            return (x, y)
    # Fallback: center of corridor (always safe)
    return (0.0, 0.0)
class _GracefulShutdown:
    """Catches Ctrl-C once to save a checkpoint, twice to force exit."""
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
        print(
            "\n[train] Shutdown requested. Will save checkpoint at next "
            "iteration. Ctrl-C again to force exit.", flush=True,
        )


# ======================================================================
def make_optimizer_log(info, alpha: float) -> dict:
    """Convert SacUpdateInfo to a TB-friendly dict."""
    return {
        "train/critic_loss":   info.critic_loss,
        "train/actor_loss":    info.actor_loss,
        "train/alpha_loss":    info.alpha_loss,
        "train/alpha":         alpha,
        "train/log_prob_mean": info.log_prob_mean,
        "train/q_min_mean":    info.q_min_mean,
        "train/td_error_mean": info.td_error_mean,
        "train/td_error_max":  info.td_error_max,
    }


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train CM-GAP_SAC")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override training.total_steps")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Override training.eval_every")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=None,
                        help="Override training.save_every")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable periodic evaluation")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- Imports here to fail fast on missing deps -----------------------
    from cm_gap_sac_navigation.envs.limo_gazebo_env import LimoGazeboEnv
    from cm_gap_sac_navigation.models.policy import build_policy_from_config
    from cm_gap_sac_navigation.rl.per_buffer import build_per_from_config
    from cm_gap_sac_navigation.rl.sac import build_sac_agent
    from cm_gap_sac_navigation.rl.safety_shield import build_shield_from_config
    from cm_gap_sac_navigation.training.eval import evaluate

    # ---- Config and device -----------------------------------------------
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

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

    # ---- TensorBoard -----------------------------------------------------
    tb_writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_dir = Path(log_dir) / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(run_dir))
            print(f"[train] TensorBoard: {run_dir}")
        except ImportError:
            print("[train] tensorboard not installed; logging disabled.")

    # ---- Build env, policy, agent, buffer --------------------------------
    print(f"[train] Building env...")
    rng = np.random.default_rng(seed)

    # Curriculum: max goal distance grows with training step.
    # Phase 1 (0-30k):    max 2.5m  — short, easy navigation
    # Phase 2 (30k-100k): max 5.0m  — medium trajectories
    # Phase 3 (100k+):    no limit  — full corridor span
    _step_tracker = [0]
    _last_spawn = [(0.0, 0.0)]

    def _curriculum_max_dist():
        s = _step_tracker[0]
        if s < 30000:
            return 2.5
        if s < 100000:
            return 5.0
        return 100.0  # effectively no limit

    def spawn_sampler():
        x, y = _sample_navigable_point(rng)
        _last_spawn[0] = (x, y)
        # Orient toward corridor center (y=0) to give the robot
        # open space ahead and avoid spawn-against-wall situations.
        # If above y=0, face downward (-pi/2). If below, face upward (+pi/2).
        # Add small noise for diversity.
        if y > 0:
            base_yaw = -np.pi / 2   # facing -y
        else:
            base_yaw = np.pi / 2    # facing +y
        yaw = base_yaw + float(rng.uniform(-0.5, 0.5))   # ±28° noise
        return (x, y, yaw)

    def goal_sampler():
        sx, sy = _last_spawn[0]
        max_d = _curriculum_max_dist()
        # Sample goals close to spawn with rejection
        for _ in range(50):
            gx, gy = _sample_navigable_point(rng)
            dist = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
            if 0.5 < dist <= max_d:
                return (gx, gy)
        # Fallback: any navigable point
        return _sample_navigable_point(rng)

    def update_curriculum_step(step):
        _step_tracker[0] = step

    env = LimoGazeboEnv(
        config_path=args.config,
        seed=seed,
        spawn_xy_yaw=spawn_sampler(),
        goal_sampler=goal_sampler,
    )
    env._spawn_sampler = spawn_sampler

    print(f"[train] Building policy + critic...")
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)

    print(f"[train] Building PER buffer (capacity={cfg.per['capacity']})...")
    buffer = build_per_from_config(cfg, action_dim=cfg.action.get("dim", 2),
                                   seed=seed)

    # ---- Build CBF safety shield (optional) ------------------------------
    shield = build_shield_from_config(cfg)
    if shield is not None:
        print(f"[train] CBF safety shield: ENABLED "
              f"(r_safe={shield.r_safe}, gamma={shield.gamma})")
    else:
        print(f"[train] CBF safety shield: DISABLED (cbf.enabled=false)")

    # ---- Resume from checkpoint -----------------------------------------
    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(sd["agent"])
        start_step = int(sd.get("step", 0))
        print(f"[train] Resumed from {args.resume} at step {start_step}")

    # ---- Training loop ---------------------------------------------------
    shutdown = _GracefulShutdown()
    obs, info = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    episodes_done: list[dict] = []   # for rolling stats

    # CBF freeze-detection state
    consec_infeasible = 0
    max_consec_infeasible = cfg.cbf.get("max_consecutive_infeasible", 20)

    t_start = time.time()
    print(f"[train] Starting training: {total_steps} steps, "
          f"warmup={warmup}, batch={batch_size}")

    for step in range(start_step, total_steps):
        update_curriculum_step(step)
        # ---- Sample action ---------------------------------------------
        if step < warmup:
            action = env.action_space.sample().astype(np.float32)
            attn_entropy = 0.0
        else:
            action_t, _, ent_t = policy.act_from_obs(
                obs, device=device, deterministic=False,
            )
            action = action_t.cpu().numpy().astype(np.float32)
            attn_entropy = float(ent_t.cpu().item())

        # ---- CBF safety shield -----------------------------------------
        # Apply shield AFTER warmup (during warmup, random actions explore).
        # Shield uses lidar + pedestrians from current obs, plus robot pose
        # from the env's gazebo interface for the world-frame projection.
        cbf_active = False
        cbf_modified = False
        cbf_infeasible = False
        if shield is not None and step >= warmup:
            x, y, yaw, _, _ = env._gz.get_robot_state()
            shield_result = shield.filter(
                u_sac=action,
                robot_xy_yaw=(x, y, yaw),
                lidar_scan=obs["lidar"],
                pedestrians_rel=obs["pedestrians"],
                ped_mask=obs["ped_mask"],
            )
            action = shield_result.safe_action
            cbf_active = shield_result.n_active_constraints > 0
            cbf_modified = shield_result.was_modified
            cbf_infeasible = shield_result.infeasible

            # Track consecutive infeasibility (freezing-robot prevention)
            if cbf_infeasible:
                consec_infeasible += 1
            else:
                consec_infeasible = 0

        if shield is not None and step >= warmup:
            print(f"[shield] u_sac=({shield_result.delta_action[0]+action[0]:+.2f}, {shield_result.delta_action[1]+action[1]:+.2f}) "
                f"→ u_safe=({action[0]:+.2f}, {action[1]:+.2f})  modified={cbf_modified}", flush=True)
        # ---- Step env --------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_flag = float(terminated)   # truncation should NOT bootstrap target

        # Add shield diagnostics to info for TensorBoard
        info["cbf_active"] = cbf_active
        info["cbf_modified"] = cbf_modified
        info["cbf_infeasible"] = cbf_infeasible

        # If the shield has been infeasible too long, force-terminate the
        # episode as a failure. This prevents the robot from sitting frozen
        # for the rest of max_steps with no learning signal.
        if consec_infeasible >= max_consec_infeasible:
            truncated = True
            info["outcome"] = info.get("outcome", "shield_frozen")
            consec_infeasible = 0

        # Augment reward with shield-active penalty (encourages safer policies)
        if cbf_modified and step >= warmup:
            shield_pen = -cfg.reward.alpha_shield
            reward = float(reward) + shield_pen
            info["r_shield"] = shield_pen
        else:
            info["r_shield"] = 0.0

        # ---- Push to buffer -------------------------------------------
        buffer.add(
            obs=obs, action=action, reward=float(reward),
            next_obs=next_obs, done=done_flag,
            attention_entropy=attn_entropy,
        )

        episode_reward += float(reward)
        episode_steps += 1

        # ---- Episode end -----------------------------------------------
        if terminated or truncated:
            episode_count += 1
            outcome = info.get("outcome", "timeout")
            episodes_done.append({
                "outcome": outcome, "reward": episode_reward,
                "steps": episode_steps,
            })
            if tb_writer:
                tb_writer.add_scalar("episode/reward", episode_reward, step)
                tb_writer.add_scalar("episode/length", episode_steps, step)
                tb_writer.add_scalar(
                    "episode/success", 1.0 if outcome == "success" else 0.0, step,
                )
                tb_writer.add_scalar(
                    "episode/collision", 1.0 if outcome == "collision" else 0.0, step,
                )
                if shield is not None:
                    tb_writer.add_scalar(
                        "episode/cbf_modified",
                        1.0 if cbf_modified else 0.0, step,
                    )
            # Rolling stats every 20 episodes.
            if episode_count % 20 == 0 and len(episodes_done) >= 20:
                last20 = episodes_done[-20:]
                succ = sum(1 for e in last20 if e["outcome"] == "success") / 20
                coll = sum(1 for e in last20 if e["outcome"] == "collision") / 20
                mean_r = sum(e["reward"] for e in last20) / 20
                elapsed = time.time() - t_start
                sps = (step - start_step + 1) / max(1.0, elapsed)
                print(
                    f"[train] step={step:7d}  ep={episode_count:5d}  "
                    f"succ20={succ*100:5.1f}%  coll20={coll*100:5.1f}%  "
                    f"meanR20={mean_r:+7.2f}  α={agent.alpha.item():.4f}  "
                    f"sps={sps:.1f}",
                    flush=True,
                )
                if tb_writer:
                    tb_writer.add_scalar("rolling/success_rate_20", succ, step)
                    tb_writer.add_scalar("rolling/collision_rate_20", coll, step)
                    tb_writer.add_scalar("rolling/mean_reward_20", mean_r, step)
                    tb_writer.add_scalar("perf/steps_per_second", sps, step)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
        else:
            obs = next_obs

        # ---- Gradient update -------------------------------------------
        if step >= warmup and buffer.can_sample(batch_size):
            batch = buffer.sample(batch_size)
            update_info, td_errors, entropies = agent.update(batch)
            buffer.update_priorities(
                indices=batch["indices"],
                td_errors=td_errors,
                entropies=entropies,
            )
            if tb_writer and step % 100 == 0:
                logs = make_optimizer_log(update_info, agent.alpha.item())
                logs["per/beta"] = batch["beta"]
                for k, v in logs.items():
                    tb_writer.add_scalar(k, v, step)

        # ---- Periodic evaluation --------------------------------------
        if (not args.no_eval) and step > 0 and step % eval_every == 0 and step >= warmup:
            print(f"\n[train] Evaluation at step {step}...")
            metrics = evaluate(
                env=env, policy=policy,
                n_episodes=args.eval_episodes,
                device=device, verbose=False,
                shield=shield,
            )
            print(metrics.pretty())
            if tb_writer:
                for k, v in metrics.to_dict().items():
                    tb_writer.add_scalar(k, v, step)
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # ---- Periodic checkpoint --------------------------------------
        if step > 0 and step % save_every == 0:
            ckpt_path = Path(log_dir) / f"checkpoint_step{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"agent": agent.state_dict(), "step": step}, ckpt_path)
            print(f"[train] Saved checkpoint: {ckpt_path}", flush=True)

        # ---- Graceful shutdown -----------------------------------------
        if shutdown.requested:
            print(f"[train] Saving emergency checkpoint at step {step}...")
            ckpt_path = Path(log_dir) / f"checkpoint_interrupted_step{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"agent": agent.state_dict(), "step": step}, ckpt_path)
            print(f"[train] Saved: {ckpt_path}", flush=True)
            break

    # ---- Final save ------------------------------------------------------
    final_ckpt = Path(log_dir) / "checkpoint_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "step": step}, final_ckpt)
    print(f"[train] Final checkpoint: {final_ckpt}")

    if tb_writer:
        tb_writer.close()
    env.close()
    print(f"[train] Done. Total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()