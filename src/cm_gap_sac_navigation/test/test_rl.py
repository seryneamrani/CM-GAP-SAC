"""Tests for the RL layer (PER buffer + SAC update)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cm_gap_sac_navigation.models.policy import build_policy_from_config
from cm_gap_sac_navigation.rl.per_buffer import (
    PrioritizedReplayBuffer,
    SumTree,
    build_per_from_config,
)
from cm_gap_sac_navigation.rl.sac import SacAgent, build_sac_agent
from cm_gap_sac_navigation.utils.config_loader import load_config


# ======================================================================
# Helpers
# ======================================================================
def _cfg():
    return load_config(Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml")


def _make_obs(cfg):
    """Random Dict observation matching cfg shapes."""
    return {
        "lidar":       np.random.randn(cfg.observation.lidar.n_beams).astype(np.float32),
        "pedestrians": np.random.randn(cfg.observation.pedestrian.k_max,
                                       cfg.observation.pedestrian.n_features).astype(np.float32),
        "ped_mask":    np.ones(cfg.observation.pedestrian.k_max, dtype=np.uint8),
        "imu":         np.random.randn(cfg.observation.imu.n_features).astype(np.float32),
        "goal":        np.random.randn(cfg.observation.goal.n_features).astype(np.float32),
    }


# ======================================================================
# SumTree
# ======================================================================
def test_sumtree_total_after_inserts():
    tree = SumTree(capacity=8)
    for i in range(8):
        tree.update(i, float(i + 1))
    assert tree.total == pytest.approx(36.0)   # 1+2+...+8


def test_sumtree_get_proportional():
    tree = SumTree(capacity=4)
    tree.update(0, 1.0)
    tree.update(1, 1.0)
    tree.update(2, 1.0)
    tree.update(3, 1.0)
    # value=2.5 should land in leaf 2 (cumulative 1+1=2 <= 2.5 < 3=1+1+1).
    idx, prio = tree.get(2.5)
    assert idx == 2
    assert prio == pytest.approx(1.0)


def test_sumtree_get_with_unequal_priorities():
    tree = SumTree(capacity=4)
    tree.update(0, 0.1)
    tree.update(1, 0.9)
    tree.update(2, 0.0)
    tree.update(3, 0.0)
    # value below 0.1 -> leaf 0; above -> leaf 1.
    idx_low, _ = tree.get(0.05)
    idx_high, _ = tree.get(0.5)
    assert idx_low == 0
    assert idx_high == 1


def test_sumtree_max_priority_tracking():
    tree = SumTree(capacity=4)
    assert tree.max_priority == 1.0
    tree.update(0, 5.0)
    assert tree.max_priority == 5.0


# ======================================================================
# PrioritizedReplayBuffer
# ======================================================================
def test_per_add_and_sample():
    cfg = _cfg()
    buf = build_per_from_config(cfg, action_dim=2, seed=0)
    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(50):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.5, nxt, 0.0, attention_entropy=0.5)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch["obs_lidar"].shape == (16, cfg.observation.lidar.n_beams)
    assert batch["obs_imu"].shape  == (16, cfg.observation.imu.n_features)
    assert batch["action"].shape   == (16, 2)
    assert batch["is_weights"].shape == (16,)
    assert batch["indices"].shape  == (16,)


def test_per_circular_overwrite():
    cfg = _cfg()
    buf = PrioritizedReplayBuffer(
        capacity=10, n_beams=36, k_max=5, ped_features=5,
        imu_features=6, goal_features=4, action_dim=2, seed=0,
    )
    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(15):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0)
    assert len(buf) == 10   # capped at capacity


def test_per_priority_update_changes_sampling():
    """After increasing priority of one slot, it should be sampled more."""
    cfg = _cfg()
    buf = PrioritizedReplayBuffer(
        capacity=20, n_beams=36, k_max=5, ped_features=5,
        imu_features=6, goal_features=4, action_dim=2, seed=42,
    )
    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(20):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0)
    # Boost slot 5's priority massively.
    buf.update_priorities(np.array([5]), np.array([100.0]))
    # Sample many batches, count appearances of index 5.
    counts = np.zeros(20, dtype=int)
    for _ in range(50):
        b = buf.sample(8)
        for i in b["indices"]:
            counts[int(i)] += 1
    # Index 5 should be sampled much more often than average (25 = 50*8/16).
    assert counts[5] > counts.mean() * 2


def test_per_entropy_weighting_amplifies_low_entropy():
    """Two transitions with same TD-error: the one with lower entropy
    should get higher priority."""
    buf = PrioritizedReplayBuffer(
        capacity=4, n_beams=36, k_max=5, ped_features=5,
        imu_features=6, goal_features=4, action_dim=2,
        eta=0.6, nu=0.5, lam=3.0, use_attention_entropy=True, seed=0,
    )
    # Use same TD error for both, but different entropies via update.
    same_td = np.array([1.0, 1.0])
    high_low_entropy = np.array([1.0, 0.0])   # uniform vs peaked
    # We can only test via update_priorities since add() takes entropy at write.
    # Simulate: add 2 transitions with entropies stored, then update.
    obs = _make_obs(load_config(Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml"))
    nxt = obs
    buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0, attention_entropy=1.0)
    buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0, attention_entropy=0.0)
    buf.update_priorities(np.array([0, 1]), same_td, entropies=high_low_entropy)
    # Read priorities back via tree internals.
    p0 = buf.tree._tree[buf.tree.capacity - 1 + 0]
    p1 = buf.tree._tree[buf.tree.capacity - 1 + 1]
    assert p1 > p0   # peaked entropy gets higher priority


def test_per_beta_annealing():
    buf = PrioritizedReplayBuffer(
        capacity=10, n_beams=36, k_max=5, ped_features=5,
        imu_features=6, goal_features=4, action_dim=2,
        beta_start=0.4, beta_end=1.0, beta_anneal_steps=100, seed=0,
    )
    assert buf._current_beta() == pytest.approx(0.4)
    buf._step = 50
    assert buf._current_beta() == pytest.approx(0.7, abs=1e-3)
    buf._step = 200
    assert buf._current_beta() == pytest.approx(1.0)


def test_per_is_weights_normalized():
    cfg = _cfg()
    buf = build_per_from_config(cfg, action_dim=2, seed=0)
    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(100):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0)
    batch = buf.sample(32)
    # IS weights normalized so max == 1.
    assert batch["is_weights"].max() == pytest.approx(1.0, abs=1e-5)
    assert batch["is_weights"].min() > 0.0


def test_per_can_sample():
    cfg = _cfg()
    buf = build_per_from_config(cfg, action_dim=2, seed=0)
    assert not buf.can_sample(8)
    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(8):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0)
    assert buf.can_sample(8)


# ======================================================================
# SAC update
# ======================================================================
def test_sac_one_update_step_runs():
    """A single SAC update on a fake batch should not crash and should
    produce finite losses."""
    cfg = _cfg()
    device = torch.device("cpu")  # CI safety
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)
    buf = build_per_from_config(cfg, action_dim=2, seed=0)

    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(64):
        buf.add(obs, np.zeros(2, dtype=np.float32), 0.0, nxt, 0.0, attention_entropy=0.5)

    batch = buf.sample(32)
    info, td_errors, entropies = agent.update(batch)

    assert np.isfinite(info.critic_loss)
    assert np.isfinite(info.actor_loss)
    assert np.isfinite(info.alpha_loss)
    assert info.alpha > 0.0
    assert td_errors.shape == (32,)
    assert entropies.shape == (32,)
    assert (td_errors >= 0.0).all()


def test_sac_critic_params_change_after_update():
    cfg = _cfg()
    device = torch.device("cpu")
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)
    buf = build_per_from_config(cfg, action_dim=2, seed=0)

    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(64):
        buf.add(obs, np.zeros(2, dtype=np.float32), 1.0, nxt, 0.0)

    # Snapshot Q1 first layer weights.
    w_before = agent.critic.q1.net[0].weight.detach().clone()

    batch = buf.sample(32)
    agent.update(batch)

    w_after = agent.critic.q1.net[0].weight.detach().clone()
    assert (w_after - w_before).abs().sum().item() > 0.0


def test_sac_target_critic_lags_behind_critic():
    """Target should change much less than critic after one update, since τ << 1."""
    cfg = _cfg()
    device = torch.device("cpu")
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)
    buf = build_per_from_config(cfg, action_dim=2, seed=0)

    obs = _make_obs(cfg)
    nxt = _make_obs(cfg)
    for _ in range(64):
        buf.add(obs, np.zeros(2, dtype=np.float32), 1.0, nxt, 0.0)

    w_critic_before = agent.critic.q1.net[0].weight.detach().clone()
    w_target_before = agent.target_critic.q1.net[0].weight.detach().clone()

    batch = buf.sample(32)
    agent.update(batch)

    delta_critic = (agent.critic.q1.net[0].weight - w_critic_before).abs().sum().item()
    delta_target = (agent.target_critic.q1.net[0].weight - w_target_before).abs().sum().item()

    # Target moved, but much less than critic (τ=0.005 by config).
    assert delta_target > 0.0
    assert delta_target < delta_critic


def test_sac_state_dict_roundtrip():
    cfg = _cfg()
    device = torch.device("cpu")
    policy = build_policy_from_config(cfg)
    agent = build_sac_agent(cfg, policy=policy, device=device)
    sd = agent.state_dict()
    # Build a fresh agent and load.
    policy2 = build_policy_from_config(cfg)
    agent2 = build_sac_agent(cfg, policy=policy2, device=device)
    agent2.load_state_dict(sd)
    # Verify alpha matches.
    assert agent2.log_alpha.item() == pytest.approx(agent.log_alpha.item())
