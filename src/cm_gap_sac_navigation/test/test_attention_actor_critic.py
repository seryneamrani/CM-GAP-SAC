"""End-to-end tests for attention, actor, critic, and policy.

These tests verify the v0.2 architecture without requiring ROS or Gazebo.
They cover:
  - LidarSelfAttention shape and masking
  - PedestrianCrossAttention with and without IMU in query
  - PedestrianCrossAttention masking (padded slots get zero weight)
  - PedestrianCrossAttention all-padded (no NaN propagation)
  - TriSourceGate softmax property (sums to 1 along source axis)
  - CrossModalAttention shape and entropy bounds
  - SquashedGaussianActor: action bounds, log-prob shape, deterministic
  - TwinCritic: two distinct outputs, min reduction
  - End-to-end CmGapSacPolicy from a fake observation Dict
  - Permutation invariance of the full pipeline w.r.t. pedestrians
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cm_gap_sac_navigation.models.actor import SquashedGaussianActor
from cm_gap_sac_navigation.models.attention import (
    CrossModalAttention,
    LidarSelfAttention,
    PedestrianCrossAttention,
    TriSourceGate,
    pedestrian_attention_entropy,
)
from cm_gap_sac_navigation.models.critic import (
    TwinCritic,
    make_target,
    soft_update,
)
from cm_gap_sac_navigation.models.encoders import (
    GoalEncoder,
    ImuEncoder,
    LidarEncoder,
    MultiModalEncoder,
    PedestrianEncoder,
)
from cm_gap_sac_navigation.models.policy import (
    CmGapSacPolicy,
    build_policy_from_config,
)
from cm_gap_sac_navigation.utils.config_loader import load_config


# ======================================================================
# Self-attention LiDAR
# ======================================================================
def test_lidar_self_attention_shape():
    sa = LidarSelfAttention(feature_dim=64, n_heads=4)
    out = sa(torch.randn(3, 9, 64))
    assert out.shape == (3, 64)   # mean-pooled over patches


def test_lidar_self_attention_n_heads_validation():
    with pytest.raises(ValueError, match="divisible"):
        LidarSelfAttention(feature_dim=65, n_heads=4)


# ======================================================================
# Cross-attention pedestrian
# ======================================================================
def test_cross_attention_shape_with_imu():
    ca = PedestrianCrossAttention(
        feature_dim=64, goal_dim=32, n_heads=4, use_imu_in_query=True,
    )
    c, alpha = ca(
        h_ped=torch.randn(2, 5, 64),
        ped_mask=torch.ones(2, 5),
        h_goal=torch.randn(2, 32),
        h_imu=torch.randn(2, 64),
    )
    assert c.shape == (2, 64)
    assert alpha.shape == (2, 5)
    # With all valid, weights should sum to ~1 per sample.
    np.testing.assert_allclose(alpha.sum(dim=-1).detach().numpy(),
                               np.ones(2), atol=1e-5)


def test_cross_attention_shape_without_imu():
    """Ablation A1: use_imu_in_query=False reduces to goal-only query."""
    ca = PedestrianCrossAttention(
        feature_dim=64, goal_dim=32, n_heads=4, use_imu_in_query=False,
    )
    c, alpha = ca(
        h_ped=torch.randn(2, 5, 64),
        ped_mask=torch.ones(2, 5),
        h_goal=torch.randn(2, 32),
        h_imu=torch.zeros(2, 64),   # ignored when use_imu_in_query=False
    )
    assert c.shape == (2, 64)
    assert alpha.shape == (2, 5)


def test_cross_attention_masking():
    """Padded pedestrian slots must receive exactly zero attention weight."""
    ca = PedestrianCrossAttention(feature_dim=64, goal_dim=32, n_heads=4)
    # Sample 0: only first 2 slots are real.
    # Sample 1: only first slot is real.
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    c, alpha = ca(
        h_ped=torch.randn(2, 5, 64),
        ped_mask=mask,
        h_goal=torch.randn(2, 32),
        h_imu=torch.randn(2, 64),
    )
    # Padded positions must be exactly 0.
    np.testing.assert_array_equal(
        (alpha * (1 - mask)).abs().detach().numpy(),
        np.zeros((2, 5), dtype=np.float32),
    )
    # Valid positions sum to 1.
    valid_sum = (alpha * mask).sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(valid_sum, np.ones(2), atol=1e-5)


def test_cross_attention_all_padded_no_nan():
    """If all pedestrian slots are padded, the output must be finite (no NaN)."""
    ca = PedestrianCrossAttention(feature_dim=64, goal_dim=32, n_heads=4)
    mask = torch.zeros(2, 5)
    c, alpha = ca(
        h_ped=torch.randn(2, 5, 64),
        ped_mask=mask,
        h_goal=torch.randn(2, 32),
        h_imu=torch.randn(2, 64),
    )
    assert torch.isfinite(c).all().item()
    assert torch.isfinite(alpha).all().item()
    # alpha is all zero in this case (after nan_to_num).
    np.testing.assert_array_equal(alpha.detach().numpy(),
                                  np.zeros((2, 5), dtype=np.float32))


# ======================================================================
# TriSourceGate
# ======================================================================
def test_tri_source_gate_partition_sums_to_one():
    """Per-feature softmax over 3 sources => g_l + g_p + g_i == 1 elementwise."""
    g = TriSourceGate(feature_dim=64, goal_dim=32)
    g_l, g_p, g_i = g(
        c_lidar=torch.randn(4, 64),
        c_ped=torch.randn(4, 64),
        h_imu=torch.randn(4, 64),
        h_goal=torch.randn(4, 32),
    )
    total = (g_l + g_p + g_i).detach().numpy()
    np.testing.assert_allclose(total, np.ones((4, 64), dtype=np.float32),
                               atol=1e-5)


def test_tri_source_gate_values_in_unit_interval():
    g = TriSourceGate(feature_dim=64, goal_dim=32)
    g_l, g_p, g_i = g(
        c_lidar=torch.randn(2, 64),
        c_ped=torch.randn(2, 64),
        h_imu=torch.randn(2, 64),
        h_goal=torch.randn(2, 32),
    )
    for g_ in (g_l, g_p, g_i):
        assert (g_ >= 0).all().item()
        assert (g_ <= 1).all().item()


# ======================================================================
# CrossModalAttention (full)
# ======================================================================
def test_cross_modal_attention_shapes():
    ca = CrossModalAttention(feature_dim=64, goal_dim=32, n_heads=4)
    out = ca(
        h_lidar=torch.randn(2, 9, 64),
        h_ped=torch.randn(2, 5, 64),
        ped_mask=torch.ones(2, 5),
        h_imu=torch.randn(2, 64),
        h_goal=torch.randn(2, 32),
    )
    assert out["xi"].shape == (2, 64 + 32)            # latent for SAC
    assert out["alpha_ped"].shape == (2, 5)
    assert out["z"].shape == (2, 64)
    assert out["gate_lidar"].shape == (2, 64)
    assert out["gate_ped"].shape == (2, 64)
    assert out["gate_imu"].shape == (2, 64)


def test_cross_modal_attention_gradient_flow():
    ca = CrossModalAttention(feature_dim=64, goal_dim=32, n_heads=4)
    h_lidar = torch.randn(2, 9, 64, requires_grad=True)
    out = ca(
        h_lidar=h_lidar,
        h_ped=torch.randn(2, 5, 64),
        ped_mask=torch.ones(2, 5),
        h_imu=torch.randn(2, 64),
        h_goal=torch.randn(2, 32),
    )
    out["xi"].sum().backward()
    assert h_lidar.grad is not None
    assert h_lidar.grad.abs().sum().item() > 0


# ======================================================================
# Pedestrian attention entropy
# ======================================================================
def test_entropy_bounded_in_zero_one():
    alpha = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2],     # uniform
                          [1.0, 0.0, 0.0, 0.0, 0.0],     # peaked
                          [0.5, 0.5, 0.0, 0.0, 0.0]])    # half
    mask = torch.tensor([[1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0]], dtype=torch.float32)
    h = pedestrian_attention_entropy(alpha, mask)
    assert h.shape == (3,)
    assert (h >= 0).all().item()
    assert (h <= 1).all().item()
    # Uniform over 5 valid -> high entropy
    assert h[0].item() > 0.95
    # Peaked -> ~0 entropy
    assert h[1].item() < 0.05


def test_entropy_zero_when_no_pedestrian():
    alpha = torch.zeros(2, 5)
    mask = torch.zeros(2, 5)
    h = pedestrian_attention_entropy(alpha, mask)
    np.testing.assert_array_equal(h.detach().numpy(), np.zeros(2))


# ======================================================================
# Squashed Gaussian Actor
# ======================================================================
def test_actor_action_within_bounds():
    actor = SquashedGaussianActor(
        latent_dim=96, action_dim=2,
        action_low=(-0.3, -1.0), action_high=(0.6, 1.0),
    )
    xi = torch.randn(64, 96)
    action, log_prob, mean_action, _ = actor(xi)
    assert action.shape == (64, 2)
    assert log_prob.shape == (64,)
    assert (action[:, 0] >= -0.3 - 1e-5).all().item()
    assert (action[:, 0] <=  0.6 + 1e-5).all().item()
    assert (action[:, 1] >= -1.0 - 1e-5).all().item()
    assert (action[:, 1] <=  1.0 + 1e-5).all().item()


def test_actor_log_prob_finite():
    actor = SquashedGaussianActor(latent_dim=96)
    _, log_prob, _, _ = actor(torch.randn(32, 96))
    assert torch.isfinite(log_prob).all().item()


def test_actor_deterministic_repeatable():
    """deterministic=True must produce identical actions on repeated calls."""
    torch.manual_seed(0)
    actor = SquashedGaussianActor(latent_dim=96).eval()
    xi = torch.randn(8, 96)
    a1, _, _, _ = actor(xi, deterministic=True)
    a2, _, _, _ = actor(xi, deterministic=True)
    np.testing.assert_allclose(a1.detach().numpy(), a2.detach().numpy())


def test_actor_stochastic_varies():
    """Without deterministic=True, two consecutive samples should differ."""
    torch.manual_seed(0)
    actor = SquashedGaussianActor(latent_dim=96).eval()
    xi = torch.randn(8, 96)
    a1, _, _, _ = actor(xi)
    a2, _, _, _ = actor(xi)
    assert (a1 - a2).abs().max().item() > 1e-4


def test_actor_gradient_flow():
    actor = SquashedGaussianActor(latent_dim=96)
    xi = torch.randn(4, 96, requires_grad=True)
    _, log_prob, _, _ = actor(xi)
    log_prob.sum().backward()
    assert xi.grad is not None and xi.grad.abs().sum().item() > 0


# ======================================================================
# Twin Critic
# ======================================================================
def test_twin_critic_shapes():
    critic = TwinCritic(latent_dim=96, action_dim=2)
    xi = torch.randn(16, 96)
    a = torch.randn(16, 2)
    q1, q2 = critic(xi, a)
    assert q1.shape == (16,)
    assert q2.shape == (16,)


def test_twin_critic_two_networks_distinct():
    """Q1 and Q2 should give different outputs on the same input."""
    torch.manual_seed(0)
    critic = TwinCritic(latent_dim=96, action_dim=2)
    xi = torch.randn(8, 96)
    a = torch.randn(8, 2)
    q1, q2 = critic(xi, a)
    assert (q1 - q2).abs().max().item() > 1e-4


def test_q_min_takes_minimum():
    critic = TwinCritic(latent_dim=96, action_dim=2)
    xi = torch.randn(8, 96)
    a = torch.randn(8, 2)
    q1, q2 = critic(xi, a)
    qmin = critic.q_min(xi, a)
    expected = torch.min(q1, q2)
    np.testing.assert_allclose(qmin.detach().numpy(), expected.detach().numpy())


def test_target_critic_no_grad():
    """make_target produces a deep-copy with frozen parameters."""
    critic = TwinCritic(latent_dim=96, action_dim=2)
    target = make_target(critic)
    for p in target.parameters():
        assert not p.requires_grad


def test_soft_update_polyak():
    """After soft_update with τ=1, target equals source. With τ=0, no change."""
    critic = TwinCritic(latent_dim=96, action_dim=2)
    target = make_target(critic)

    # τ=1 -> target := source
    soft_update(source=critic, target=target, tau=1.0)
    for ps, pt in zip(critic.parameters(), target.parameters()):
        np.testing.assert_allclose(ps.data.numpy(), pt.data.numpy())

    # Now perturb source, τ=0 -> target unchanged.
    target_before = [p.data.clone() for p in target.parameters()]
    for p in critic.parameters():
        p.data.add_(1.0)
    soft_update(source=critic, target=target, tau=0.0)
    for pt_new, pt_old in zip(target.parameters(), target_before):
        np.testing.assert_allclose(pt_new.data.numpy(), pt_old.numpy())


# ======================================================================
# End-to-end policy
# ======================================================================
def _make_test_policy() -> CmGapSacPolicy:
    """Build a small policy directly (not from config) for unit tests."""
    le = LidarEncoder(n_beams=36, feature_dim=64)
    pe = PedestrianEncoder(n_features=5, feature_dim=64)
    ie = ImuEncoder(n_features=6, feature_dim=64)
    ge = GoalEncoder(n_features=4, feature_dim=32)
    enc = MultiModalEncoder(le, pe, ie, ge)
    attn = CrossModalAttention(feature_dim=64, goal_dim=32, n_heads=4)
    actor = SquashedGaussianActor(
        latent_dim=attn.latent_dim, action_dim=2,
        action_low=(-0.3, -1.0), action_high=(0.6, 1.0),
    )
    return CmGapSacPolicy(encoder=enc, attention=attn, actor=actor)


def test_policy_end_to_end_shapes():
    policy = _make_test_policy()
    B, K = 4, 5
    out = policy(
        lidar=torch.randn(B, 36),
        pedestrians=torch.randn(B, K, 5),
        ped_mask=torch.ones(B, K),
        imu=torch.randn(B, 6),
        goal=torch.randn(B, 4),
    )
    assert out["action"].shape == (B, 2)
    assert out["log_prob"].shape == (B,)
    assert out["xi"].shape == (B, 96)
    assert out["alpha_ped"].shape == (B, K)
    assert out["ped_attn_entropy"].shape == (B,)
    assert out["gates"]["lidar"].shape == (B, 64)


def test_policy_action_within_robot_bounds():
    policy = _make_test_policy()
    out = policy(
        lidar=torch.randn(8, 36),
        pedestrians=torch.randn(8, 5, 5),
        ped_mask=torch.ones(8, 5),
        imu=torch.randn(8, 6),
        goal=torch.randn(8, 4),
    )
    a = out["action"]
    assert (a[:, 0] >= -0.3 - 1e-5).all().item()
    assert (a[:, 0] <=  0.6 + 1e-5).all().item()
    assert (a[:, 1] >= -1.0 - 1e-5).all().item()
    assert (a[:, 1] <=  1.0 + 1e-5).all().item()


def test_policy_gradient_flows_to_encoders():
    """Gradient from log_prob must reach the LiDAR encoder weights."""
    policy = _make_test_policy()
    out = policy(
        lidar=torch.randn(4, 36),
        pedestrians=torch.randn(4, 5, 5),
        ped_mask=torch.ones(4, 5),
        imu=torch.randn(4, 6),
        goal=torch.randn(4, 4),
    )
    out["log_prob"].sum().backward()
    grad = policy.encoder.lidar.conv1.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


def test_policy_act_from_obs_numpy():
    """Convenience method consuming a numpy obs Dict like the env produces."""
    policy = _make_test_policy().eval()
    obs = {
        "lidar":       np.random.randn(36).astype(np.float32),
        "pedestrians": np.random.randn(5, 5).astype(np.float32),
        "ped_mask":    np.ones(5, dtype=np.uint8),
        "imu":         np.random.randn(6).astype(np.float32),
        "goal":        np.random.randn(4).astype(np.float32),
    }
    action, log_prob, entropy = policy.act_from_obs(obs, device=torch.device("cpu"))
    assert action.shape == (2,)
    assert action[0].item() >= -0.3 - 1e-5 and action[0].item() <= 0.6 + 1e-5
    assert action[1].item() >= -1.0 - 1e-5 and action[1].item() <= 1.0 + 1e-5
    assert log_prob.numel() == 1
    assert entropy.numel() == 1


def test_policy_pedestrian_permutation_invariance():
    """Permuting pedestrians must not change the action (deterministic mode).

    This is the formal end-to-end test of the Deep Sets + masked
    cross-attention design: the pipeline must be invariant to pedestrian
    ordering.
    """
    torch.manual_seed(0)
    policy = _make_test_policy().eval()
    B, K = 1, 5

    lidar = torch.randn(B, 36)
    ped = torch.randn(B, K, 5)
    mask = torch.ones(B, K)
    imu = torch.randn(B, 6)
    goal = torch.randn(B, 4)

    perm = torch.tensor([2, 0, 4, 1, 3])

    with torch.no_grad():
        out_orig = policy(
            lidar=lidar, pedestrians=ped, ped_mask=mask, imu=imu, goal=goal,
            deterministic=True,
        )
        out_perm = policy(
            lidar=lidar, pedestrians=ped[:, perm, :], ped_mask=mask,
            imu=imu, goal=goal, deterministic=True,
        )

    np.testing.assert_allclose(
        out_orig["action"].numpy(), out_perm["action"].numpy(),
        atol=1e-5,
    )


# ======================================================================
# Build from config (integration with YAML)
# ======================================================================
def test_build_policy_from_config():
    """Verify the YAML-driven factory produces a working policy."""
    cfg_path = Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml"
    cfg = load_config(cfg_path)
    policy = build_policy_from_config(cfg)

    # Smoke forward with shapes derived from cfg.
    B = 2
    out = policy(
        lidar=torch.randn(B, cfg.observation.lidar.n_beams),
        pedestrians=torch.randn(B, cfg.observation.pedestrian.k_max,
                                cfg.observation.pedestrian.n_features),
        ped_mask=torch.ones(B, cfg.observation.pedestrian.k_max),
        imu=torch.randn(B, cfg.observation.imu.n_features),
        goal=torch.randn(B, cfg.observation.goal.n_features),
    )
    assert out["action"].shape == (B, cfg.action.get("dim", 2))
    assert torch.isfinite(out["log_prob"]).all().item()


def test_build_policy_from_config_ablation_a1():
    """Toggling use_imu_in_query=False must still produce valid actions."""
    cfg_path = Path(__file__).parents[1] / "config" / "cm_gap_sac.yaml"
    cfg = load_config(cfg_path)
    cfg.attention["use_imu_in_query"] = False
    policy = build_policy_from_config(cfg)
    out = policy(
        lidar=torch.randn(2, cfg.observation.lidar.n_beams),
        pedestrians=torch.randn(2, 5, 5),
        ped_mask=torch.ones(2, 5),
        imu=torch.randn(2, 6),
        goal=torch.randn(2, 4),
    )
    assert out["action"].shape == (2, 2)
    assert torch.isfinite(out["log_prob"]).all().item()
