"""Tests for encoder modules (v0.2)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cm_gap_sac_navigation.models.encoders import (
    GoalEncoder,
    ImuEncoder,
    LidarEncoder,
    MultiModalEncoder,
    PedestrianEncoder,
)


# ----------------------------------------------------------------------
# LidarEncoder
# ----------------------------------------------------------------------
def test_lidar_encoder_output_shape():
    enc = LidarEncoder(n_beams=36, feature_dim=64)
    B = 4
    out = enc(torch.randn(B, 36))
    assert out.shape == (B, enc.n_patches, 64)


def test_lidar_encoder_n_patches_36_to_9():
    enc = LidarEncoder(n_beams=36, feature_dim=64)
    assert enc.n_patches == 9


def test_lidar_encoder_circular_pad_continuity():
    torch.manual_seed(0)
    enc = LidarEncoder(n_beams=36, feature_dim=64).eval()
    x1 = torch.randn(1, 36)
    x2 = x1.clone(); x2[0, 0] += 5.0
    with torch.no_grad():
        out1, out2 = enc(x1), enc(x2)
    assert (out1[:, -1] - out2[:, -1]).abs().max().item() > 1e-6


def test_lidar_encoder_gradient_flow():
    enc = LidarEncoder(n_beams=36, feature_dim=64)
    x = torch.randn(2, 36, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0


# ----------------------------------------------------------------------
# PedestrianEncoder
# ----------------------------------------------------------------------
def test_pedestrian_encoder_shape_and_mask():
    enc = PedestrianEncoder(n_features=5, feature_dim=64)
    mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )
    out = enc(torch.randn(3, 5, 5), mask)
    assert out.shape == (3, 5, 64)
    assert torch.all(out[(1 - mask).bool()] == 0.0)


def test_pedestrian_encoder_permutation_equivariance():
    torch.manual_seed(0)
    enc = PedestrianEncoder(n_features=5, feature_dim=64).eval()
    x = torch.randn(1, 5, 5)
    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.no_grad():
        out_orig = enc(x)
        out_perm = enc(x[:, perm, :])
    np.testing.assert_allclose(
        out_orig[0, perm].numpy(), out_perm[0].numpy(), atol=1e-6,
    )


# ----------------------------------------------------------------------
# ImuEncoder (NEW v0.2)
# ----------------------------------------------------------------------
def test_imu_encoder_shape():
    enc = ImuEncoder(n_features=6, feature_dim=64)
    out = enc(torch.randn(8, 6))
    assert out.shape == (8, 64)


def test_imu_encoder_gradient_flow():
    enc = ImuEncoder(n_features=6, feature_dim=64)
    x = torch.randn(4, 6, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0


def test_imu_encoder_rejects_wrong_dim():
    enc = ImuEncoder(n_features=6, feature_dim=64)
    with pytest.raises(ValueError):
        enc(torch.randn(6))


# ----------------------------------------------------------------------
# GoalEncoder
# ----------------------------------------------------------------------
def test_goal_encoder_shape():
    enc = GoalEncoder(n_features=4, feature_dim=32)
    out = enc(torch.randn(8, 4))
    assert out.shape == (8, 32)


# ----------------------------------------------------------------------
# MultiModalEncoder integration (4 modalities in v0.2)
# ----------------------------------------------------------------------
def test_multimodal_encoder_end_to_end():
    le = LidarEncoder(n_beams=36, feature_dim=64)
    pe = PedestrianEncoder(n_features=5, feature_dim=64)
    ie = ImuEncoder(n_features=6, feature_dim=64)
    ge = GoalEncoder(n_features=4, feature_dim=32)
    mm = MultiModalEncoder(le, pe, ie, ge)

    B, K = 2, 5
    out = mm(
        lidar=torch.randn(B, 36),
        pedestrians=torch.randn(B, K, 5),
        ped_mask=torch.ones(B, K, dtype=torch.float32),
        imu=torch.randn(B, 6),
        goal=torch.randn(B, 4),
    )
    assert out["h_lidar"].shape == (B, le.n_patches, 64)
    assert out["h_pedestrian"].shape == (B, K, 64)
    assert out["h_imu"].shape == (B, 64)
    assert out["h_goal"].shape == (B, 32)
    assert out["ped_mask"].shape == (B, K)


def test_multimodal_dim_mismatch_raises():
    le = LidarEncoder(n_beams=36, feature_dim=64)
    pe = PedestrianEncoder(n_features=5, feature_dim=32)   # mismatch
    ie = ImuEncoder(n_features=6, feature_dim=64)
    ge = GoalEncoder(n_features=4, feature_dim=32)
    with pytest.raises(ValueError, match="feature_dim"):
        MultiModalEncoder(le, pe, ie, ge)


def test_multimodal_imu_dim_mismatch_raises():
    le = LidarEncoder(n_beams=36, feature_dim=64)
    pe = PedestrianEncoder(n_features=5, feature_dim=64)
    ie = ImuEncoder(n_features=6, feature_dim=32)        # mismatch
    ge = GoalEncoder(n_features=4, feature_dim=32)
    with pytest.raises(ValueError, match="feature_dim"):
        MultiModalEncoder(le, pe, ie, ge)
