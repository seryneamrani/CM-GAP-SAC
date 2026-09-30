"""Tests for pedestrian state assembly."""
import numpy as np
import pytest

from cm_gap_sac_navigation.perception.track_state import (
    PedestrianTrack,
    assemble_pedestrian_state,
)


def make_track(tid: int, x: float, y: float, vx: float = 0.0,
               vy: float = 0.0, age: int = 10) -> PedestrianTrack:
    return PedestrianTrack(
        track_id=tid,
        position_world=np.array([x, y], dtype=np.float32),
        velocity_world=np.array([vx, vy], dtype=np.float32),
        age_frames=age,
    )


def test_empty_tracks_returns_zero_padded():
    feat, mask = assemble_pedestrian_state(
        tracks=[],
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=0.0,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    assert feat.shape == (5, 5)
    assert mask.shape == (5,)
    assert mask.sum() == 0
    np.testing.assert_array_equal(feat, np.zeros((5, 5)))


def test_single_pedestrian_correctly_placed():
    tracks = [make_track(1, x=2.0, y=0.0, vx=0.5, vy=0.0, age=25)]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=0.0,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    assert mask[0] == 1
    assert mask[1:].sum() == 0
    np.testing.assert_allclose(feat[0, :2], [2.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(feat[0, 2:4], [0.5, 0.0], atol=1e-6)
    assert feat[0, 4] == pytest.approx(0.5)  # 25 / 50


def test_relevance_radius_filters_far_tracks():
    tracks = [
        make_track(1, x=2.0, y=0.0),
        make_track(2, x=10.0, y=0.0),  # too far
    ]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=0.0,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    assert mask.sum() == 1
    np.testing.assert_allclose(feat[0, :2], [2.0, 0.0], atol=1e-6)


def test_more_than_k_max_keeps_nearest():
    tracks = [make_track(i, x=float(i), y=0.0) for i in range(1, 8)]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=0.0,
        k_max=3,
        relevance_radius=10.0,
        track_age_norm=50,
    )
    assert mask.sum() == 3
    # Three nearest are at x=1, 2, 3 (distances 1, 2, 3). Order in output
    # should be ascending by distance.
    np.testing.assert_allclose(
        feat[:3, 0], [1.0, 2.0, 3.0], atol=1e-6,
    )


def test_yaw_rotation_applied():
    """Robot yaw=pi/2 -> a track at world (1, 0) should appear at (0, -1) in robot frame."""
    tracks = [make_track(1, x=1.0, y=0.0)]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=np.pi / 2,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    np.testing.assert_allclose(feat[0, :2], [0.0, -1.0], atol=1e-6)


def test_age_clamped_at_norm():
    tracks = [make_track(1, x=1.0, y=0.0, age=200)]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.zeros(2, dtype=np.float32),
        robot_yaw=0.0,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    assert feat[0, 4] == pytest.approx(1.0)  # clamped


def test_robot_translation_subtracted():
    tracks = [make_track(1, x=5.0, y=3.0)]
    feat, mask = assemble_pedestrian_state(
        tracks=tracks,
        robot_xy=np.array([4.0, 3.0], dtype=np.float32),
        robot_yaw=0.0,
        k_max=5,
        relevance_radius=4.0,
        track_age_norm=50,
    )
    np.testing.assert_allclose(feat[0, :2], [1.0, 0.0], atol=1e-6)
