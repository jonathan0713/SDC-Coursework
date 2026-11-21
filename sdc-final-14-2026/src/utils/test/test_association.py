import numpy as np
import pytest

from utils.association import (
    build_cost_matrix,
    gate,
    hungarian_assignment,
    mahalanobis_distance,
    nearest_neighbor_match,
)

IDENTITY_2D = np.eye(2)


# ── mahalanobis_distance ──────────────────────────────────────────────────────

def test_mahalanobis_zero_distance():
    x = np.array([1.0, 2.0])
    assert mahalanobis_distance(x, x, IDENTITY_2D) == pytest.approx(0.0)


def test_mahalanobis_identity_equals_euclidean():
    x = np.array([4.0, 0.0])
    mean = np.array([1.0, 0.0])
    assert mahalanobis_distance(x, mean, IDENTITY_2D) == pytest.approx(3.0)


def test_mahalanobis_scaled_covariance():
    # cov = diag(4, 1) → cov_inv = diag(0.25, 1)
    # diff = [2, 0] → d = sqrt(2²·0.25) = sqrt(1) = 1.0
    cov_inv = np.diag([0.25, 1.0])
    x = np.array([2.0, 0.0])
    mean = np.array([0.0, 0.0])
    assert mahalanobis_distance(x, mean, cov_inv) == pytest.approx(1.0)


def test_mahalanobis_symmetric():
    x = np.array([3.0, 4.0])
    mean = np.array([0.0, 0.0])
    d1 = mahalanobis_distance(x, mean, IDENTITY_2D)
    d2 = mahalanobis_distance(mean, x, IDENTITY_2D)
    assert d1 == pytest.approx(d2)


# ── nearest_neighbor_match ────────────────────────────────────────────────────

def test_nn_empty_tracks():
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        np.empty((0, 2)), np.array([[1.0, 2.0]]), IDENTITY_2D, max_distance=5.0
    )
    assert matched == []
    assert unmatched_dets == [0]
    assert unmatched_tracks == []


def test_nn_empty_detections():
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        np.array([[1.0, 2.0]]), np.empty((0, 2)), IDENTITY_2D, max_distance=5.0
    )
    assert matched == []
    assert unmatched_dets == []
    assert unmatched_tracks == [0]


def test_nn_perfect_one_to_one():
    tracks = np.array([[0.0, 0.0], [10.0, 10.0]])
    dets = np.array([[0.1, 0.1], [10.1, 10.1]])
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        tracks, dets, IDENTITY_2D, max_distance=5.0
    )
    assert len(matched) == 2
    assert unmatched_dets == []
    assert unmatched_tracks == []
    # 每個 track 應配對到最近的 detection
    assert (0, 0) in matched
    assert (1, 1) in matched


def test_nn_gating_rejects_far_detections():
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[100.0, 100.0]])  # 遠超 max_distance
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        tracks, dets, IDENTITY_2D, max_distance=5.0
    )
    assert matched == []
    assert unmatched_dets == [0]
    assert unmatched_tracks == [0]


def test_nn_one_track_two_close_detections():
    # 兩個 detection 都在範圍內，應選最近的那個
    tracks = np.array([[0.0, 0.0]])
    dets = np.array([[1.0, 0.0], [0.5, 0.0]])  # det[1] 更近
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        tracks, dets, IDENTITY_2D, max_distance=5.0
    )
    assert matched == [(0, 1)]
    assert unmatched_dets == [0]
    assert unmatched_tracks == []


def test_nn_two_tracks_compete_for_one_detection():
    # det[0] 同時靠近兩個 track，應給距離最小的 track
    tracks = np.array([[1.0, 0.0], [0.5, 0.0]])  # track[1] 更近 det[0]
    dets = np.array([[0.0, 0.0]])
    matched, unmatched_dets, unmatched_tracks = nearest_neighbor_match(
        tracks, dets, IDENTITY_2D, max_distance=5.0
    )
    assert len(matched) == 1
    assert matched[0][1] == 0           # det[0] 被配對
    assert matched[0][0] == 1           # 給 track[1]（較近）
    assert 0 in unmatched_tracks        # track[0] 未配對


# ── gate ──────────────────────────────────────────────────────────────────────

def _covs(n: int, scale: float = 1.0) -> list[np.ndarray]:
    return [np.eye(2) * scale for _ in range(n)]


def test_gate_passes_close_pairs():
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[0.5, 0.0]])
    pairs = gate(preds, dets, _covs(1), max_distance=2.0)
    assert (0, 0) in pairs


def test_gate_rejects_far_pairs():
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[100.0, 0.0]])
    pairs = gate(preds, dets, _covs(1), max_distance=2.0)
    assert len(pairs) == 0


def test_gate_uses_per_track_covariance():
    preds = np.array([[0.0, 0.0], [0.0, 0.0]])
    dets  = np.array([[3.0, 0.0]])
    # track 0: tight covariance → dist = 3 > threshold 2 → gated out
    # track 1: loose covariance (scale=100) → dist = 3/10 < threshold → passes
    covs = [np.eye(2) * 1.0, np.eye(2) * 100.0]
    pairs = gate(preds, dets, covs, max_distance=2.0)
    assert (0, 0) not in pairs
    assert (1, 0) in pairs


def test_gate_euclidean_hard_gate_overrides_large_cov():
    # det is 20 m away; huge covariance makes Mahalanobis dist ≈ 0.63 (would pass
    # soft gate alone), but Euclidean hard gate at default 10 m must reject it.
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[20.0, 0.0]])
    huge_cov = [np.eye(2) * 1000.0]
    pairs = gate(preds, dets, huge_cov, max_distance=2.0)  # max_euclidean=10.0
    assert len(pairs) == 0


def test_gate_custom_max_euclidean_allows_far_pair():
    # Same scenario but caller raises the Euclidean cap → pair should now pass.
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[20.0, 0.0]])
    huge_cov = [np.eye(2) * 1000.0]
    pairs = gate(preds, dets, huge_cov, max_distance=2.0, max_euclidean=50.0)
    assert (0, 0) in pairs


# ── build_cost_matrix ─────────────────────────────────────────────────────────

def test_build_cost_matrix_shape():
    preds = np.array([[0.0, 0.0], [5.0, 5.0]])
    dets  = np.array([[0.1, 0.0], [5.1, 5.0], [20.0, 20.0]])
    valid = {(0, 0), (1, 1)}
    cost  = build_cost_matrix(valid, preds, dets, _covs(2))
    assert cost.shape == (2, 3)


def test_build_cost_matrix_invalid_pairs_are_inf():
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[1.0, 0.0], [2.0, 0.0]])
    valid = {(0, 0)}   # only (0,0) is valid
    cost  = build_cost_matrix(valid, preds, dets, _covs(1))
    assert np.isfinite(cost[0, 0])
    assert np.isinf(cost[0, 1])


def test_build_cost_matrix_valid_cost_equals_mahalanobis():
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[3.0, 4.0]])
    valid = {(0, 0)}
    cost  = build_cost_matrix(valid, preds, dets, _covs(1))
    expected = np.sqrt(3**2 + 4**2)   # identity cov → euclidean = 5
    assert cost[0, 0] == pytest.approx(expected)


def test_build_cost_matrix_range_rate_added_to_cost():
    # With matching range_rate the cost equals pure Mahalanobis.
    preds = np.array([[0.0, 0.0]])
    dets  = np.array([[3.0, 4.0]])
    valid = {(0, 0)}
    track_rr = np.array([5.0])
    det_rr   = np.array([5.0])   # identical → rr_cost = 0
    cost = build_cost_matrix(valid, preds, dets, _covs(1),
                             track_range_rates=track_rr, det_range_rates=det_rr)
    expected = np.sqrt(3**2 + 4**2)
    assert cost[0, 0] == pytest.approx(expected)


def test_build_cost_matrix_range_rate_disambiguates_nearby_clusters():
    # Two detections at the same position; track has rr=5.
    # det[0] matches rr, det[1] doesn't → cost[0,0] < cost[0,1].
    preds    = np.array([[0.0, 0.0]])
    dets     = np.array([[1.0, 0.0], [1.0, 0.0]])
    valid    = {(0, 0), (0, 1)}
    track_rr = np.array([5.0])
    det_rr   = np.array([5.0, -2.0])   # 7 m/s difference for det[1]
    cost = build_cost_matrix(valid, preds, dets, _covs(1),
                             track_range_rates=track_rr, det_range_rates=det_rr,
                             range_rate_sigma=2.0)
    assert cost[0, 0] < cost[0, 1]


# ── hungarian_assignment ──────────────────────────────────────────────────────

def test_hungarian_empty_matrix():
    matched, u_dets, u_tracks = hungarian_assignment(np.empty((0, 3)))
    assert matched == []
    assert u_dets == [0, 1, 2]
    assert u_tracks == []


def test_hungarian_all_inf():
    cost = np.full((2, 2), np.inf)
    matched, u_dets, u_tracks = hungarian_assignment(cost)
    assert matched == []
    assert u_dets == [0, 1]
    assert u_tracks == [0, 1]


def test_hungarian_optimal_assignment():
    # Optimal: (0→1, cost 0.1) + (1→0, cost 0.2) = 0.3
    # Greedy nearest would pick (0→0 cost 1.0) first → suboptimal
    cost = np.array([
        [1.0, 0.1],
        [0.2, 5.0],
    ])
    matched, u_dets, u_tracks = hungarian_assignment(cost)
    assert (0, 1) in matched
    assert (1, 0) in matched
    assert u_dets == []
    assert u_tracks == []


def test_hungarian_partial_match_with_inf():
    cost = np.array([
        [1.0, np.inf],
        [np.inf, 2.0],
    ])
    matched, u_dets, u_tracks = hungarian_assignment(cost)
    assert (0, 0) in matched
    assert (1, 1) in matched
