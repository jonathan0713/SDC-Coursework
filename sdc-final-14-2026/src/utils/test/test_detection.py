import numpy as np
import pytest

from utils.detection import build_detections, calculate_cluster_centroids


def make_radar_pc(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


# ── 正常案例 ──────────────────────────────────────────────────────────────────

def test_two_clusters_centroids():
    moving_mask = np.array([0, 0, 1, 1])
    radar_pc = make_radar_pc([
        [1.0, 2.0, 3.0, -1.0, 5.0],
        [3.0, 4.0, 5.0, -3.0, 6.0],
        [10.0, 20.0, 30.0, 5.0, 7.0],
        [12.0, 22.0, 32.0, 7.0, 8.0],
    ])
    centroids, velocities, points_dict = calculate_cluster_centroids(moving_mask, radar_pc)

    assert set(centroids.keys()) == {0, 1}
    np.testing.assert_allclose(centroids[0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(centroids[1], [11.0, 21.0, 31.0])
    assert velocities[0] == pytest.approx(-2.0)
    assert velocities[1] == pytest.approx(6.0)


def test_points_dict_contains_all_points():
    moving_mask = np.array([0, 0, 0])
    radar_pc = make_radar_pc([
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0, 2.0, 1.0],
        [3.0, 0.0, 0.0, 3.0, 1.0],
    ])
    _, _, points_dict = calculate_cluster_centroids(moving_mask, radar_pc)

    assert len(points_dict[0]) == 3
    np.testing.assert_array_equal(points_dict[0], radar_pc)


# ── 邊界案例 ──────────────────────────────────────────────────────────────────

def test_no_valid_clusters_returns_empty():
    # moving_mask 全是負值（例如 separate_static_and_moving 後仍可能有 -2）
    moving_mask = np.array([-2, -2])
    radar_pc = make_radar_pc([
        [1.0, 2.0, 3.0, 1.0, 1.0],
        [4.0, 5.0, 6.0, 2.0, 1.0],
    ])
    centroids, velocities, points_dict = calculate_cluster_centroids(moving_mask, radar_pc)

    assert centroids == {}
    assert velocities == {}
    assert points_dict == {}


def test_single_point_cluster():
    moving_mask = np.array([0])
    radar_pc = make_radar_pc([[5.0, 6.0, 7.0, 3.0, 2.0]])
    centroids, velocities, points_dict = calculate_cluster_centroids(moving_mask, radar_pc)

    assert 0 in centroids
    np.testing.assert_allclose(centroids[0], [5.0, 6.0, 7.0])
    assert velocities[0] == pytest.approx(3.0)
    assert len(points_dict[0]) == 1


def test_negative_labels_ignored_mixed_mask():
    # mask 中同時有效 cluster 和負值
    moving_mask = np.array([0, -2, 1])
    radar_pc = make_radar_pc([
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [99.0, 99.0, 99.0, 99.0, 99.0],  # 應被忽略
        [2.0, 0.0, 0.0, 2.0, 1.0],
    ])
    centroids, _, _ = calculate_cluster_centroids(moving_mask, radar_pc)

    assert set(centroids.keys()) == {0, 1}
    np.testing.assert_allclose(centroids[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(centroids[1], [2.0, 0.0, 0.0])


# ── build_detections ─────────────────────────────────────────────────────────

def test_build_detections_shape():
    centroids = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([4.0, 5.0, 6.0])}
    velocities = {0: -1.0, 1: 2.0}
    det = build_detections(centroids, velocities)

    assert det.shape == (2, 5)


def test_build_detections_values():
    centroids = {0: np.array([1.0, 2.0, 3.0])}
    velocities = {0: -5.0}
    det = build_detections(centroids, velocities)

    np.testing.assert_allclose(det[0, :3], [1.0, 2.0, 3.0])
    assert det[0, 3] == pytest.approx(0.0)   # cluster_id
    assert det[0, 4] == pytest.approx(-5.0)  # range_rate


def test_build_detections_cluster_id_preserved():
    centroids = {7: np.array([0.0, 0.0, 0.0])}
    velocities = {7: 0.0}
    det = build_detections(centroids, velocities)

    assert int(det[0, 3]) == 7


def test_build_detections_empty_input():
    det = build_detections({}, {})

    assert det.shape == (0, 5)


# ── range_rate_average ────────────────────────────────────────────────────────

def test_range_rate_average():
    moving_mask = np.array([0, 0, 0])
    radar_pc = make_radar_pc([
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 0.0],
    ])
    _, velocities, _ = calculate_cluster_centroids(moving_mask, radar_pc)

    assert velocities[0] == pytest.approx(3.0)
