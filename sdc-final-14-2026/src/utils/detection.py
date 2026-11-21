from __future__ import annotations

import numpy as np


def calculate_cluster_centroids(
    moving_mask: np.ndarray,
    moving_radar_pc: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, np.ndarray]]:
    """
    Calculate cluster-level detections from moving radar points.

    Inputs:
        moving_mask:
            cluster ids for moving radar points

        moving_radar_pc:
            moving radar points with columns:
            [x, y, z, range_rate, rcs]

    Returns:
        cluster_centroids:
            dict[int, np.ndarray], cluster_id -> [x, y, z]

        cluster_velocities:
            dict[int, float], cluster_id -> mean range_rate

        cluster_points_dict:
            dict[int, np.ndarray], cluster_id -> points
    """
    cluster_centroids: dict[int, np.ndarray] = {}
    cluster_velocities: dict[int, float] = {}
    cluster_points_dict: dict[int, np.ndarray] = {}

    for cluster_id in np.unique(moving_mask):
        if cluster_id < 0:
            continue

        points = moving_radar_pc[moving_mask == cluster_id]
        cluster_centroids[cluster_id] = points[:, :3].mean(axis=0)
        cluster_velocities[cluster_id] = float(points[:, 3].mean())
        cluster_points_dict[cluster_id] = points

    return cluster_centroids, cluster_velocities, cluster_points_dict


def build_detections(
    cluster_centroids: dict[int, np.ndarray],
    cluster_velocities: dict[int, float],
    cluster_points_dict: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Build detection array for Tracker.update().

    Expected detection format:
        [x, y, z, cluster_id, range_rate, n_points]
    """
    detections = []

    for cluster_id, centroid in cluster_centroids.items():
        x, y, z = centroid
        range_rate = cluster_velocities[cluster_id]
        n_pts = len(cluster_points_dict[cluster_id]) if cluster_points_dict else 0
        detections.append([x, y, z, cluster_id, range_rate, n_pts])

    if len(detections) == 0:
        return np.empty((0, 6), dtype=float)

    return np.asarray(detections, dtype=float)
