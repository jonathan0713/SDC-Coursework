from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance between point x and distribution mean.

    Args:
        x:       observation vector, shape (D,)
        mean:    distribution mean, shape (D,)
        cov_inv: inverse covariance matrix, shape (D, D)

    Returns:
        Scalar Mahalanobis distance.
    """
    diff = x - mean
    return float(np.sqrt(diff @ cov_inv @ diff))


def nearest_neighbor_match(
    track_positions: np.ndarray,
    detection_positions: np.ndarray,
    cov_inv: np.ndarray,
    max_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedy nearest-neighbor matching with Mahalanobis gating.

    Repeatedly picks the globally smallest Mahalanobis distance in the cost
    matrix and assigns that track-detection pair, until no valid pair remains.
    Pairs whose distance exceeds max_distance are gated out before matching.

    Args:
        track_positions:     (M, D) array of track mean positions
        detection_positions: (N, D) array of detection positions
        cov_inv:             (D, D) inverse covariance used for gating
        max_distance:        Mahalanobis distance threshold for gating

    Returns:
        matched:              list of (track_idx, detection_idx) pairs
        unmatched_detections: list of detection indices with no assigned track
        unmatched_tracks:     list of track indices with no assigned detection
    """
    n_tracks = len(track_positions)
    n_dets = len(detection_positions)

    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_dets)), list(range(n_tracks))

    # Build (M, N) cost matrix; gate invalid pairs with inf
    cost = np.full((n_tracks, n_dets), np.inf)
    for i, t_pos in enumerate(track_positions):
        for j, d_pos in enumerate(detection_positions):
            dist = mahalanobis_distance(d_pos, t_pos, cov_inv)
            if dist <= max_distance:
                cost[i, j] = dist

    matched: list[tuple[int, int]] = []
    available_tracks = set(range(n_tracks))
    available_dets = set(range(n_dets))

    while available_tracks and available_dets:
        track_list = list(available_tracks)
        det_list = list(available_dets)

        sub_cost = cost[np.ix_(track_list, det_list)]
        min_val = sub_cost.min()

        if np.isinf(min_val):
            break

        t_sub, d_sub = np.unravel_index(sub_cost.argmin(), sub_cost.shape)
        t_idx = track_list[t_sub]
        d_idx = det_list[d_sub]

        matched.append((t_idx, d_idx))
        available_tracks.remove(t_idx)
        available_dets.remove(d_idx)

    return matched, sorted(available_dets), sorted(available_tracks)


# ── Pipeline-style association (gate → cost matrix → Hungarian) ───────────────

def gate(
    predicted_positions: np.ndarray,
    detection_positions: np.ndarray,
    predicted_covs: list[np.ndarray],
    max_distance: float,
    max_euclidean: float = 10.0,
) -> set[tuple[int, int]]:
    """
    Return all (track_idx, det_idx) pairs that pass both distance gates:
      1. Hard gate: Euclidean distance <= max_euclidean (prevents covariance
         blow-up from making Mahalanobis gating permissive for aged tracks).
      2. Soft gate: Mahalanobis distance <= max_distance, using per-track
         predicted covariance.

    Args:
        predicted_positions: (M, 2) predicted track positions
        detection_positions: (N, 2) detection positions
        predicted_covs:      list of M (2×2) covariance matrices
        max_distance:        Mahalanobis distance threshold
        max_euclidean:       Euclidean distance hard cap (metres)

    Returns:
        valid_pairs: set of (track_idx, det_idx) that pass both gates
    """
    valid_pairs: set[tuple[int, int]] = set()

    for i, (t_pos, cov) in enumerate(zip(predicted_positions, predicted_covs)):
        cov_inv = np.linalg.inv(cov)
        for j, d_pos in enumerate(detection_positions):
            # Hard gate: reject pairs whose Euclidean distance already exceeds
            # the physical plausibility limit, regardless of covariance size.
            if np.linalg.norm(d_pos - t_pos) > max_euclidean:
                continue
            dist = mahalanobis_distance(d_pos, t_pos, cov_inv)
            if dist <= max_distance:
                valid_pairs.add((i, j))

    return valid_pairs


def build_cost_matrix(
    valid_pairs: set[tuple[int, int]],
    predicted_positions: np.ndarray,
    detection_positions: np.ndarray,
    predicted_covs: list[np.ndarray],
    track_range_rates: np.ndarray | None = None,
    det_range_rates: np.ndarray | None = None,
    range_rate_sigma: float = 1.0,
    y_cost_weight: float = 0.0,
) -> np.ndarray:
    """
    Build an (M×N) cost matrix. Only valid_pairs receive a finite cost;
    all other entries are inf.

    Cost = Mahalanobis_position
         + |rr_det - rr_track| / range_rate_sigma   (if range rates provided)
         + |Δy| * y_cost_weight                      (anisotropic lateral penalty)

    The range_rate term disambiguates nearby clusters that move at different
    radial speeds.  The y_cost_weight term adds a fixed lateral penalty that is
    independent of the KF covariance (which can grow when a track is unmatched),
    ensuring lateral separation is consistently penalised regardless of track age.

    Args:
        valid_pairs:         set of (track_idx, det_idx) from gate()
        predicted_positions: (M, 2) predicted track positions
        detection_positions: (N, 2) detection positions
        predicted_covs:      list of M (2×2) covariance matrices
        track_range_rates:   (M,) last observed range_rate per track (m/s)
        det_range_rates:     (N,) range_rate per detection (m/s)
        range_rate_sigma:    normalisation constant for range_rate cost (m/s)
        y_cost_weight:       lateral (y) penalty weight (cost per metre); 0 = disabled.

    Returns:
        cost: (M, N) ndarray
    """
    M = len(predicted_positions)
    N = len(detection_positions)
    cost = np.full((M, N), np.inf)

    use_rr = (
        track_range_rates is not None
        and det_range_rates is not None
        and len(track_range_rates) == M
        and len(det_range_rates) == N
    )

    for i, j in valid_pairs:
        cov_inv = np.linalg.inv(predicted_covs[i])
        pos_cost = mahalanobis_distance(
            detection_positions[j], predicted_positions[i], cov_inv
        )
        rr_cost = (
            abs(float(det_range_rates[j]) - float(track_range_rates[i])) / range_rate_sigma
            if use_rr else 0.0
        )
        y_cost = abs(float(detection_positions[j][1]) - float(predicted_positions[i][1])) * y_cost_weight
        cost[i, j] = pos_cost + rr_cost + y_cost

    return cost


def hungarian_assignment(
    cost_matrix: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Solve the assignment problem with the Hungarian algorithm.
    Pairs with infinite cost are never assigned.

    Args:
        cost_matrix: (M, N) ndarray, inf marks gated-out pairs

    Returns:
        matched:              list of (track_idx, det_idx)
        unmatched_detections: list of det indices with no track
        unmatched_tracks:     list of track indices with no detection
    """
    M, N = cost_matrix.shape

    if M == 0 or N == 0:
        return [], list(range(N)), list(range(M))

    if not np.isfinite(cost_matrix).any():
        return [], list(range(N)), list(range(M))

    # linear_sum_assignment cannot handle inf; replace with a large sentinel.
    # Any assigned pair that maps back to inf in the original is rejected below.
    _INF_SENTINEL = 1e9
    finite_cost = np.where(np.isfinite(cost_matrix), cost_matrix, _INF_SENTINEL)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    matched: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if np.isinf(cost_matrix[r, c]):
            continue
        matched.append((int(r), int(c)))

    matched_tracks = {r for r, _ in matched}
    matched_dets   = {c for _, c in matched}

    unmatched_tracks = sorted(set(range(M)) - matched_tracks)
    unmatched_dets   = sorted(set(range(N)) - matched_dets)

    return matched, unmatched_dets, unmatched_tracks
