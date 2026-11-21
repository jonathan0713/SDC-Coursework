from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt

from utils.association import build_cost_matrix, gate, hungarian_assignment
from utils.filter import Filter, IdentityFilter, KalmanFilter


MAX_TRAJ = 5

TrackState = Literal["active", "deleted"]


@dataclass
class Track:
    filter: Filter
    track_id: int
    det: np.ndarray           # [x, y, z, cluster_id, range_rate]
    hits: int = 1
    age: int = 0
    state: TrackState = "active"
    prev_states: deque = field(default_factory=lambda: deque(maxlen=MAX_TRAJ))
    camera_frozen: int = 0    # consecutive camera-gate-frozen frames (debug/stats)
    last_pred: "np.ndarray | None" = field(default=None)  # [x,y] after predict(), before correct()

    @property
    def mean(self) -> np.ndarray:
        return self.filter.state


class Tracker:
    def __init__(
        self,
        max_age: int = 5,
        max_distance: float = 3.0,
        max_euclidean: float = 10.0,
        range_rate_sigma: float = 2.0,
        y_cost_weight: float = 0.0,
        rr_vel_gate: float = 0.0,
        rr_spawn_min_pts: int = 0,
        filter_factory: Callable[[np.ndarray], Filter] | None = None,
        debug_logger=None,
        camera_gate=None,
    ):
        self.max_age = max_age
        self.max_distance = max_distance
        self.max_euclidean = max_euclidean
        self.range_rate_sigma = range_rate_sigma
        self.y_cost_weight = y_cost_weight
        self.rr_vel_gate = rr_vel_gate
        self.rr_spawn_min_pts = rr_spawn_min_pts
        self.next_track_id = 0
        self.tracks: list[Track] = []
        self._filter_factory: Callable[[np.ndarray, np.ndarray | None], Filter] = (
            filter_factory if filter_factory is not None
            else lambda pos, vel=None: KalmanFilter(pos, initial_vel=vel)
        )
        self.debug_logger = debug_logger
        self.camera_gate = camera_gate   # CameraGate | None
        self._frame_idx = 0
        self.last_matched_track_ids: set[int] = set()

    @classmethod
    def from_config(cls, config) -> "Tracker":
        """
        Construct from a TrackerConfig-like object (duck-typed, no hard import).
        Expected shape: config.lifecycle, config.filter, config.gate, config.cost

        config.filter.type controls the filter backend:
          "kalman"   — Constant-velocity Kalman Filter (default)
          "identity" — No-op filter: returns raw detection position unchanged.
                       Useful to isolate tracker logic from KF smoothing.
        """
        filter_type = getattr(config.filter, "type", "kalman")

        if filter_type == "identity":
            def _make_filter(pos: np.ndarray, vel: np.ndarray | None = None) -> Filter:
                return IdentityFilter(pos)
        else:
            def _make_filter(pos: np.ndarray, vel: np.ndarray | None = None) -> Filter:
                return KalmanFilter(
                    pos,
                    initial_vel=vel,
                    dt=config.filter.dt,
                    q_var=config.filter.q_var,
                    r_var=config.filter.r_var,
                    p_var=config.filter.p_var,
                    r_var_vel=config.filter.r_var_vel,
                    q_var_lat=config.filter.q_var_lat,
                )

        return cls(
            max_age=config.lifecycle.max_age,
            max_distance=config.gate.max_distance,
            max_euclidean=config.gate.max_euclidean,
            range_rate_sigma=config.cost.range_rate_sigma,
            y_cost_weight=config.cost.y_cost_weight,
            rr_vel_gate=config.gate.rr_vel_gate,
            rr_spawn_min_pts=config.gate.rr_spawn_min_pts,
            filter_factory=_make_filter,
        )

    def update(
        self,
        detections: np.ndarray,
        frame_img: "np.ndarray | None" = None,
        frame_idx: int | None = None,
    ) -> np.ndarray:
        """
        Update the tracker with new detections.

        Expected detection format:  [x, y, z, cluster_id, range_rate]
        Returned track format:      [x, y, z, cluster_id, track_id]

        Parameters
        ----------
        detections : (N, 5) array
        frame_img  : (H, W, 3) RGB image for camera-gated age freeze (optional)
        frame_idx  : current frame index (required when camera_gate is active)
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=float)

        # [pre] YOLO inference for this frame — must happen before 05a/05b
        # so that register_track() has fresh bounding boxes available.
        _gate_active = frame_img is not None and self.camera_gate is not None
        if _gate_active:
            _fidx = frame_idx if frame_idx is not None else self._frame_idx
            self.camera_gate.set_frame(frame_img, _fidx)

        # [02] Predict: advance every track's filter one step
        for track in self.tracks:
            track.filter.predict()
            track.last_pred = track.filter.state.copy()   # capture before correct()

        if self.tracks:
            predicted_positions = np.array([t.filter.state for t in self.tracks])
            predicted_covs = [t.filter.covariance for t in self.tracks]
        else:
            predicted_positions = np.empty((0, 2))
            predicted_covs = []

        detection_positions = detections[:, :2] if len(detections) > 0 else np.empty((0, 2))

        # [03a] Gate: enumerate valid (track, det) pairs by Mahalanobis distance
        valid_pairs = gate(
            predicted_positions,
            detection_positions,
            predicted_covs,
            self.max_distance,
            max_euclidean=self.max_euclidean,
        )

        # [03b] Build cost matrix (inf for gated-out pairs).
        # Both sides use raw range rates (relative to radar); no ego compensation.
        track_rvs = np.array([t.det[4] for t in self.tracks]) if self.tracks else np.empty(0)
        det_rvs = detections[:, 4] if len(detections) > 0 else np.empty(0)

        cost = build_cost_matrix(
            valid_pairs, predicted_positions, detection_positions, predicted_covs,
            track_range_rates=track_rvs, det_range_rates=det_rvs,
            range_rate_sigma=self.range_rate_sigma,
            y_cost_weight=self.y_cost_weight,
        )

        # [04] Hungarian assignment
        matched, unmatched_detections, unmatched_tracks = hungarian_assignment(cost)

        # [04b] Optional debug logging (captured before correct() mutates state)
        if self.debug_logger is not None:
            self.debug_logger.log_frame(
                self._frame_idx,
                self.tracks,
                detections,
                valid_pairs,
                cost,
                matched,
                list(unmatched_detections),
                list(unmatched_tracks),
            )

        # [04-②] Camera-mask rescue: Hungarian in image space.
        # Track representation: SAM2 mask distance map (distance_transform_edt).
        # Cluster representation: projected to image space via _project().
        # Gate: pixel distance from projection to nearest mask edge ≤ cam_rescue_gate_px.
        # Cost: that same pixel distance (0 = inside mask, increases outward).
        # Uses _current_masks from the previous frame — valid because objects don't
        # teleport between frames.
        if _gate_active:
            frozen_items: list[tuple[int, Track]] = [
                (i, self.tracks[i])
                for i in unmatched_tracks
                if self.tracks[i].camera_frozen > 0
            ]
            if frozen_items and unmatched_detections:
                n_f = len(frozen_items)
                n_d = len(unmatched_detections)
                rescue_cost = np.full((n_f, n_d), np.inf)
                gate_px = self.camera_gate._rescue_gate_px

                # Pre-project all unmatched detections once
                det_uvs: list[tuple[int, int] | None] = []
                for det_idx in unmatched_detections:
                    det = detections[det_idx]
                    uv = self.camera_gate._project(
                        np.array([det[0], det[1], float(det[2])])
                    )
                    det_uvs.append(
                        (int(round(uv[0])), int(round(uv[1]))) if uv is not None else None
                    )

                for fi, (_, track) in enumerate(frozen_items):
                    mask = self.camera_gate._current_masks.get(track.track_id)
                    if mask is None:
                        continue
                    dist_map = distance_transform_edt(~mask)
                    h, w = mask.shape
                    for di, uv in enumerate(det_uvs):
                        if uv is None:
                            continue
                        u_px, v_px = uv
                        if not (0 <= v_px < h and 0 <= u_px < w):
                            continue
                        d = dist_map[v_px, u_px]
                        if d <= gate_px:
                            rescue_cost[fi, di] = d

                cam_matched, _, _ = hungarian_assignment(rescue_cost)
                if cam_matched:
                    rescued_f = {fi for fi, _ in cam_matched}
                    rescued_d = {di for _, di in cam_matched}
                    for fi, di in cam_matched:
                        matched.append((frozen_items[fi][0], unmatched_detections[di]))
                    rescued_tids = {frozen_items[fi][0] for fi in rescued_f}
                    unmatched_tracks = [t for t in unmatched_tracks if t not in rescued_tids]
                    unmatched_detections = [
                        det_idx for pos, det_idx in enumerate(unmatched_detections)
                        if pos not in rescued_d
                    ]

        # Record which track_ids were matched this frame (includes camera-rescued matches)
        self.last_matched_track_ids = {
            self.tracks[ti].track_id for ti, _ in matched
        }

        # [05a] Matched: correct filter with assigned detection (Hungarian + camera-rescued)
        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]

            # Soft rr gate: if |rr_det − rr_predicted| > rr_vel_gate, drop rr from the
            # observation so the velocity state is not updated by a potentially wrong
            # Doppler (e.g. oncoming vehicle, noisy sparse cluster).
            # rr_predicted is projected from the KF velocity, which is smoother than
            # the last stored range rate.
            if self.rr_vel_gate > 0:
                _kf_x = getattr(track.filter, 'x', None)
                if _kf_x is not None and len(_kf_x) >= 4:
                    _pos = _kf_x[:2]
                    _r = float(np.linalg.norm(_pos))
                    _rr_pred = (
                        float(_kf_x[2] * _pos[0] + _kf_x[3] * _pos[1]) / _r
                        if _r > 1e-6 else 0.0
                    )
                    _use_rr = abs(detection[4] - _rr_pred) <= self.rr_vel_gate
                else:
                    _use_rr = True
            else:
                _use_rr = True

            # Augmented observation [x, y, ṙ] or position-only [x, y].
            z = (
                np.array([detection[0], detection[1], detection[4]])
                if _use_rr else
                np.array([detection[0], detection[1]])
            )
            track.filter.correct(z)
            track.hits += 1
            track.age = 0
            track.camera_frozen = 0   # radar match resets camera freeze counter
            track.det = detection
            track.prev_states.append(track.filter.state.copy())

            if _gate_active:
                p3d_retry = np.array([
                    track.filter.state[0], track.filter.state[1], float(detection[2])
                ])
                if track.track_id not in self.camera_gate._registered:
                    # Track first appeared outside FOV; retry each radar match.
                    self.camera_gate.register_track(track.track_id, _fidx, p3d_retry)
                elif self.camera_gate._current_masks.get(track.track_id) is None:
                    # Mask lost (camera occluded); re-init while radar fix is fresh.
                    self.camera_gate.register_track(track.track_id, _fidx, p3d_retry, force=True)

        # [05b] Unmatched detections → spawn new tracks
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            det_pos = detection[:2]
            det_rr = float(detection[4])
            r = float(np.linalg.norm(det_pos))
            # Sparse-cluster rr guard: when the cluster is thin (few radar points),
            # Doppler can spike to implausible values. If the cluster is below
            # rr_spawn_min_pts AND |rr| exceeds rr_vel_gate, the rr is treated as
            # unreliable and velocity is initialised to zero instead.
            _n_pts = int(detection[5]) if len(detection) > 5 else 0
            _sparse = self.rr_spawn_min_pts > 0 and _n_pts < self.rr_spawn_min_pts
            _rr_noisy = self.rr_vel_gate > 0 and abs(det_rr) > self.rr_vel_gate
            v_init = (
                np.zeros(2) if _sparse and _rr_noisy
                else ((det_rr / r) * det_pos if r > 1e-6 else np.zeros(2))
            )
            self.tracks.append(
                Track(
                    filter=self._filter_factory(det_pos, v_init),
                    track_id=self.next_track_id,
                    det=detection,
                )
            )
            # Try to register the new track with SAM2 immediately via YOLO bbox.
            if _gate_active:
                p3d_new = np.array([det_pos[0], det_pos[1], float(detection[2])])
                self.camera_gate.register_track(self.next_track_id, _fidx, p3d_new)
            self.next_track_id += 1

        # [05c pre] Propagate SAM2 one step to the current frame.
        # Must happen after all register_track() calls for this frame.
        if _gate_active:
            self.camera_gate.propagate(_fidx)
            # Drift guard: remove masks where the KF-projected position has drifted
            # outside the mask region.  Removed masks trigger force re-register on
            # the next radar match (05a elif branch).
            _reg = self.camera_gate._registered
            _drift_p3ds = {
                t.track_id: np.array([
                    t.filter.state[0], t.filter.state[1], float(t.det[2])
                ])
                for t in self.tracks
                if t.track_id in _reg
            }
            self.camera_gate.validate_masks(_drift_p3ds)

        # [05c] Unmatched tracks → camera gate check, then age / freeze
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            p3d = np.array([
                track.filter.state[0],
                track.filter.state[1],
                float(track.det[2]),
            ])
            if _gate_active and self.camera_gate.is_visible(p3d, track_id=track.track_id):
                track.camera_frozen += 1
                track.prev_states.append(track.filter.state.copy())
                continue   # age frozen this frame

            track.camera_frozen = 0
            track.age += 1
            if track.age > self.max_age:
                track.state = "deleted"

        self.tracks = [t for t in self.tracks if t.state != "deleted"]

        # Collect and return only tracks matched this frame (age == 0)
        confirmed_tracks = []
        for track in self.tracks:
            if track.age != 0:
                continue
            pos = track.filter.state
            det = track.det
            confirmed_tracks.append([
                float(pos[0]),
                float(pos[1]),
                float(det[2]),
                int(det[3]),
                int(track.track_id),
            ])

        self._frame_idx += 1

        if not confirmed_tracks:
            return np.empty((0, 5), dtype=float)
        return np.asarray(confirmed_tracks, dtype=float)
