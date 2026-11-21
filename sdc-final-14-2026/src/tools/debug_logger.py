"""
Per-frame JSON Lines debug logger for the MOT tracker.

Each line is a self-contained JSON object with the full tracking state at one frame:

  {
    "frame": int,
    "tracks": [
      {
        "track_id": int,
        "state":    "tentative" | "confirmed",
        "hits":     int,
        "age":      int,
        "pred_pos": [x, y],         -- KF state after predict(), before correct()
        "velocity": [vx, vy],
        "cov_diag": [p00, p11, p22, p33],
        "last_range_rate": float    -- det[4] stored on the track
      }, ...
    ],
    "detections": [
      {
        "det_idx":    int,
        "x":          float,
        "y":          float,
        "range_rate": float,
        "cluster_id": int
      }, ...
    ],
    "valid_pairs":       [[track_idx, det_idx], ...],
    "cost_matrix":       [[float|null, ...], ...],   -- null = inf (gated out)
    "matched":           [[track_idx, det_idx], ...],
    "unmatched_tracks":  [track_idx, ...],
    "unmatched_dets":    [det_idx, ...]
  }

Usage
-----
  from tools.debug_logger import DebugLogger

  logger = DebugLogger("debug.jsonl")
  # inside tracker update loop:
  logger.log_frame(frame_idx, tracks, detections, valid_pairs, cost,
                   matched, unmatched_dets, unmatched_tracks)
  logger.close()

  # or as context manager:
  with DebugLogger("debug.jsonl") as logger:
      ...
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _finite(v: float) -> float | None:
    """Convert inf / nan to None for JSON serialisation."""
    if math.isinf(v) or math.isnan(v):
        return None
    return float(v)


class DebugLogger:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "DebugLogger":
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    # ── main API ──────────────────────────────────────────────────────────────

    def log_frame(
        self,
        frame_idx: int,
        tracks: list,          # list[Track], predicted but not yet corrected
        detections,            # np.ndarray shape (N, 5)
        valid_pairs: list,     # list of (track_idx, det_idx)
        cost,                  # np.ndarray shape (T, D) — may contain inf
        matched: list,         # list of (track_idx, det_idx)
        unmatched_dets: list,
        unmatched_tracks: list,
    ):
        record: dict[str, Any] = {
            "frame": int(frame_idx),
            "tracks": self._encode_tracks(tracks),
            "detections": self._encode_detections(detections),
            "valid_pairs": [[int(ti), int(di)] for ti, di in valid_pairs],
            "cost_matrix": self._encode_cost(cost),
            "matched": [[int(ti), int(di)] for ti, di in matched],
            "unmatched_tracks": [int(i) for i in unmatched_tracks],
            "unmatched_dets": [int(i) for i in unmatched_dets],
        }
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_tracks(tracks: list) -> list[dict]:
        out = []
        for t in tracks:
            # KalmanFilter exposes full 4D vector as .x and 4x4 matrix as .P.
            # Fall back to the 2D .state/.covariance for non-KF filters.
            full_x = getattr(t.filter, "x", None)
            full_P = getattr(t.filter, "P", None)
            if full_x is not None and len(full_x) >= 4:
                pred_pos  = [float(full_x[0]), float(full_x[1])]
                velocity  = [float(full_x[2]), float(full_x[3])]
                cov_diag  = [float(full_P[i, i]) for i in range(4)]
            else:
                s = t.filter.state
                c = t.filter.covariance
                pred_pos  = [float(s[0]), float(s[1])]
                velocity  = [None, None]
                cov_diag  = [float(c[0, 0]), float(c[1, 1]), None, None]
            out.append({
                "track_id":        int(t.track_id),
                "state":           t.state,
                "hits":            int(t.hits),
                "age":             int(t.age),
                "pred_pos":        pred_pos,
                "velocity":        velocity,
                "cov_diag":        cov_diag,
                "last_range_rate": float(t.det[4]),
            })
        return out

    @staticmethod
    def _encode_detections(detections) -> list[dict]:
        out = []
        for idx, d in enumerate(detections):
            out.append({
                "det_idx":    idx,
                "x":          float(d[0]),
                "y":          float(d[1]),
                "range_rate": float(d[4]),
                "cluster_id": int(d[3]),
                "n_points":   int(d[5]) if len(d) > 5 else None,
            })
        return out

    @staticmethod
    def _encode_cost(cost) -> list[list]:
        if cost is None or cost.size == 0:
            return []
        rows = []
        for row in cost:
            rows.append([_finite(v) for v in row])
        return rows
