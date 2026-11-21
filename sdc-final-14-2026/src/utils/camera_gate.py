"""
Camera-gated age freeze for the MOT tracker.

Hybrid detection:
  1. YOLO  — per-frame bounding box detection (fast, class-aware)
  2. SAM2 VideoPredictor — persistent per-object segmentation across frames

When a Track first appears (05b new track), we attempt to register it with the
SAM2 VideoPredictor using the YOLO bounding box as the initial prompt.
On each subsequent frame, propagate_in_video advances the segmentation by one
step using SAM2's temporal memory — no fresh per-frame point prompt needed.

For unmatched tracks (05c):
  - If registered with SAM2: check whether the propagated mask is still valid.
  - If never registered (YOLO had no box at creation): fall back to YOLO hit-check.

Usage pattern
-------------
    gate = CameraGate(config, seq_dir)         # once at sequence start
    for frame_idx, frame in frames:
        gate.set_frame(img_rgb, frame_idx)     # YOLO inference (eager)
        ...tracker update (05a / 05b)...
        gate.register_track(tid, frame_idx, p3d)  # called from 05b for new tracks
        gate.propagate(frame_idx)              # advance SAM2 one step
        for track in unmatched_tracks:
            if gate.is_visible(p3d, track_id=tid):
                freeze_age(track)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from runner.configs import CameraGateConfig

_SAM2_CKPT_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "sam2_checkpoints" / "sam2_hiera_tiny.pt"
)


class CameraGate:
    """
    YOLO + SAM2-VideoPredictor hybrid camera confirmation for track age freezing.

    Parameters
    ----------
    config  : CameraGateConfig
    seq_dir : Path  — sequence root (e.g. data/seq_4/); image/ sub-dir is required.
    """

    def __init__(self, config: "CameraGateConfig", seq_dir: Path) -> None:
        from ultralytics import YOLO
        from sam2.build_sam import build_sam2_video_predictor

        # ── Extrinsics (K, R, t) ──────────────────────────────────────────────
        if config.extrinsics:
            ext_path = Path(config.extrinsics)
        else:
            from utils.paths import OUTPUT_ROOT
            ext_path = OUTPUT_ROOT / f"{seq_dir.name}_extrinsics.npz"
        if not ext_path.exists():
            raise FileNotFoundError(
                f"[CameraGate] extrinsics not found: {ext_path}\n"
                f"  Run annotate_extrinsics_sam2.py first, or set config.camera_gate.extrinsics."
            )
        ext = np.load(str(ext_path))
        self.K = ext["K"].astype(float)
        self.R = ext["R"].astype(float)
        self.t = ext["t"].astype(float)
        # ── Config ────────────────────────────────────────────────────────────
        self._proj_margin    = int(config.proj_margin_px)
        self._yolo_conf      = float(config.yolo_conf)
        self._yolo_classes   = list(config.yolo_classes)
        self._mask_min       = int(config.sam2_mask_min_px)
        self._mask_max       = int(config.sam2_mask_max_px)
        self._proj_z         = float(config.proj_z_height)
        self._rescue_gate_px = int(config.cam_rescue_gate_px)
        self._mask_drift_px  = int(config.cam_mask_drift_px)

        # ── YOLO (first-stage confirmation + registration bbox) ───────────────
        self._yolo = YOLO(str(config.yolo_model))

        # ── SAM2 VideoPredictor ───────────────────────────────────────────────
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = Path(config.sam2_ckpt) if config.sam2_ckpt else _SAM2_CKPT_DEFAULT
        self._sam2 = build_sam2_video_predictor(
            config.sam2_cfg, str(ckpt), device=self._device,
        )

        # Initialise SAM2 with the full image sequence (all frames pre-loaded to CPU).
        image_dir = seq_dir / "image"
        if not image_dir.exists():
            raise FileNotFoundError(
                f"[CameraGate] image directory not found: {image_dir}\n"
                f"  SAM2 VideoPredictor requires the full frame sequence."
            )
        with torch.inference_mode():
            self._state = self._sam2.init_state(
                str(image_dir),
                offload_video_to_cpu=True,   # keep GPU free; features computed on demand
            )

        # ── Per-frame state ───────────────────────────────────────────────────
        self._frame_idx: int = -1
        self._frame_rgb: np.ndarray | None = None
        self._img_h: int = 0
        self._img_w: int = 0
        self._yolo_boxes = None

        # ── SAM2 tracking state ───────────────────────────────────────────────
        # track_id → bool : whether this track has been registered with SAM2
        self._registered: set[int] = set()
        # track_id → YOLO class id recorded at first successful registration
        self._track_cls: dict[int, int] = {}
        # track_id → mask (bool H×W) for the current frame (populated by propagate())
        self._current_masks: dict[int, np.ndarray] = {}

        print(f"[CameraGate] extrinsics = {ext_path}")
        print(f"  K  f={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}  cy={self.K[1,2]:.1f}")
        print(f"  YOLO  model={config.yolo_model}  conf={self._yolo_conf}  "
              f"classes={self._yolo_classes}")
        print(f"  SAM2  VideoPredictor  ckpt={ckpt.name}  "
              f"mask_px=[{self._mask_min}, {self._mask_max}]")
        print(f"  image_dir = {image_dir}  (frames: {self._state['num_frames']})")

    # ── Per-frame interface ───────────────────────────────────────────────────

    def set_frame(self, img_rgb: np.ndarray, frame_idx: int) -> None:
        """
        Prepare for a new video frame.
        Must be called once per frame BEFORE register_track / propagate / is_visible.
        Runs YOLO eagerly so the result is shared across all track checks this frame.
        """
        self._frame_rgb = img_rgb
        self._frame_idx = frame_idx
        self._img_h, self._img_w = img_rgb.shape[:2]
        results = self._yolo(
            img_rgb, classes=self._yolo_classes,
            conf=self._yolo_conf, verbose=False,
        )
        self._yolo_boxes = results[0].boxes

    def register_track(
        self,
        track_id: int,
        frame_idx: int,
        p3d: np.ndarray,
        force: bool = False,
    ) -> bool:
        """
        Register a track with the SAM2 VideoPredictor using the best YOLO bbox.

        Called from 05b (new track) and optionally from 05a (re-condition after
        a long camera-frozen period).

        Parameters
        ----------
        track_id  : Track.track_id
        frame_idx : current frame index (must match SAM2's frame ordering)
        p3d       : 3D position [x, y, z] in radar/ego frame
        force     : if True, re-condition even if already registered

        Returns True if the track was successfully registered.
        """
        if track_id in self._registered and not force:
            return True

        uv = self._project(p3d)
        if uv is None:
            return False

        # On force re-register, restrict to the class recorded at first registration.
        required_cls = self._track_cls.get(track_id) if force else None

        # Occlusion guard (initial registration only): if the projection falls
        # inside an existing track's mask, the new track is occluded.  Carve out
        # all existing masks and re-run YOLO so the partially-visible bbox can be
        # found.  SAM2 still uses original pre-computed frame features; the carved
        # bbox guides attention to the correct (partially visible) region.
        if not force and self._current_masks:
            if any(
                self._mask_contains_projection(m, uv[0], uv[1], margin_px=0)
                for m in self._current_masks.values()
            ):
                return self._register_with_carveout(track_id, frame_idx, uv)

        result = self._best_yolo_box(uv[0], uv[1], required_cls=required_cls)
        if result is None:
            return False
        box, cls = result

        with torch.inference_mode():
            self._sam2.add_new_points_or_box(
                self._state,
                frame_idx,
                track_id,
                box=np.array(box, dtype=np.float32),
            )
        self._registered.add(track_id)
        if track_id not in self._track_cls:
            self._track_cls[track_id] = cls
        return True

    def propagate(self, frame_idx: int) -> None:
        """
        Advance SAM2 one step to frame_idx and cache the resulting masks.

        Must be called after all register_track() calls for this frame and
        before any is_visible() checks for this frame.
        """
        if not self._registered:
            return

        self._current_masks = {}
        with torch.inference_mode():
            for fid, obj_ids, mask_logits in self._sam2.propagate_in_video(
                self._state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                if fid != frame_idx:
                    continue
                for oid, logit in zip(obj_ids, mask_logits):
                    # logit: (1, H, W)  positive = foreground
                    mask = (logit[0].cpu().numpy() > 0.0)
                    area = int(mask.sum())
                    if self._mask_min <= area <= self._mask_max:
                        self._current_masks[int(oid)] = mask

    def is_visible(self, p3d: np.ndarray, track_id: int | None = None) -> bool:
        """
        Returns True if the camera confirms a vehicle near the 3D point.

        For SAM2-registered tracks: checks whether propagate() produced a valid mask.
        For unregistered tracks: falls back to a YOLO bounding-box hit-check.

        Parameters
        ----------
        p3d      : (3,) [x, y, z] in radar/ego frame
        track_id : Track.track_id (pass for SAM2 lookup; None triggers YOLO-only)
        """
        if track_id is not None and track_id in self._registered:
            return track_id in self._current_masks

        # Fallback: unregistered track → check YOLO bbox
        uv = self._project(p3d)
        if uv is None:
            return False
        return self._yolo_hit(uv[0], uv[1])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _project(self, p3d: np.ndarray) -> tuple[float, float] | None:
        p = np.array([p3d[0], p3d[1], self._proj_z])
        p_cam = self.R @ p + self.t
        if p_cam[2] < 0.1:
            return None
        proj = self.K @ p_cam
        u, v = proj[0] / proj[2], proj[1] / proj[2]
        if 0.0 <= u < self._img_w and 0.0 <= v < self._img_h:
            return float(u), float(v)
        return None

    def _yolo_hit(self, u: float, v: float) -> bool:
        if self._yolo_boxes is None:
            return False
        m = self._proj_margin
        for b in self._yolo_boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if (x1 - m) <= u <= (x2 + m) and (y1 - m) <= v <= (y2 + m):
                return True
        return False

    def _best_yolo_box(
        self, u: float, v: float, required_cls: int | None = None
    ) -> tuple[tuple, int] | None:
        """Return (box_coords, cls) for the YOLO box closest to (u, v).

        required_cls: if set, only boxes with this COCO class id are considered.
        """
        if self._yolo_boxes is None:
            return None
        m = self._proj_margin
        best_box, best_cls, best_dist = None, None, float("inf")
        for b in self._yolo_boxes:
            cls = int(b.cls[0])
            if required_cls is not None and cls != required_cls:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if (x1 - m) <= u <= (x2 + m) and (y1 - m) <= v <= (y2 + m):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = (u - cx) ** 2 + (v - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_box = (x1, y1, x2, y2)
                    best_cls = cls
        if best_box is None:
            return None
        return best_box, best_cls

    def validate_masks(self, track_p3ds: dict[int, np.ndarray]) -> list[int]:
        """
        Remove masks whose SAM2 region no longer aligns with the KF position.

        For each registered track with a current mask, project its KF position
        to image space.  If the projected point is further than _mask_drift_px
        pixels from the nearest mask pixel, the mask has drifted and is removed.
        Removal triggers force re-register on the next radar match (05a elif
        branch), so the mask is corrected without any extra logic in the caller.

        Parameters
        ----------
        track_p3ds : {track_id: [x, y, z]} — KF state for every registered track

        Returns list of invalidated track_ids (for debugging).
        """
        invalidated: list[int] = []
        for track_id, p3d in track_p3ds.items():
            mask = self._current_masks.get(track_id)
            if mask is None:
                continue
            uv = self._project(p3d)
            if uv is None:
                continue
            if not self._mask_contains_projection(mask, uv[0], uv[1], self._mask_drift_px):
                del self._current_masks[track_id]
                invalidated.append(track_id)
        return invalidated

    def _mask_contains_projection(
        self, mask: np.ndarray, u: float, v: float, margin_px: int
    ) -> bool:
        """True if (u, v) is within the mask or within margin_px pixels of it."""
        u_i, v_i = int(round(u)), int(round(v))
        h, w = mask.shape
        if not (0 <= v_i < h and 0 <= u_i < w):
            return False
        r0 = max(0, v_i - margin_px)
        r1 = min(h, v_i + margin_px + 1)
        c0 = max(0, u_i - margin_px)
        c1 = min(w, u_i + margin_px + 1)
        return bool(mask[r0:r1, c0:c1].any())

    def _register_with_carveout(
        self,
        track_id: int,
        frame_idx: int,
        uv: tuple[float, float],
    ) -> bool:
        """
        Zero-fill all existing mask regions in a copy of the current frame,
        re-run YOLO on the modified image, and use the best resulting bbox to
        condition SAM2 for the new track.

        This handles the case where a new track's radar projection lands inside
        another track's mask (e.g. the new vehicle is partially occluded by the
        vehicle in front).  The carve-out exposes the visible portion so YOLO
        can find the correct bbox.  SAM2's add_new_points_or_box() still uses
        the pre-computed features of the original (unmodified) frame; the bbox
        prompt guides its attention to the visible region.
        """
        if self._frame_rgb is None:
            return False

        carved = self._frame_rgb.copy()
        for mask in self._current_masks.values():
            carved[mask] = 0

        results = self._yolo(
            carved, classes=self._yolo_classes,
            conf=self._yolo_conf, verbose=False,
        )
        carved_boxes = results[0].boxes
        if carved_boxes is None or len(carved_boxes) == 0:
            return False

        u, v = uv
        m = self._proj_margin
        best_box, best_cls, best_dist = None, None, float("inf")
        for b in carved_boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if (x1 - m) <= u <= (x2 + m) and (y1 - m) <= v <= (y2 + m):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = (u - cx) ** 2 + (v - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_box = (x1, y1, x2, y2)
                    best_cls = cls

        if best_box is None:
            return False

        with torch.inference_mode():
            self._sam2.add_new_points_or_box(
                self._state, frame_idx, track_id,
                box=np.array(best_box, dtype=np.float32),
            )
        self._registered.add(track_id)
        if track_id not in self._track_cls:
            self._track_cls[track_id] = best_cls
        return True

    # ── Visualization helper ──────────────────────────────────────────────────

    def render_viz(
        self,
        all_p3ds: "list[tuple[int, np.ndarray]]",
        confirmed_p3ds: "list[tuple[int, np.ndarray]] | None" = None,
        unmatched_p3ds: "list[tuple[int, np.ndarray]] | None" = None,
    ) -> dict:
        """
        Collect visualization data for the current frame.
        Uses pre-computed SAM2 masks from the last propagate() call.

        Parameters
        ----------
        all_p3ds       : ALL active tracks (confirmed + frozen); masks shown for any
                         track registered with SAM2 that has a valid mask this frame.
        confirmed_p3ds : radar-confirmed tracks (optional, for projection dots).
        unmatched_p3ds : tracks NOT matched to radar this frame (drawn as squares).

        Returns
        -------
        {
          "img"        : np.ndarray RGB,
          "boxes"      : [(x1,y1,x2,y2,cls,conf), ...],
          "frozen"     : [(tid, u, v, mask|None), ...],  mask entry for every track
          "projected"  : [(tid, u, v), ...],
          "unmatched"  : [(tid, u, v), ...],
        }
        """
        boxes: list = []
        if self._yolo_boxes is not None:
            for b in self._yolo_boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])))

        frozen_out: list = []
        for tid, p3d in all_p3ds:
            uv = self._project(p3d)
            mask = self._current_masks.get(tid)   # None if not registered / SAM2 missed
            if uv is not None:
                frozen_out.append((tid, uv[0], uv[1], mask))
            elif mask is not None:
                frozen_out.append((tid, None, None, mask))

        projected: list = []
        if confirmed_p3ds:
            for tid, p3d in confirmed_p3ds:
                uv = self._project(p3d)
                if uv:
                    projected.append((tid, uv[0], uv[1]))

        unmatched_out: list = []
        if unmatched_p3ds:
            for tid, p3d in unmatched_p3ds:
                uv = self._project(p3d)
                if uv:
                    unmatched_out.append((tid, uv[0], uv[1]))

        return {
            "img":       self._frame_rgb,
            "boxes":     boxes,
            "frozen":    frozen_out,
            "projected": projected,
            "unmatched": unmatched_out,
        }
