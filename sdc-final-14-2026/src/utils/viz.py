from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# Coordinate convention in all radar plot functions:
#   plot-x = -radar_y  (lateral, left positive)
#   plot-y =  radar_x  (forward)

# ── viz constants ─────────────────────────────────────────────────────────────
_CMAP = plt.cm.tab20

_TRACK_DOT_SIZE        = 5.0
_TRACK_DOT_EDGE_COLOR  = "black"
_TRACK_DOT_EDGE_WIDTH  = 1.0

_TRACK_LABEL_FONTSIZE  = 11
_TRACK_LABEL_COLOR     = "black"
_TRACK_LABEL_XOFFSET   = 0.8   # metres lateral
_TRACK_LABEL_YOFFSET   = 0.8   # metres forward

_TRAJ_MAX_POINTS       = 15    # how many past centroids to show per track
_TRAJ_MIN_ALPHA        = 0.0
_TRAJ_MAX_ALPHA        = 0.7
_TRAJ_DOT_MIN_SIZE     = 2.0
_TRAJ_DOT_MAX_SIZE     = 2.0
_TRAJ_LINE_WIDTH       = 0.8
_TRAJ_LINE_ALPHA       = 0.4

_UNMATCHED_MARKER      = "^"   # upward triangle for unmatched clusters
_UNMATCHED_MARKER_SIZE = 8
_UNMATCHED_EDGE_COLOR  = "black"
_UNMATCHED_EDGE_WIDTH  = 1.2
_UNMATCHED_ALPHA       = 0.85

_VIDEO_FPS             = 10
_CLUSTER_DOT_SIZE      = 5

# COCO class names used by the camera gate
_COCO_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}


def _track_color(track_id: int):
    return _CMAP(int(track_id) % 20)


# ── radar plot helpers ────────────────────────────────────────────────────────

def plot_static_points(ax: plt.Axes, static_radar_pc: np.ndarray) -> None:
    if len(static_radar_pc) == 0:
        return
    ax.scatter(
        -static_radar_pc[:, 1],
        static_radar_pc[:, 0],
        s=2, c="gray", alpha=0.25, linewidths=0, label="static",
    )


def plot_moving_clusters(
    ax: plt.Axes,
    moving_radar_pc: np.ndarray,
    moving_mask: np.ndarray,
    confirmed_tracks: np.ndarray | None = None,
) -> None:
    """Color confirmed clusters by their track_id; unconfirmed clusters by cluster_id."""
    if len(moving_radar_pc) == 0:
        return

    cluster_color: dict[int, tuple] = {}
    if confirmed_tracks is not None and len(confirmed_tracks) > 0:
        for track in confirmed_tracks:
            cluster_color[int(track[3])] = _track_color(int(track[4]))

    for i, cid in enumerate(np.unique(moving_mask)):
        if cid < 0:
            continue
        pts = moving_radar_pc[moving_mask == cid]
        color = cluster_color.get(int(cid), _CMAP(i % 20))
        ax.scatter(
            -pts[:, 1], pts[:, 0],
            s=_CLUSTER_DOT_SIZE * _CLUSTER_DOT_SIZE, color=color, alpha=0.8, linewidths=0,
        )


def plot_confirmed_tracks(
    ax: plt.Axes,
    confirmed_tracks: np.ndarray,
) -> None:
    for track in confirmed_tracks:
        x, y, _, _, track_id = track
        tid = int(track_id)
        color = _track_color(tid)
        ax.plot(
            -y, x, "o",
            markersize=_TRACK_DOT_SIZE,
            color=color,
            markeredgecolor=_TRACK_DOT_EDGE_COLOR,
            markeredgewidth=_TRACK_DOT_EDGE_WIDTH,
            zorder=6,
        )
        ax.annotate(
            str(tid),
            xy=(-y + _TRACK_LABEL_XOFFSET, x + _TRACK_LABEL_YOFFSET),
            fontsize=_TRACK_LABEL_FONTSIZE, fontweight="bold",
            color=_TRACK_LABEL_COLOR, ha="left", va="bottom", zorder=7,
        )


def plot_unmatched_tracks(
    ax: plt.Axes,
    unmatched_tracks: np.ndarray,
) -> None:
    """Draw tracks not matched to radar this frame as hollow squares."""
    for track in unmatched_tracks:
        x, y, _, _, track_id = track
        tid = int(track_id)
        color = _track_color(tid)
        ax.plot(
            -y, x, "s",
            markersize=_TRACK_DOT_SIZE + 1,
            color="none",
            markeredgecolor=color,
            markeredgewidth=1.8,
            zorder=6,
        )
        ax.annotate(
            str(tid),
            xy=(-y + _TRACK_LABEL_XOFFSET, x + _TRACK_LABEL_YOFFSET),
            fontsize=_TRACK_LABEL_FONTSIZE, fontweight="bold",
            color=_TRACK_LABEL_COLOR, ha="left", va="bottom", zorder=7,
        )


_KF_PRED_BOX_HALF = 1.0   # half-size of KF prediction box in world metres


def plot_kf_predictions(
    ax: plt.Axes,
    kf_predictions: np.ndarray,
) -> None:
    """Draw KF predicted positions (before correct()) as hollow dashed rectangles."""
    s = _KF_PRED_BOX_HALF
    for track in kf_predictions:
        x, y, _, _, track_id = track
        tid = int(track_id)
        color = _track_color(tid)
        rect = mpatches.Rectangle(
            (-y - s, x - s), 2 * s, 2 * s,
            linewidth=1.2,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)


def plot_trajectories(
    ax: plt.Axes,
    track_trajectories: dict[int, list[tuple[float, float]]],
    active_track_ids: set[int],
) -> None:
    """
    Draw trajectory tails for active tracks only.
    Each tail shows the last _TRAJ_MAX_POINTS centroids connected by thin lines,
    with dots that fade from transparent (oldest) to opaque (newest).
    """
    for tid, positions in track_trajectories.items():
        if tid not in active_track_ids:
            continue

        tail = positions[-_TRAJ_MAX_POINTS:]
        n = len(tail)
        if n < 1:
            continue

        color = _track_color(tid)

        if n >= 2:
            xs = [-p[1] for p in tail]
            ys = [p[0] for p in tail]
            ax.plot(xs, ys, "-",
                    linewidth=_TRAJ_LINE_WIDTH, color=color,
                    alpha=_TRAJ_LINE_ALPHA, zorder=3)

        for i, (px, py) in enumerate(tail):
            t = i / max(n - 1, 1)
            alpha = _TRAJ_MIN_ALPHA + (_TRAJ_MAX_ALPHA - _TRAJ_MIN_ALPHA) * t
            size  = _TRAJ_DOT_MIN_SIZE + (_TRAJ_DOT_MAX_SIZE - _TRAJ_DOT_MIN_SIZE) * t
            ax.plot(-py, px, ".", markersize=size, color=color, alpha=alpha, zorder=4)


def plot_unmatched_clusters(
    ax: plt.Axes,
    cluster_centroids: dict[int, np.ndarray],
    confirmed_tracks: np.ndarray,
) -> None:
    """
    Plot the centroid of each cluster that has no confirmed track as a triangle.
    """
    if not cluster_centroids:
        return

    confirmed_cids: set[int] = set()
    if confirmed_tracks is not None and len(confirmed_tracks) > 0:
        confirmed_cids = {int(t[3]) for t in confirmed_tracks}

    for cid, centroid in cluster_centroids.items():
        if int(cid) in confirmed_cids:
            continue
        x, y = float(centroid[0]), float(centroid[1])
        color = _CMAP(int(cid) % 20)
        ax.plot(
            -y, x,
            _UNMATCHED_MARKER,
            markersize=_UNMATCHED_MARKER_SIZE,
            color=color,
            markeredgecolor=_UNMATCHED_EDGE_COLOR,
            markeredgewidth=_UNMATCHED_EDGE_WIDTH,
            alpha=_UNMATCHED_ALPHA,
            zorder=5,
        )


# ── camera panel helpers ──────────────────────────────────────────────────────

def plot_camera_panel(
    ax: plt.Axes,
    img_rgb: np.ndarray | None,
    frame_idx: int,
) -> None:
    """Bottom-left: clean camera frame, full panel."""
    ax.axis("off")
    if img_rgb is not None:
        ax.imshow(img_rgb)
    else:
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, "No image", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
    ax.set_title(f"Camera | frame {frame_idx:04d}", fontsize=9)


def plot_camera_gate_panel(
    ax: plt.Axes,
    cam_viz: dict | None,
    frame_idx: int,
) -> None:
    """
    Bottom-right: camera frame overlaid with YOLO boxes, SAM2 frozen-track masks,
    and projected positions of all radar-confirmed tracks.

    cam_viz keys (from CameraGate.render_viz):
      "img"       : np.ndarray RGB
      "boxes"     : [(x1,y1,x2,y2,cls,conf), ...]
      "frozen"    : [(track_id, u, v, mask|None), ...]
      "projected" : [(track_id, u, v), ...]
    """
    ax.axis("off")

    if cam_viz is None or cam_viz.get("img") is None:
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, "Camera gate disabled", transform=ax.transAxes,
                ha="center", va="center", color="#555", fontsize=9)
        ax.set_title(f"YOLO + SAM2 | frame {frame_idx:04d}", fontsize=9)
        return

    img = cam_viz["img"]
    ax.imshow(img)

    # ── YOLO bounding boxes (green, class label only) ────────────────────────
    for x1, y1, x2, y2, cls, _ in cam_viz.get("boxes", []):
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor="#22c55e", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 + 10, _COCO_NAMES.get(cls, str(cls)),
                fontsize=6, color="#22c55e", fontweight="bold")

    # ── SAM2 mask overlay + track centre markers (colour only, no text) ───────
    h, w = img.shape[:2]
    mask_overlay = np.zeros((h, w, 4), dtype=float)
    for tid, u, v, mask in cam_viz.get("frozen", []):
        color3 = _track_color(tid)[:3]
        if mask is not None:
            mask_overlay[mask, :3] = color3
            mask_overlay[mask, 3] = 0.45
        if u is not None:
            ax.plot(u, v, "o", color=color3, ms=6,
                    mec="white", mew=1.0, zorder=8)
    if mask_overlay[:, :, 3].max() > 0:
        ax.imshow(mask_overlay, zorder=5)

    # ── Unmatched tracks (square markers) ────────────────────────────────────
    for tid, u, v in cam_viz.get("unmatched", []):
        color3 = _track_color(tid)[:3]
        ax.plot(u, v, "s", color=color3, ms=7,
                mec="white", mew=1.0, zorder=9)

    n_yolo      = len(cam_viz.get("boxes",     []))
    n_frozen    = len(cam_viz.get("frozen",    []))
    n_unmatched = len(cam_viz.get("unmatched", []))
    ax.set_title(
        f"YOLO ({n_yolo} det)  SAM2 frozen ({n_frozen})  unmatched ({n_unmatched})"
        f" | frame {frame_idx:04d}",
        fontsize=9,
    )


# ── video export ─────────────────────────────────────────────────────────────

def save_frames_as_video(
    image_folder: Path,
    output_path: Path,
) -> None:
    """Combine all frame_XXXX.png images in image_folder into an mp4 video."""
    frame_paths = sorted(image_folder.glob("frame_*.png"))
    if not frame_paths:
        print(f"[viz] No frames found in {image_folder}, skipping video export.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(str(output_path), fps=_VIDEO_FPS) as writer:
        for path in frame_paths:
            writer.append_data(iio.imread(str(path)))

    print(f"[viz] Video saved: {output_path}")
