"""
Visualise raw radar point clouds after cluster masking.

For each frame:
  - Static points   → gray, small
  - Moving points   → colored by cluster_id (from mask)
  - Cluster centroid label → cluster id, same color
  - Camera image inset in the bottom-left corner

Output: one PNG per frame + an mp4 video.

Usage
─────
  python src/tools/viz_mask_clusters.py --seq seq_4

  python src/tools/viz_mask_clusters.py \\
      --seq seq_1 \\
      --out data/outputs/mask_viz \\
      --every 1 \\
      --xlim -40 40 \\
      --ylim 0 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    separate_static_and_moving,
)
from utils.paths import VIZ_ROOT, image_dir, mask_dir, radar_dir
from utils.viz import save_frames_as_video

_CMAP = plt.cm.tab20


def _cluster_color(cid: int):
    return _CMAP(int(cid) % 20)


def _plot_frame(
    ax: plt.Axes,
    static_pc: np.ndarray,
    moving_pc: np.ndarray,
    moving_mask: np.ndarray,
    frame_idx: int,
    image_path: Path | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.cla()

    # Static points
    if len(static_pc):
        ax.scatter(
            -static_pc[:, 1], static_pc[:, 0],
            s=3, c="gray", alpha=0.25, linewidths=0, label="static",
        )

    # Moving points colored by cluster_id
    cluster_ids = np.unique(moving_mask)
    for cid in cluster_ids:
        pts = moving_pc[moving_mask == cid]
        color = _cluster_color(cid)
        ax.scatter(
            -pts[:, 1], pts[:, 0],
            s=14, color=color, alpha=0.85, linewidths=0,
        )
        # Centroid label
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ax.annotate(
            str(int(cid)),
            xy=(-cy, cx),
            xytext=(-cy + 0.6, cx + 0.6),
            fontsize=9, fontweight="bold",
            color=color,
            va="bottom", ha="left", zorder=6,
        )

    # Camera inset — bottom-left
    if image_path is not None and image_path.exists():
        img = plt.imread(str(image_path))
        inset = ax.inset_axes([0.0, 0.0, 0.30, 0.22])
        inset.imshow(img)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("Camera", fontsize=7, pad=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Lateral (–Y, m)")
    ax.set_ylabel("Forward (X, m)")
    ax.set_title(f"Mask clusters | frame {frame_idx:04d}  "
                 f"(static={len(static_pc)}, moving={len(moving_pc)}, "
                 f"clusters={len(cluster_ids)})")
    ax.grid(True, alpha=0.2)


def run(
    seq: str,
    out: Path,
    every: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _mask_dir  = mask_dir(seq)
    _radar_dir = radar_dir(seq)
    _image_dir = image_dir(seq)
    out.mkdir(parents=True, exist_ok=True)

    frame_names = list_frame_names(_mask_dir)
    print(f"[viz_mask] seq={seq}  frames={len(frame_names)}  out={out}")

    fig, ax = plt.subplots(figsize=(10, 8))

    for frame_idx, mask_name in enumerate(frame_names):
        if every > 1 and frame_idx % every != 0:
            continue

        mask, radar_pc = load_radar_and_mask_data(_mask_dir, _radar_dir, mask_name)
        moving_mask, _, moving_pc, static_pc, _ = separate_static_and_moving(mask, radar_pc)

        stem = Path(mask_name).stem
        img_path = _image_dir / f"{stem}.jpg" if _image_dir.exists() else None

        _plot_frame(ax, static_pc, moving_pc, moving_mask,
                    frame_idx, img_path, xlim, ylim)

        fig.savefig(out / f"frame_{frame_idx:04d}.png", dpi=110, bbox_inches="tight")

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx:04d} saved")

    plt.close(fig)

    save_frames_as_video(out, VIZ_ROOT / f"{seq}_mask_clusters.mp4")
    print(f"[viz_mask] done — frames in {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise radar point clouds colored by cluster mask."
    )
    p.add_argument("--seq",   required=True,
                   choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    p.add_argument("--out",   default=None,
                   help="Output folder for PNGs. Defaults to data/outputs/<seq>_mask_viz/")
    p.add_argument("--every", type=int, default=1,
                   help="Render every N-th frame (default 1 = all frames)")
    p.add_argument("--xlim",  type=float, nargs=2, default=[-50.0, 50.0],
                   metavar=("X_MIN", "X_MAX"),
                   help="Lateral axis range in metres (default: -50 50)")
    p.add_argument("--ylim",  type=float, nargs=2, default=[0.0, 80.0],
                   metavar=("Y_MIN", "Y_MAX"),
                   help="Forward axis range in metres (default: 0 80)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out) if args.out else VIZ_ROOT / f"{args.seq}_mask_viz"
    run(
        seq=args.seq,
        out=out,
        every=args.every,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
    )
