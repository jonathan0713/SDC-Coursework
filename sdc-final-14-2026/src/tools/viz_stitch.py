"""
Side-by-side stitch of mask-cluster viz (left) and tracking viz (right).

Usage
─────
  python src/tools/viz_stitch.py \\
      --left  data/viz/seq_4_mask_viz \\
      --right data/outputs/seq_4_student_<timestamp> \\
      --out   data/viz/seq_4_stitch          # optional, auto-named if omitted

  The --right argument points to the run folder produced by main.py.
  tracking_vis/ inside it is used automatically.
  The stitched frames and an mp4 are written to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import io

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import VIZ_ROOT
from utils.viz import save_frames_as_video


def _load(path: Path) -> np.ndarray:
    return iio.imread(str(path))


def stitch(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    """Place left and right images side-by-side via matplotlib subplots."""
    h_l, w_l = left_img.shape[:2]
    h_r, w_r = right_img.shape[:2]

    # Width ratio preserving each image's aspect
    total_w = w_l + w_r
    fig_h = 8.0
    fig_w = fig_h * total_w / max(h_l, h_r)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [w_l, w_r]},
    )
    for ax, img in ((ax_l, left_img), (ax_r, right_img)):
        ax.imshow(img[..., :3])
        ax.axis("off")

    fig.tight_layout(pad=0.1)
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=100)
    plt.close(fig)
    bio.seek(0)
    return iio.imread(bio)


def run(left_dir: Path, right_dir: Path, out_dir: Path) -> None:
    # Resolve tracking_vis sub-folder if right_dir is a run folder
    tracking_vis = right_dir / "tracking_vis"
    if tracking_vis.is_dir():
        right_dir = tracking_vis

    left_frames  = sorted(left_dir.glob("frame_*.png"))
    right_frames = sorted(right_dir.glob("frame_*.png"))

    if not left_frames:
        print(f"[stitch] No frames found in {left_dir}")
        return
    if not right_frames:
        print(f"[stitch] No frames found in {right_dir}")
        return

    # Match by stem name; take intersection
    left_map  = {p.stem: p for p in left_frames}
    right_map = {p.stem: p for p in right_frames}
    stems = sorted(set(left_map) & set(right_map))

    if not stems:
        print("[stitch] No matching frame names between the two folders.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stitch] {len(stems)} frames  left={left_dir.name}  right={right_dir.parent.name}")

    for stem in stems:
        left_img  = _load(left_map[stem])
        right_img = _load(right_map[stem])
        combined  = stitch(left_img, right_img)
        iio.imwrite(str(out_dir / f"{stem}.png"), combined)

        idx = int(stem.split("_")[-1])
        if idx % 50 == 0:
            print(f"  {stem} saved")

    video_path = out_dir.parent / f"{out_dir.name}.mp4"
    save_frames_as_video(out_dir, video_path)
    print(f"[stitch] done — frames in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stitch mask-cluster viz (left) with tracking viz (right)."
    )
    p.add_argument("--left",  required=True,
                   help="Folder of mask-cluster viz frames (e.g. data/viz/seq_4_mask_viz)")
    p.add_argument("--right", required=True,
                   help="Tracking run folder (data/outputs/seq_4_student_<ts>) "
                        "or its tracking_vis/ sub-folder directly")
    p.add_argument("--out",   default=None,
                   help="Output folder. Defaults to data/viz/<right-folder-name>_stitch/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    left_dir  = Path(args.left)
    right_dir = Path(args.right)
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = VIZ_ROOT / f"{right_dir.name}_stitch"

    run(left_dir, right_dir, out_dir)
