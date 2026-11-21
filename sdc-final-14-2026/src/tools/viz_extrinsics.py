"""
批量驗證相機外參：將全序列的雷達 cluster 中心投影至相機影像，輸出驗證影片。

左欄：相機影像 + cluster 投影點（彩色圓點 + c{id} 標籤）
右欄：雷達俯視圖（同色三角 + c{id} 標籤）

用法
----
    python tools/viz_extrinsics.py \\
        --seq seq_4 \\
        --extrinsics data/outputs/seq_4_extrinsics.npz \\
        --out data/viz/seq_4_extrinsics_check.mp4

    # 每 2 幀取一幀（加速輸出）
    python tools/viz_extrinsics.py --seq seq_4 \\
        --extrinsics data/outputs/seq_4_extrinsics.npz \\
        --every 2
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    separate_static_and_moving,
)
from utils.detection import calculate_cluster_centroids
from utils.paths import DATA_ROOT, VIZ_ROOT
from utils.viz import save_frames_as_video

CMAP = plt.get_cmap("tab10")


def _project_pt(
    pt3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[float, float] | None:
    """將單一 3D 點投影到影像平面；不在視野內回傳 None。"""
    p_cam = R @ pt3d + t
    if p_cam[2] <= 0.1:
        return None
    proj = K @ p_cam
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    return (float(u), float(v)) if 0 <= u < img_w and 0 <= v < img_h else None


def render_frame(
    frame_idx: int,
    image_path: Path,
    mask_dir: Path,
    radar_dir: Path,
    frame_names: list[str],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """渲染一幀驗證圖（左: 相機+投影，右: 雷達俯視），回傳 RGB uint8 array。"""
    raw = cv2.imread(str(image_path))
    img = (
        cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if raw is not None
        else np.zeros((928, 1440, 3), np.uint8)
    )
    img_h, img_w = img.shape[:2]

    fname = frame_names[frame_idx]
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, fname)
    moving_mask, _, moving_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(moving_mask, moving_pc)

    fig, (ax_img, ax_rad) = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # 左欄：相機影像 + cluster 投影點
    ax_img.imshow(img)
    n_vis = 0
    for cid, centroid in centroids.items():
        color = CMAP(cid % 10)
        uv = _project_pt(centroid, K, R, t, img_w, img_h)
        if uv is not None:
            u, v = uv
            ax_img.plot(u, v, "o", color=color, ms=10,
                        markeredgecolor="white", mew=1.5)
            ax_img.text(
                u + 6, v - 6, f"c{cid}", color=color,
                fontsize=8, fontweight="bold",
                bbox=dict(fc="white", alpha=0.55, pad=1, ec="none"),
            )
            n_vis += 1
    ax_img.set_title(
        f"frame {frame_idx:04d}  │  {n_vis}/{len(centroids)} clusters projected",
        fontsize=9,
    )
    ax_img.axis("off")

    # 右欄：雷達俯視圖
    for cid, centroid in centroids.items():
        color = CMAP(cid % 10)
        ax_rad.plot(
            -centroid[1], centroid[0], "^", color=color,
            ms=9, markeredgecolor="white", mew=1.2,
        )
        ax_rad.text(
            -centroid[1] + 0.4, centroid[0] + 0.4,
            f"c{cid}", color=color, fontsize=8,
        )
    ax_rad.set_xlim(-25, 25)
    ax_rad.set_ylim(0, 60)
    ax_rad.set_xlabel("−Y  (lateral, m)", fontsize=8)
    ax_rad.set_ylabel("X  (forward, m)", fontsize=8)
    ax_rad.set_title("Radar top-down", fontsize=9)
    ax_rad.grid(True, alpha=0.3)
    ax_rad.axvline(0, color="k", lw=0.8, ls="--")

    fig.tight_layout(pad=0.3)
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=100)
    plt.close(fig)
    bio.seek(0)
    return iio.imread(bio)


def run(seq: str, extrinsics_path: Path, out_path: Path, every: int = 1) -> None:
    ext = np.load(extrinsics_path)
    R, t, K = ext["R"], ext["t"], ext["K"]
    print(f"[K]  f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"[R]\n{R}")
    print(f"[t]  {t}")

    seq_path = DATA_ROOT / seq
    m_dir    = seq_path / "mask_cluster"
    r_dir    = seq_path / "radar"
    i_dir    = seq_path / "image"

    frame_names = list_frame_names(m_dir)
    n = len(frame_names)
    print(f"[{seq}]  {n} frames")

    frames_dir = out_path.parent / (out_path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    indices = range(0, n, every)
    for i, idx in enumerate(indices):
        img_path  = i_dir / f"{idx:04d}.jpg"
        frame_arr = render_frame(idx, img_path, m_dir, r_dir, frame_names, K, R, t)
        iio.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), frame_arr)
        if i % 20 == 0 or idx == n - 1:
            print(f"  frame {idx:04d}/{n - 1}")

    save_frames_as_video(frames_dir, out_path)
    print(f"[done]  {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量外參驗證影片")
    parser.add_argument("--seq", required=True,
                        choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--extrinsics", required=True,
                        help="extrinsics .npz 路徑（含 R, t, K）")
    parser.add_argument("--out", default=None,
                        help="輸出 mp4（預設 data/viz/<seq>_extrinsics_check.mp4）")
    parser.add_argument("--every", type=int, default=1,
                        help="每 N 幀取一幀（預設 1）")
    args = parser.parse_args()

    out_path = (
        Path(args.out) if args.out
        else VIZ_ROOT / f"{args.seq}_extrinsics_check.mp4"
    )
    run(args.seq, Path(args.extrinsics), out_path, args.every)


if __name__ == "__main__":
    main()
