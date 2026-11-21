"""
YOLO 車輛偵測視覺化工具。

用途：驗證 YOLO 偵測結果是否正確，尤其是影像中心附近像素較小的車輛。
若提供 extrinsics，會同時疊加雷達 cluster 投影點，方便比對雷達與視覺的對應。

左欄：相機影像 + YOLO bounding box（含類別與信心值）+ cluster 投影點（可選）
右欄：雷達俯視圖（同幀 cluster）

用法
----
    # 基本（自動尋找 extrinsics）
    python tools/viz_yolo_detections.py --seq seq_4

    # 不疊加 cluster 投影
    python tools/viz_yolo_detections.py --seq seq_4 --no-proj

    # 調整信心值門檻 / 只輸出前 N 幀
    python tools/viz_yolo_detections.py --seq seq_4 --conf 0.4 --frames 100

    # 每 N 幀取一幀（加速預覽）
    python tools/viz_yolo_detections.py --seq seq_4 --every 3
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
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    separate_static_and_moving,
)
from utils.detection import calculate_cluster_centroids
from utils.paths import DATA_ROOT, OUTPUT_ROOT, VIZ_ROOT
from utils.viz import save_frames_as_video

# COCO 類別名稱（只用到車輛相關）
_COCO_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}
CMAP = plt.get_cmap("tab10")


# ── 投影 ──────────────────────────────────────────────────────────────────────

def _project_pt(pt3d, K, R, t, img_w, img_h):
    p_cam = R @ pt3d + t
    if p_cam[2] <= 0.1:
        return None
    proj = K @ p_cam
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    return (float(u), float(v)) if 0 <= u < img_w and 0 <= v < img_h else None


# ── 單幀渲染 ─────────────────────────────────────────────────────────────────

def render_frame(
    frame_idx: int,
    image_path: Path,
    mask_dir: Path,
    radar_dir: Path,
    frame_names: list[str],
    yolo_model,
    yolo_conf: float,
    yolo_classes: list[int],
    K: np.ndarray | None,
    R: np.ndarray | None,
    t: np.ndarray | None,
    show_proj: bool,
) -> np.ndarray:
    # 讀影像
    raw = cv2.imread(str(image_path))
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw is not None else np.zeros((928, 1440, 3), np.uint8)
    img_h, img_w = img.shape[:2]

    # 讀 cluster
    fname = frame_names[frame_idx]
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, fname)
    moving_mask, _, moving_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(moving_mask, moving_pc)

    # YOLO 偵測
    results = yolo_model(img, classes=yolo_classes, conf=yolo_conf, verbose=False)
    boxes = results[0].boxes  # xyxy, conf, cls

    fig, (ax_img, ax_rad) = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # 左欄：影像 + YOLO boxes
    ax_img.imshow(img)

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf)
        cls  = int(b.cls)
        label = f"{_COCO_NAMES.get(cls, cls)} {conf:.2f}"
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="#22c55e", facecolor="none",
        )
        ax_img.add_patch(rect)
        ax_img.text(x1 + 3, y1 - 5, label,
                    fontsize=7.5, color="#22c55e", fontweight="bold",
                    bbox=dict(fc="black", alpha=0.45, pad=1, ec="none"))

    # 疊加 cluster 投影點（可選）
    if show_proj and K is not None:
        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            uv = _project_pt(centroid, K, R, t, img_w, img_h)
            if uv is not None:
                u, v = uv
                ax_img.plot(u, v, "o", color=color,
                            ms=9, markeredgecolor="white", mew=1.5, zorder=5)
                ax_img.text(u + 5, v - 5, f"c{cid}", color=color,
                            fontsize=7.5, fontweight="bold",
                            bbox=dict(fc="white", alpha=0.5, pad=1, ec="none"))

    n_det = len(boxes)
    title = f"frame {frame_idx:04d}  │  YOLO: {n_det} vehicles"
    if show_proj and K is not None:
        title += f"  │  clusters: {len(centroids)}"
    ax_img.set_title(title, fontsize=9)
    ax_img.axis("off")

    # 右欄：雷達俯視
    for cid, centroid in centroids.items():
        color = CMAP(cid % 10)
        ax_rad.plot(-centroid[1], centroid[0], "^", color=color,
                    ms=9, markeredgecolor="white", mew=1.2)
        ax_rad.text(-centroid[1] + 0.4, centroid[0] + 0.4,
                    f"c{cid}", color=color, fontsize=8)
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


# ── 主流程 ───────────────────────────────────────────────────────────────────

def run(
    seq: str,
    out_path: Path,
    yolo_model_name: str,
    yolo_conf: float,
    yolo_classes: list[int],
    extrinsics_path: Path | None,
    show_proj: bool,
    every: int,
    max_frames: int | None,
) -> None:
    from ultralytics import YOLO
    model = YOLO(yolo_model_name)

    # 載入外參（可選）
    K = R = t = None
    if show_proj:
        ext_path = extrinsics_path or OUTPUT_ROOT / f"{seq}_extrinsics.npz"
        if ext_path.exists():
            ext = np.load(ext_path)
            K, R, t = ext["K"], ext["R"], ext["t"]
            print(f"[extrinsics] {ext_path}  f={K[0,0]:.1f}")
        else:
            print(f"[extrinsics] 找不到 {ext_path}，跳過 cluster 投影疊加")

    seq_path  = DATA_ROOT / seq
    m_dir     = seq_path / "mask_cluster"
    r_dir     = seq_path / "radar"
    i_dir     = seq_path / "image"

    frame_names = list_frame_names(m_dir)
    n = len(frame_names) if max_frames is None else min(len(frame_names), max_frames)
    print(f"[{seq}]  共 {len(frame_names)} 幀，處理前 {n} 幀，每 {every} 幀取一幀")

    frames_dir = out_path.parent / (out_path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    indices = range(0, n, every)
    for i, idx in enumerate(indices):
        img_path  = i_dir / f"{idx:04d}.jpg"
        frame_arr = render_frame(
            idx, img_path, m_dir, r_dir, frame_names,
            model, yolo_conf, yolo_classes,
            K, R, t, show_proj,
        )
        iio.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), frame_arr)
        if i % 20 == 0 or idx == n - 1:
            print(f"  frame {idx:04d}/{n - 1}")

    save_frames_as_video(frames_dir, out_path)
    print(f"[done]  {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 車輛偵測視覺化")
    parser.add_argument("--seq", required=True,
                        choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--out", default=None,
                        help="輸出 mp4（預設 data/viz/<seq>_yolo_detections.mp4）")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO 模型名稱或路徑（預設 yolo11n.pt）")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="信心值門檻（預設 0.35）")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3, 5, 7],
                        help="COCO 類別（預設 2=car 3=motorcycle 5=bus 7=truck）")
    parser.add_argument("--extrinsics", default=None,
                        help="extrinsics .npz 路徑（預設自動尋找）")
    parser.add_argument("--no-proj", action="store_true",
                        help="不疊加 cluster 投影點")
    parser.add_argument("--every", type=int, default=1,
                        help="每 N 幀取一幀（預設 1）")
    parser.add_argument("--frames", type=int, default=None,
                        help="只處理前 N 幀（預設全部）")
    args = parser.parse_args()

    out_path = (
        Path(args.out) if args.out
        else VIZ_ROOT / f"{args.seq}_yolo_detections.mp4"
    )
    ext_path = Path(args.extrinsics) if args.extrinsics else None

    run(
        seq=args.seq,
        out_path=out_path,
        yolo_model_name=args.model,
        yolo_conf=args.conf,
        yolo_classes=args.classes,
        extrinsics_path=ext_path,
        show_proj=not args.no_proj,
        every=args.every,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    main()
