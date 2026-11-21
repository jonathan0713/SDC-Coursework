"""
互動式投影 Z 高度調整工具。

雷達 cluster centroid 的 z 軸雜訊會導致投影點在影像上垂直跳動。
此工具讓你以滑桿即時調整「固定 z 高度」，觀察鎖定後的投影穩定性。

介面說明
--------
左圖：相機影像 + cluster 中心投影點
  ✕  灰色叉號：原始雷達 z 投影（供對照）
  ●  彩色實心圓：固定 z 高度投影（主要參考）
右圖：雷達俯視圖，標注各 cluster 的原始 z 值（m）

[Prev] / [Next]：切換幀
[Print z]：將目前 z 高度印至終端，並顯示 config 建議值

用法
----
    python tools/tune_proj_z.py \\
        --seq seq_4 \\
        --extrinsics data/outputs/seq_4_extrinsics.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    separate_static_and_moving,
)
from utils.detection import calculate_cluster_centroids
from utils.paths import DATA_ROOT


CMAP = plt.get_cmap("tab10")


def _project(
    pt3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_w: int,
    img_h: int,
    z_override: float | None = None,
) -> tuple[float, float] | None:
    p = pt3d.copy()
    if z_override is not None:
        p = np.array([p[0], p[1], z_override])
    p_cam = R @ p + t
    if p_cam[2] <= 0.1:
        return None
    proj = K @ p_cam
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    return (float(u), float(v)) if 0 <= u < img_w and 0 <= v < img_h else None


def _load_image(image_dir: Path, idx: int) -> np.ndarray | None:
    p = image_dir / f"{idx:04d}.jpg"
    raw = cv2.imread(str(p))
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw is not None else None


def _load_centroids(
    mask_dir: Path, radar_dir: Path, frame_names: list[str], idx: int
) -> dict:
    fname = frame_names[idx]
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, fname)
    moving_mask, _, moving_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(moving_mask, moving_pc)
    return centroids


class ProjZGUI:
    def __init__(self, seq_dir: Path, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        self.mask_dir    = seq_dir / "mask_cluster"
        self.radar_dir   = seq_dir / "radar"
        self.image_dir   = seq_dir / "image"
        self.K, self.R, self.t = K, R, t
        self.frame_names = list_frame_names(self.mask_dir)
        self.n_frames    = len(self.frame_names)
        self.frame_idx   = 0
        self.z_height    = 0.0

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title("投影 Z 高度調整")

        self.ax_img = self.fig.add_axes([0.01, 0.28, 0.56, 0.70])
        self.ax_rad = self.fig.add_axes([0.60, 0.28, 0.38, 0.70])

        # 滑桿
        ax_sl = self.fig.add_axes([0.08, 0.10, 0.55, 0.030])
        self.sl_z = Slider(ax_sl, "z 高度 (m)", -2.0, 3.0, valinit=0.0, color="#4a90d9")
        self.sl_z.on_changed(self._on_slider)

        # 按鈕
        ax_prev  = self.fig.add_axes([0.66, 0.14, 0.07, 0.05])
        ax_next  = self.fig.add_axes([0.75, 0.14, 0.07, 0.05])
        ax_print = self.fig.add_axes([0.85, 0.14, 0.11, 0.05])
        self.btn_prev  = Button(ax_prev,  "◀ Prev")
        self.btn_next  = Button(ax_next,  "Next ▶")
        self.btn_print = Button(ax_print, "Print z", color="#c8f0c8")
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_print.on_clicked(lambda _: self._print_z())

        self._frame_label = self.fig.text(
            0.74, 0.23, "", ha="center", fontsize=11, fontweight="bold",
        )
        self._hint = self.fig.text(
            0.50, 0.04,
            "灰色 ✕ = 原始 z 投影　彩色 ● = 固定 z 投影　"
            "右圖數字 = 各 cluster 原始 z (m)",
            ha="center", fontsize=9, color="#555",
        )

        self._refresh()
        plt.show()

    def _on_slider(self, val: float) -> None:
        self.z_height = float(val)
        self._refresh()

    def _step(self, delta: int) -> None:
        self.frame_idx = (self.frame_idx + delta) % self.n_frames
        self._refresh()

    def _print_z(self) -> None:
        print(f"\n[tune_proj_z]  z_height = {self.z_height:.3f} m")
        print(f"  config 建議設定:  proj_z_height: {self.z_height:.3f}")

    def _refresh(self) -> None:
        img       = _load_image(self.image_dir, self.frame_idx)
        centroids = _load_centroids(
            self.mask_dir, self.radar_dir, self.frame_names, self.frame_idx,
        )
        img_h, img_w = img.shape[:2] if img is not None else (928, 1440)

        self.ax_img.clear()
        self.ax_rad.clear()

        if img is not None:
            self.ax_img.imshow(img)

        n_orig = n_fixed = 0
        z_vals: list[float] = []

        for cid, centroid in centroids.items():
            color  = CMAP(cid % 10)
            z_vals.append(float(centroid[2]))

            # 原始 z 投影（灰色叉號，供對照）
            uv_orig = _project(centroid, self.K, self.R, self.t, img_w, img_h)
            if uv_orig is not None:
                self.ax_img.plot(
                    uv_orig[0], uv_orig[1], "x",
                    color="gray", ms=9, mew=2.0, alpha=0.65, zorder=4,
                )
                n_orig += 1

            # 固定 z 投影（彩色實心圓，主要參考）
            uv_fix = _project(
                centroid, self.K, self.R, self.t, img_w, img_h,
                z_override=self.z_height,
            )
            if uv_fix is not None:
                self.ax_img.plot(
                    uv_fix[0], uv_fix[1], "o",
                    color=color, ms=12,
                    markeredgecolor="white", mew=1.5, zorder=6,
                )
                self.ax_img.text(
                    uv_fix[0] + 6, uv_fix[1] - 6, f"c{cid}",
                    color=color, fontsize=9, fontweight="bold",
                    bbox=dict(fc="white", alpha=0.5, pad=1, ec="none"),
                )
                n_fixed += 1

        z_range_str = (
            f"  |  z_raw: [{min(z_vals):.2f}, {max(z_vals):.2f}] m"
            if z_vals else ""
        )
        self.ax_img.set_title(
            f"frame {self.frame_idx:04d}  |  fixed z = {self.z_height:.2f} m"
            f"  |  orig={n_orig}  fixed={n_fixed}" + z_range_str,
            fontsize=9,
        )
        self.ax_img.axis("off")

        # 右欄：雷達俯視 + 原始 z 標注
        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            self.ax_rad.plot(
                -centroid[1], centroid[0], "^",
                color=color, ms=10, markeredgecolor="white", mew=1.2,
            )
            self.ax_rad.text(
                -centroid[1] + 0.5, centroid[0] + 0.5,
                f"c{cid}\nz={centroid[2]:.2f}m",
                color=color, fontsize=8,
            )

        self.ax_rad.set_xlim(-25, 25)
        self.ax_rad.set_ylim(0, 70)
        self.ax_rad.set_xlabel("−Y  (lateral, m)")
        self.ax_rad.set_ylabel("X  (forward, m)")
        self.ax_rad.set_title("Radar top-down  |  原始 z 值標注")
        self.ax_rad.grid(True, alpha=0.3)
        self.ax_rad.axvline(0, color="k", lw=0.8, ls="--")

        self._frame_label.set_text(f"Frame {self.frame_idx + 1} / {self.n_frames}")
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description="互動式投影 Z 高度調整工具")
    parser.add_argument(
        "--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"],
    )
    parser.add_argument(
        "--extrinsics", required=True,
        help="extrinsics .npz 路徑（含 K, R, t）",
    )
    args = parser.parse_args()

    ext = np.load(args.extrinsics)
    K, R, t = ext["K"], ext["R"], ext["t"]
    print(f"[載入 K]  f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"[載入 R]\n{R}")
    print(f"[載入 t]  {t}")

    seq_dir = DATA_ROOT / args.seq
    ProjZGUI(seq_dir, K, R, t)


if __name__ == "__main__":
    main()
