"""
互動式相機內參調整工具。

滑桿即時調整 f / cx / cy / k1，同步顯示雷達 cluster 中心的影像投影。
外參 (R, t) 固定不動；[Save K] 同時更新 K.npy 與 extrinsics.npz 裡的 K。

介面說明
--------
左圖：相機影像 + cluster 中心投影點（以 cluster ID 著色）
右圖：雷達俯視參考圖

[Prev] / [Next]：切換幀
[Save K]：將目前 f / cx / cy 存入 K.npy 並同步更新 extrinsics.npz

調整建議
--------
• 所有點整體左右偏  → 調整 cx
• 所有點整體上下偏  → 調整 cy
• 遠處點偏更多、近處準確  → 調整 f（距離越遠的點對 f 越敏感）
• 邊緣往外或往內偏、中心準 → 調整 k1（負值=桶形、正值=枕形）
  注意：k1 僅用於即時預覽，不寫入 K.npy（K.npy 為 pinhole 模型）

用法
----
    python tools/adjust_intrinsics.py \\
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
from utils.paths import DATA_ROOT, OUTPUT_ROOT


CMAP = plt.get_cmap("tab10")


# ── 投影（含可選 k1 徑向畸變） ────────────────────────────────────────────────

def _project(
    pt3d: np.ndarray,
    f: float, cx: float, cy: float, k1: float,
    R: np.ndarray, t: np.ndarray,
    img_w: int, img_h: int,
) -> tuple[float, float] | None:
    p_cam = R @ pt3d + t
    if p_cam[2] <= 0.1:
        return None
    xn = p_cam[0] / p_cam[2]   # 歸一化座標
    yn = p_cam[1] / p_cam[2]
    if k1 != 0.0:
        r2 = xn**2 + yn**2
        s  = 1.0 + k1 * r2
        xn, yn = xn * s, yn * s
    u = f * xn + cx
    v = f * yn + cy
    return (float(u), float(v)) if 0 <= u < img_w and 0 <= v < img_h else None


# ── 資料載入 ──────────────────────────────────────────────────────────────────

def _load_image(image_dir: Path, idx: int) -> np.ndarray | None:
    p = image_dir / f"{idx:04d}.jpg"
    raw = cv2.imread(str(p))
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw is not None else None


def _load_centroids(mask_dir: Path, radar_dir: Path,
                    frame_names: list[str], idx: int) -> dict:
    fname = frame_names[idx]
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, fname)
    moving_mask, _, moving_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(moving_mask, moving_pc)
    return centroids


# ── GUI ───────────────────────────────────────────────────────────────────────

class IntrinsicsGUI:
    def __init__(
        self,
        seq_dir: Path,
        R: np.ndarray,
        t: np.ndarray,
        init_f: float,
        init_cx: float,
        init_cy: float,
        save_K_path: Path,
        extrinsics_path: Path,
    ):
        self.mask_dir       = seq_dir / "mask_cluster"
        self.radar_dir      = seq_dir / "radar"
        self.image_dir      = seq_dir / "image"
        self.R              = R
        self.t              = t
        self.save_K_path    = save_K_path
        self.extrinsics_path = extrinsics_path
        self.frame_names    = list_frame_names(self.mask_dir)
        self.n_frames       = len(self.frame_names)
        self.frame_idx      = 0

        self.f  = init_f
        self.cx = init_cx
        self.cy = init_cy
        self.k1 = 0.0

        # ── 版面 ──────────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title("Intrinsics 互動調整")

        self.ax_img = self.fig.add_axes([0.01, 0.30, 0.56, 0.68])
        self.ax_rad = self.fig.add_axes([0.59, 0.30, 0.40, 0.68])

        # 滑桿（從下往上排列）
        slider_specs = [
            # (label,       attr,  vmin,   vmax,   v0)
            ("k1  (畸變)", "k1",  -0.50,  0.50,  0.0),
            ("cy  (px)",   "cy",   200.0, 700.0, init_cy),
            ("cx  (px)",   "cx",   300.0, 1100.0, init_cx),
            ("f   (px)",   "f",    500.0, 1600.0, init_f),
        ]
        self.sliders: dict[str, Slider] = {}
        for i, (label, attr, vmin, vmax, v0) in enumerate(slider_specs):
            ax_s = self.fig.add_axes([0.08, 0.03 + i * 0.055, 0.55, 0.030])
            sl = Slider(ax_s, label, vmin, vmax, valinit=v0, color="#4a90d9")
            sl.on_changed(lambda val, a=attr: self._on_slider(a, val))
            self.sliders[attr] = sl

        # 按鈕
        ax_prev = self.fig.add_axes([0.68, 0.14, 0.07, 0.05])
        ax_next = self.fig.add_axes([0.77, 0.14, 0.07, 0.05])
        ax_save = self.fig.add_axes([0.87, 0.14, 0.10, 0.05])
        self.btn_prev = Button(ax_prev, "◀ Prev")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_save = Button(ax_save, "Save K", color="#c8f0c8")
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_save.on_clicked(lambda _: self._save())

        self._status = self.fig.text(0.70, 0.09, "", ha="center",
                                      fontsize=9, color="green")
        self._frame_label = self.fig.text(0.70, 0.23, "", ha="center",
                                           fontsize=11, fontweight="bold")
        self._refresh()
        plt.show()

    # ── 事件 ──────────────────────────────────────────────────────────────────

    def _on_slider(self, attr: str, val: float) -> None:
        setattr(self, attr, float(val))
        self._refresh()

    def _step(self, delta: int) -> None:
        self.frame_idx = (self.frame_idx + delta) % self.n_frames
        self._refresh()

    def _save(self) -> None:
        K = np.array(
            [[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]],
            dtype=float,
        )
        # 更新 K.npy
        self.save_K_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.save_K_path), K)

        # 同步更新 extrinsics.npz 裡的 K
        if self.extrinsics_path.exists():
            ext = np.load(self.extrinsics_path)
            np.savez(str(self.extrinsics_path), R=ext["R"], t=ext["t"], K=K)

        msg = f"Saved  f={self.f:.1f}  cx={self.cx:.1f}  cy={self.cy:.1f}"
        if self.k1 != 0.0:
            msg += f"\n（k1={self.k1:.4f} 僅供預覽，未存入 K.npy）"
        print(f"\n[Save] {msg}")
        print(f"  K.npy         → {self.save_K_path}")
        print(f"  extrinsics.npz → {self.extrinsics_path}")

        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    # ── 繪製 ──────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        img      = _load_image(self.image_dir, self.frame_idx)
        centroids = _load_centroids(self.mask_dir, self.radar_dir,
                                    self.frame_names, self.frame_idx)
        img_h, img_w = img.shape[:2] if img is not None else (928, 1440)

        self.ax_img.clear()
        self.ax_rad.clear()

        # 左欄：影像 + 投影點
        if img is not None:
            self.ax_img.imshow(img)

        n_vis = 0
        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            uv = _project(centroid, self.f, self.cx, self.cy, self.k1,
                          self.R, self.t, img_w, img_h)
            if uv is not None:
                u, v = uv
                self.ax_img.plot(u, v, "o", color=color,
                                 ms=12, markeredgecolor="white", mew=1.5)
                self.ax_img.text(
                    u + 6, v - 6, f"c{cid}", color=color,
                    fontsize=9, fontweight="bold",
                    bbox=dict(fc="white", alpha=0.5, pad=1, ec="none"),
                )
                n_vis += 1

        self.ax_img.set_title(
            f"frame {self.frame_idx:04d}  |  {n_vis}/{len(centroids)} visible  |  "
            f"f={self.f:.1f}  cx={self.cx:.1f}  cy={self.cy:.1f}  k1={self.k1:.4f}",
            fontsize=10,
        )
        self.ax_img.axis("off")

        # 右欄：雷達俯視
        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            self.ax_rad.plot(-centroid[1], centroid[0], "^",
                             color=color, ms=10,
                             markeredgecolor="white", mew=1.2)
            self.ax_rad.text(-centroid[1] + 0.5, centroid[0] + 0.5,
                             f"c{cid}", color=color, fontsize=9)

        self.ax_rad.set_xlim(-25, 25)
        self.ax_rad.set_ylim(0, 70)
        self.ax_rad.set_xlabel("−Y  (lateral, m)")
        self.ax_rad.set_ylabel("X  (forward, m)")
        self.ax_rad.set_title("Radar top-down  (固定參考)")
        self.ax_rad.grid(True, alpha=0.3)
        self.ax_rad.axvline(0, color="k", lw=0.8, ls="--")

        self._frame_label.set_text(
            f"Frame {self.frame_idx + 1} / {self.n_frames}"
        )
        self.fig.canvas.draw_idle()


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="互動式內參調整工具")
    parser.add_argument("--seq", required=True,
                        choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--extrinsics", required=True,
                        help="extrinsics .npz 路徑（含 R, t, K）")
    parser.add_argument("--save-K", default=None,
                        help="K.npy 儲存路徑（預設 data/outputs/<seq>_camera_K.npy）")
    args = parser.parse_args()

    ext_path = Path(args.extrinsics)
    ext      = np.load(ext_path)
    R, t, K  = ext["R"], ext["t"], ext["K"]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])

    print(f"[載入 K]  f={f:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"[載入 R]\n{R}")
    print(f"[載入 t]  {t}")

    seq_dir  = DATA_ROOT / args.seq
    save_K   = (
        Path(args.save_K) if args.save_K
        else OUTPUT_ROOT / f"{args.seq}_camera_K.npy"
    )

    IntrinsicsGUI(seq_dir, R, t, f, cx, cy, save_K, ext_path)


if __name__ == "__main__":
    main()
