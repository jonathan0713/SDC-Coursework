#!/usr/bin/env python3
"""
互動式相機-雷達 Extrinsics 估算工具。

介面說明
--------
左圖：相機影像 + 移動群集中心投影點（以群集 ID 著色）
右圖：雷達俯視圖，標記相同群集中心

底部滑桿：即時調整 pitch / yaw / roll / height / forward / lateral
[Prev] / [Next] 按鈕：切換幀
[Save] 按鈕：將當前 R, t, K 儲存為 .npz

座標系約定
----------
雷達框架  : x = 前方, y = 左方, z = 上方
相機框架  : x = 右方, y = 下方, z = 前方（OpenCV 標準）

用法
----
    python tools/estimate_extrinsics.py --seq seq_4 \\
        --K data/outputs/seq_4_camera_K.npy \\
        --pitch 5 --height 1.3

    # 存檔
    python tools/estimate_extrinsics.py --seq seq_4 \\
        --K data/outputs/seq_4_camera_K.npy \\
        --pitch 7 --height 1.3 \\
        --save-ext data/outputs/seq_4_extrinsics.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sdc_tracking_utils import list_frame_names, load_radar_and_mask_data, separate_static_and_moving
from utils.detection import calculate_cluster_centroids
from utils.paths import DATA_ROOT, OUTPUT_ROOT


CMAP = plt.get_cmap("tab10")


# ── Extrinsics 建構 ──────────────────────────────────────────────────────────

R_BASE = np.array([
    [ 0., -1.,  0.],   # cam-x = -radar-y
    [ 0.,  0., -1.],   # cam-y = -radar-z
    [ 1.,  0.,  0.],   # cam-z =  radar-x
], dtype=float)


def _Rx(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _Ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def build_RT(pitch: float, yaw: float, roll: float,
             height: float, forward: float, lateral: float,
             ) -> tuple[np.ndarray, np.ndarray]:
    """建立 R (3×3) 和 t (3,)，使 P_cam = R @ P_radar + t。"""
    R = _Rz(roll) @ _Ry(yaw) @ _Rx(pitch) @ R_BASE
    cam_pos = np.array([forward, lateral, height], dtype=float)
    t = -R @ cam_pos
    return R, t


# ── 投影 ─────────────────────────────────────────────────────────────────────

def project(pts3d: np.ndarray, K: np.ndarray,
            R: np.ndarray, t: np.ndarray,
            img_w: int, img_h: int,
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    將 (N,3) 雷達點投影到影像平面。
    回傳 uv (M,2) 和 visible (N,) bool mask。
    """
    if len(pts3d) == 0:
        return np.empty((0, 2)), np.zeros(0, bool)
    pts_cam = (R @ pts3d.T).T + t
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return np.empty((0, 2)), in_front
    pc = pts_cam[in_front]
    proj = (K @ pc.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    return uv, in_front, in_img


# ── 資料載入 ─────────────────────────────────────────────────────────────────

def load_frame(mask_dir, radar_dir, frame_names, idx):
    """載入第 idx 幀的群集中心及原始移動點雲。"""
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, frame_names[idx])
    moving_mask, _, moving_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(moving_mask, moving_pc)
    return centroids  # dict[int, np.ndarray(3,)]


def load_image_rgb(image_dir, frame_idx):
    path = image_dir / f"{frame_idx:04d}.jpg"
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


# ── 互動式 GUI ────────────────────────────────────────────────────────────────

class ExtrinsicsGUI:
    def __init__(self, seq_dir: Path, K: np.ndarray,
                 init: dict, save_path: Path | None):
        self.mask_dir  = seq_dir / "mask_cluster"
        self.radar_dir = seq_dir / "radar"
        self.image_dir = seq_dir / "image"
        self.K         = K
        self.save_path = save_path
        self.frame_names = list_frame_names(self.mask_dir)
        self.n_frames  = len(self.frame_names)
        self.frame_idx = 0

        self.params = dict(init)  # pitch, yaw, roll, height, forward, lateral

        # ── 圖形版面 ──────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title("Extrinsics 互動調整")

        # 影像區與雷達俯視區
        self.ax_img = self.fig.add_axes([0.01, 0.28, 0.56, 0.70])
        self.ax_rad = self.fig.add_axes([0.59, 0.28, 0.40, 0.70])

        # 滑桿區（6 個）
        slider_specs = [
            # (label,    key,       min,   max,  valinit)
            ("Pitch (°)", "pitch",  -20,    30,  init["pitch"]),
            ("Yaw   (°)", "yaw",    -30,    30,  init["yaw"]),
            ("Roll  (°)", "roll",   -15,    15,  init["roll"]),
            ("Height (m)","height",   0.0,   5.0, init["height"]),
            ("Forward(m)","forward", -5.0,  5.0, init["forward"]),
            ("Lateral(m)","lateral", -5.0,  5.0, init["lateral"]),
        ]
        self.sliders: dict[str, Slider] = {}
        n = len(slider_specs)
        for i, (label, key, vmin, vmax, v0) in enumerate(slider_specs):
            ax_s = self.fig.add_axes([0.08, 0.02 + i * 0.038, 0.55, 0.025])
            sl = Slider(ax_s, label, vmin, vmax, valinit=v0, color="#4a90d9")
            sl.on_changed(lambda val, k=key: self._on_slider(k, val))
            self.sliders[key] = sl

        # 按鈕
        ax_prev = self.fig.add_axes([0.68, 0.12, 0.07, 0.05])
        ax_next = self.fig.add_axes([0.77, 0.12, 0.07, 0.05])
        ax_save = self.fig.add_axes([0.88, 0.12, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, "◀ Prev")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_save = Button(ax_save, "Save", color="#c8f0c8")
        self.btn_prev.on_clicked(lambda _: self._step_frame(-1))
        self.btn_next.on_clicked(lambda _: self._step_frame(+1))
        self.btn_save.on_clicked(lambda _: self._save())

        # 幀號標籤
        self.title_text = self.fig.text(0.70, 0.21, "", ha="center",
                                         fontsize=11, fontweight="bold")

        self._refresh()
        plt.show()

    # ── 事件處理 ──────────────────────────────────────────────────────────────

    def _on_slider(self, key: str, val: float):
        self.params[key] = float(val)
        self._refresh()

    def _step_frame(self, delta: int):
        self.frame_idx = (self.frame_idx + delta) % self.n_frames
        self._refresh()

    def _save(self):
        R, t = build_RT(**self.params)
        path = self.save_path or (OUTPUT_ROOT / "extrinsics.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, R=R, t=t, K=self.K)
        print(f"\n[Saved] {path}")
        print(f"  R =\n{R}")
        print(f"  t = {t}")
        self.fig.text(0.88, 0.08, f"Saved:\n{path.name}",
                      ha="center", fontsize=8, color="green")
        self.fig.canvas.draw_idle()

    # ── 繪製 ──────────────────────────────────────────────────────────────────

    def _refresh(self):
        R, t = build_RT(**self.params)
        img   = load_image_rgb(self.image_dir, self.frame_idx)
        centroids = load_frame(self.mask_dir, self.radar_dir,
                               self.frame_names, self.frame_idx)

        img_h, img_w = (img.shape[:2] if img is not None
                        else (928, 1440))

        self.ax_img.clear()
        self.ax_rad.clear()

        # ── 左：影像 + 投影 ───────────────────────────────────────────────────
        if img is not None:
            self.ax_img.imshow(img)

        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            pts3d = centroid[np.newaxis, :]           # (1, 3)
            result = project(pts3d, self.K, R, t, img_w, img_h)
            uv, in_front, in_img = result
            if in_front.any() and in_img.any():
                u, v = uv[in_img][0]
                self.ax_img.plot(u, v, "o", color=color,
                                 ms=12, markeredgecolor="white", mew=1.5)
                self.ax_img.text(u + 6, v - 6, f"c{cid}",
                                 color=color, fontsize=9, fontweight="bold",
                                 bbox=dict(fc="white", alpha=0.5, pad=1, ec="none"))

        p = self.params
        self.ax_img.set_title(
            f"frame {self.frame_idx:04d}  |  "
            f"pitch={p['pitch']:.1f}°  yaw={p['yaw']:.1f}°  "
            f"roll={p['roll']:.1f}°  h={p['height']:.2f}m",
            fontsize=10,
        )
        self.ax_img.axis("off")

        # ── 右：雷達俯視 ──────────────────────────────────────────────────────
        for cid, centroid in centroids.items():
            color = CMAP(cid % 10)
            self.ax_rad.plot(-centroid[1], centroid[0], "o",
                             color=color, ms=10,
                             markeredgecolor="white", mew=1.2)
            self.ax_rad.text(-centroid[1] + 0.5, centroid[0] + 0.5,
                             f"c{cid}", color=color, fontsize=9)

        self.ax_rad.set_xlim(-25, 25)
        self.ax_rad.set_ylim(0, 70)
        self.ax_rad.set_xlabel("-Y  (lateral, m)")
        self.ax_rad.set_ylabel("X  (forward, m)")
        self.ax_rad.set_title("Radar top-down  (moving cluster centroids)")
        self.ax_rad.grid(True, alpha=0.3)
        self.ax_rad.axvline(0, color="k", lw=0.8, ls="--")

        self.title_text.set_text(
            f"Frame {self.frame_idx + 1} / {self.n_frames}"
        )
        self.fig.canvas.draw_idle()


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="互動式 Extrinsics 估算工具")
    parser.add_argument("--seq", required=True,
                        choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--K", required=True,
                        help="intrinsics .npy 路徑（由 estimate_intrinsics.py 產生）")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--pitch",   type=float, default=5.0)
    parser.add_argument("--yaw",     type=float, default=0.0)
    parser.add_argument("--roll",    type=float, default=0.0)
    parser.add_argument("--height",  type=float, default=1.3)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--save-ext", default=None,
                        help="存檔路徑 .npz（也可在 GUI 中按 Save 存檔）")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    seq_dir   = data_root / args.seq
    K         = np.load(args.K)

    print(f"[K] f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"[初始參數] pitch={args.pitch}° yaw={args.yaw}° roll={args.roll}°  "
          f"height={args.height}m forward={args.forward}m lateral={args.lateral}m")

    init = dict(pitch=args.pitch, yaw=args.yaw, roll=args.roll,
                height=args.height, forward=args.forward, lateral=args.lateral)
    save_path = Path(args.save_ext) if args.save_ext else None

    ExtrinsicsGUI(seq_dir, K, init, save_path)


if __name__ == "__main__":
    main()
