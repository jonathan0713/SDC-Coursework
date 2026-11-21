#!/usr/bin/env python3
"""
用 SAM2 互動標注車輛位置，並擬合相機-雷達外參（Extrinsics）。

流程
----
1. [標注模式]  顯示影像，使用者點擊車輛，SAM2 生成 mask，取 2D centroid。
2. [配對模式]  手動指定哪個 cluster 對應哪個 mask；不確定的 cluster 可跳過。
3. [PnP]       用所有幀的對應點跑 solvePnPRansac，輸出精確 R, t。
4. [驗證]      重新投影雷達 cluster，顯示誤差。

用法
----
    python tools/annotate_extrinsics_sam2.py \\
        --seq seq_4 \\
        --K data/outputs/seq_4_camera_K.npy \\
        --frames 0 50 100 200 \\
        --pitch 0 --height 1.3 \\
        --out data/outputs/seq_4_extrinsics.npz

標注模式操作
------------
    左鍵          新增前景點（綠色），SAM2 自動生成 mask
    右鍵          新增背景點（紅色），排除雜訊
    Space / C     確認當前 mask，準備標注下一台車
    Z             撤銷當前物件最後一個點
    Enter         完成標注，進入配對模式
    Q / Esc       放棄此幀並結束整個流程

配對模式操作
------------
    左鍵點 cluster 三角 (C0/C1...)     選取該 cluster（黃色高亮）
    左鍵點 mask 方塊 (M0/M1...)        將選取的 cluster 配對至此 mask
    S                                  跳過選取的 cluster（不配對）
    Z                                  撤銷最後一對配對
    Enter                              完成此幀，進入下一幀
    Q / Esc                            跳過此幀
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sdc_tracking_utils import list_frame_names, load_radar_and_mask_data, separate_static_and_moving
from utils.detection import calculate_cluster_centroids
from utils.paths import DATA_ROOT, OUTPUT_ROOT

CKPT_DEFAULT = Path(__file__).parent.parent.parent / "data/sam2_checkpoints/sam2_hiera_tiny.pt"
CFG_DEFAULT  = "sam2_hiera_t"
CMAP = plt.get_cmap("tab10")


# ── Extrinsics 工具 ───────────────────────────────────────────────────────────

R_BASE = np.array([
    [ 0., -1.,  0.],
    [ 0.,  0., -1.],
    [ 1.,  0.,  0.],
], dtype=float)

def _Rx(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def _Ry(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def _Rz(d): c,s=np.cos(np.radians(d)),np.sin(np.radians(d)); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)

def build_RT(pitch=0., yaw=0., roll=0., height=1.3, forward=0., lateral=0.):
    R = _Rz(roll) @ _Ry(yaw) @ _Rx(pitch) @ R_BASE
    t = -R @ np.array([forward, lateral, height], float)
    return R, t

def project_pts(pts3d, K, R, t, img_w, img_h):
    if len(pts3d) == 0:
        return np.empty((0, 2)), np.zeros(len(pts3d), bool)
    pc = (R @ pts3d.T).T + t
    front = pc[:, 2] > 0.1
    uv_all = np.full((len(pts3d), 2), np.nan)
    if front.any():
        proj = (K @ pc[front].T).T
        uv_all[front] = proj[:, :2] / proj[:, 2:3]
    in_img = front & (uv_all[:,0]>=0) & (uv_all[:,0]<img_w) & \
                     (uv_all[:,1]>=0) & (uv_all[:,1]<img_h)
    return uv_all, in_img


# ── 資料載入 ─────────────────────────────────────────────────────────────────

def get_cluster_centroids(mask_dir, radar_dir, frame_names, idx):
    mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, frame_names[idx])
    mv_mask, _, mv_pc, _, _ = separate_static_and_moving(mask, radar_pc)
    centroids, _, _ = calculate_cluster_centroids(mv_mask, mv_pc)
    return centroids  # dict[int, np.ndarray(3,)]


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def _disable_default_keymaps():
    for k in ("zoom", "back", "forward", "save", "quit", "home"):
        plt.rcParams[f"keymap.{k}"] = []

def _pair_for_cluster(pairs: list[tuple[int, int]], cid: int) -> int | None:
    """回傳 cid 在 pairs 中對應的 mask index，無則 None。"""
    for c, m in pairs:
        if c == cid:
            return m
    return None


# ── 標注模式 ─────────────────────────────────────────────────────────────────

class FrameAnnotator:
    """SAM2 互動標注：逐台車點擊標記，Space 確認，Enter 完成。"""

    def __init__(self, img_rgb: np.ndarray, predictor: SAM2ImagePredictor,
                 frame_idx: int, cluster_centroids: dict[int, np.ndarray]):
        self.img = img_rgb
        self.predictor = predictor
        self.frame_idx = frame_idx
        self.h, self.w = img_rgb.shape[:2]
        self.cluster_centroids = cluster_centroids

        self.cur_fg: list[list[float]] = []
        self.cur_bg: list[list[float]] = []
        self.cur_mask: np.ndarray | None = None

        self.confirmed_masks: list[np.ndarray] = []
        self.centroids: list[np.ndarray] = []

        self.done = False
        self.skip = False

        _disable_default_keymaps()

        self.fig = plt.figure(figsize=(20, 8))
        self.fig.canvas.manager.set_window_title(
            f"[標注模式]  frame {frame_idx:04d}"
        )
        self.ax     = self.fig.add_axes([0.01, 0.08, 0.60, 0.88])
        self.ax_rad = self.fig.add_axes([0.65, 0.08, 0.33, 0.88])
        self.ax.axis("off")

        predictor.set_image(img_rgb)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self._redraw()
        plt.show(block=True)

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        u, v = float(event.xdata), float(event.ydata)
        if event.button == 1:
            self.cur_fg.append([u, v])
        elif event.button == 3:
            self.cur_bg.append([u, v])
        else:
            return
        self._run_sam2()
        self._redraw()

    def _on_key(self, event):
        if event.key in (" ", "c"):
            if self.cur_mask is not None:
                ys, xs = np.where(self.cur_mask)
                if len(xs):
                    self.confirmed_masks.append(self.cur_mask.copy())
                    self.centroids.append(np.array([xs.mean(), ys.mean()]))
            self.cur_fg.clear(); self.cur_bg.clear(); self.cur_mask = None
            self._redraw()

        elif event.key == "enter":
            if self.cur_mask is not None:
                ys, xs = np.where(self.cur_mask)
                if len(xs):
                    self.confirmed_masks.append(self.cur_mask.copy())
                    self.centroids.append(np.array([xs.mean(), ys.mean()]))
            self.done = True
            plt.close(self.fig)

        elif event.key in ("q", "escape"):
            self.skip = True; self.done = True
            plt.close(self.fig)

        elif event.key == "z":
            if self.cur_fg: self.cur_fg.pop()
            elif self.cur_bg: self.cur_bg.pop()
            self._run_sam2(); self._redraw()

    def _run_sam2(self):
        all_pts    = self.cur_fg + self.cur_bg
        all_labels = [1]*len(self.cur_fg) + [0]*len(self.cur_bg)
        if not all_pts:
            self.cur_mask = None; return
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(all_pts,    dtype=np.float32),
                point_labels=np.array(all_labels, dtype=np.int32),
                multimask_output=True,
            )
        self.cur_mask = masks[int(np.argmax(scores))].astype(bool)

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)
        overlay = np.zeros((*self.img.shape[:2], 4), dtype=float)

        for i, m in enumerate(self.confirmed_masks):
            c = np.array(CMAP(i % 10)[:3])
            overlay[m, :3] = c; overlay[m, 3] = 0.35
            ys, xs = np.where(m)
            if len(xs):
                self.ax.plot(xs.mean(), ys.mean(), "x", color=c, ms=12, mew=2.5)
                self.ax.text(xs.mean()+6, ys.mean()-6, f"M{i}", color=c,
                             fontsize=10, fontweight="bold",
                             bbox=dict(fc="white", alpha=0.6, pad=1, ec="none"))

        if self.cur_mask is not None:
            c = np.array(CMAP(len(self.confirmed_masks) % 10)[:3])
            overlay[self.cur_mask, :3] = c; overlay[self.cur_mask, 3] = 0.55
            ys, xs = np.where(self.cur_mask)
            if len(xs):
                self.ax.plot(xs.mean(), ys.mean(), "+", color=c, ms=16, mew=3)

        self.ax.imshow(overlay)
        for u, v in self.cur_fg:
            self.ax.plot(u, v, "o", color="lime",   ms=8, mew=1.5, mec="white")
        for u, v in self.cur_bg:
            self.ax.plot(u, v, "o", color="tomato", ms=8, mew=1.5, mec="white")

        self.ax.set_title(
            f"[標注] Frame {self.frame_idx:04d}  confirmed={len(self.confirmed_masks)}"
            f"  | LClick=fg  RClick=bg  Z=undo  Space=確認物件  Enter=完成  Q=放棄",
            fontsize=9,
        )
        self.ax.axis("off")

        self.ax_rad.clear()
        for cid, c3d in self.cluster_centroids.items():
            color = CMAP(cid % 10)
            self.ax_rad.plot(-c3d[1], c3d[0], "o", color=color,
                             ms=12, mew=1.5, mec="white", zorder=3)
            self.ax_rad.text(-c3d[1]+0.4, c3d[0]+0.4, f"C{cid}",
                             color=color, fontsize=10, fontweight="bold")
        self.ax_rad.set_xlim(-25, 25); self.ax_rad.set_ylim(0, 70)
        self.ax_rad.set_xlabel("−Y  (lateral, m)"); self.ax_rad.set_ylabel("X  (forward, m)")
        self.ax_rad.set_title(f"Radar top-down  |  {len(self.cluster_centroids)} clusters")
        self.ax_rad.grid(True, alpha=0.3); self.ax_rad.axvline(0, color="k", lw=0.8, ls="--")
        self.fig.canvas.draw_idle()


# ── 配對模式 ─────────────────────────────────────────────────────────────────

class ManualMatcher:
    """
    手動指定 SAM2 mask ↔ 雷達 cluster 的對應關係。

    影像左半：mask overlay（M0/M1...方塊）+ 淡色投影位置參考
    右半：雷達俯視，cluster 以三角顯示（C0/C1...），點擊選取後再點影像中的 mask 完成配對

    操作
    ----
    LClick 雷達三角 (C0/C1...)   選取 cluster（黃色高亮）
    LClick 影像方塊 (M0/M1...)   將選取的 cluster 配對至此 mask
    S                            跳過選取的 cluster（標記為紅色 X）
    Z                            撤銷最後一對配對
    Enter                        完成此幀
    Q / Esc                      跳過此幀
    """

    HIT_RADIUS       = 30   # px，mask 方塊點擊半徑（影像座標）
    HIT_RADIUS_RADAR = 2.0  # m，cluster 三角點擊半徑（雷達座標）

    def __init__(self, img_rgb: np.ndarray, frame_idx: int,
                 confirmed_masks: list[np.ndarray],
                 mask_centroids: list[np.ndarray],
                 cluster_centroids: dict[int, np.ndarray],
                 K: np.ndarray, R_init: np.ndarray, t_init: np.ndarray):
        self.img = img_rgb
        self.h, self.w = img_rgb.shape[:2]
        self.frame_idx = frame_idx
        self.confirmed_masks = confirmed_masks
        self.mask_centroids  = mask_centroids   # list of [u, v]
        self.cluster_centroids = cluster_centroids

        # 投影所有 cluster 到影像平面（用初始外參）
        cids  = list(cluster_centroids.keys())
        pts3d = np.array([cluster_centroids[c] for c in cids])
        uv_proj, in_img = project_pts(pts3d, K, R_init, t_init, self.w, self.h)
        self.cluster_uv: dict[int, np.ndarray | None] = {}
        for i, cid in enumerate(cids):
            self.cluster_uv[cid] = uv_proj[i] if in_img[i] else None

        # 狀態
        self.selected_cluster: int | None = None
        self.pairs: list[tuple[int, int]] = []   # [(cluster_id, mask_idx), ...]
        self.used_masks: set[int] = set()
        self.skipped_clusters: set[int] = set()
        self.done = False
        self.skip_frame = False

        _disable_default_keymaps()

        self.fig = plt.figure(figsize=(20, 8))
        self.fig.canvas.manager.set_window_title(
            f"[配對模式]  frame {frame_idx:04d}"
        )
        self.ax     = self.fig.add_axes([0.01, 0.08, 0.60, 0.88])
        self.ax_rad = self.fig.add_axes([0.65, 0.08, 0.33, 0.88])
        self.ax.axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self._redraw()
        plt.show(block=True)

    # ── 事件 ─────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.xdata is None:
            return
        x, y = event.xdata, event.ydata

        if event.inaxes is self.ax_rad:
            # 雷達圖：點擊 cluster 三角選取 / 取消選取
            for cid, c3d in self.cluster_centroids.items():
                if cid in self.skipped_clusters:
                    continue
                if _pair_for_cluster(self.pairs, cid) is not None:
                    continue
                rx, ry = -c3d[1], c3d[0]   # 雷達圖座標
                if np.hypot(x - rx, y - ry) < self.HIT_RADIUS_RADAR:
                    self.selected_cluster = None if self.selected_cluster == cid else cid
                    self._redraw()
                    return

        elif event.inaxes is self.ax:
            # 影像圖：點擊 mask 方塊（需已選取 cluster）
            if self.selected_cluster is not None:
                for mi, mc in enumerate(self.mask_centroids):
                    if mi in self.used_masks:
                        continue
                    if np.hypot(x - mc[0], y - mc[1]) < self.HIT_RADIUS:
                        self.pairs.append((self.selected_cluster, mi))
                        self.used_masks.add(mi)
                        self.selected_cluster = None
                        self._redraw()
                        return

    def _on_key(self, event):
        if event.key == "enter":
            self.done = True
            plt.close(self.fig)

        elif event.key in ("q", "escape"):
            self.skip_frame = True; self.done = True
            plt.close(self.fig)

        elif event.key == "s":
            if self.selected_cluster is not None:
                self.skipped_clusters.add(self.selected_cluster)
                self.selected_cluster = None
                self._redraw()

        elif event.key == "z":
            if self.pairs:
                _, mi = self.pairs.pop()
                self.used_masks.discard(mi)
                self._redraw()

    # ── 繪製 ─────────────────────────────────────────────────────────────────

    def _redraw(self):
        # ── 影像面板 ─────────────────────────────────────────────────────────
        self.ax.clear()
        self.ax.imshow(self.img)

        # mask overlay
        overlay = np.zeros((*self.img.shape[:2], 4), dtype=float)
        for i, m in enumerate(self.confirmed_masks):
            c = np.array(CMAP(i % 10)[:3])
            overlay[m, :3] = c; overlay[m, 3] = 0.25
        self.ax.imshow(overlay)

        # mask 中心（方塊，M0/M1...）
        for mi, mc in enumerate(self.mask_centroids):
            paired = mi in self.used_masks
            c = CMAP(mi % 10) if paired else "white"
            ec = CMAP(mi % 10)
            self.ax.plot(mc[0], mc[1], "s", color=c, ms=16,
                         mew=2.5, mec=ec, zorder=5)
            self.ax.text(mc[0]+10, mc[1]-10, f"M{mi}",
                         color=ec, fontsize=11, fontweight="bold",
                         bbox=dict(fc="white", alpha=0.75, pad=2, ec="none"), zorder=6)

        # 影像上顯示 cluster 投影位置（淡色參考，不可點擊）
        for cid, uv in self.cluster_uv.items():
            if uv is None:
                continue
            pair_mi = _pair_for_cluster(self.pairs, cid)
            if pair_mi is not None:
                # 已配對：顯示投影點 + 連線到 mask
                c = CMAP(pair_mi % 10)
                mc = self.mask_centroids[pair_mi]
                self.ax.plot(uv[0], uv[1], "^", color=c, ms=10,
                             mew=1.5, mec="white", alpha=0.8, zorder=4)
                self.ax.plot([uv[0], mc[0]], [uv[1], mc[1]],
                             "--", color=c, lw=1.5, alpha=0.5, zorder=3)
                self.ax.text(uv[0]+6, uv[1]-6, f"C{cid}", color=c, fontsize=9,
                             bbox=dict(fc="white", alpha=0.6, pad=1, ec="none"), zorder=5)
            elif cid == self.selected_cluster:
                # 已選取：顯示高亮投影點提示對應影像位置
                self.ax.plot(uv[0], uv[1], "^", color="yellow", ms=14,
                             mew=2, mec="darkorange", alpha=0.9, zorder=4)
                self.ax.text(uv[0]+6, uv[1]-6, f"C{cid}", color="darkorange", fontsize=9,
                             bbox=dict(fc="white", alpha=0.7, pad=1, ec="darkorange"), zorder=5)
            else:
                # 未配對：淡灰色小點
                self.ax.plot(uv[0], uv[1], "^", color="white", ms=8,
                             mew=1.5, mec="#888888", alpha=0.4, zorder=4)

        sel_str = (f"已選 C{self.selected_cluster}（雷達圖）→ 點影像中的 Mx 完成配對，或按 S 跳過"
                   if self.selected_cluster is not None
                   else "在雷達圖點擊三角選取 cluster")
        self.ax.set_title(
            f"[配對] Frame {self.frame_idx:04d}  "
            f"配對={len(self.pairs)}  跳過={len(self.skipped_clusters)}  |  "
            f"{sel_str}  |  Z=撤銷  Enter=完成  Q=skip幀",
            fontsize=9,
        )
        self.ax.axis("off")

        # ── 雷達面板（cluster 三角，可點擊） ─────────────────────────────────
        self.ax_rad.clear()
        for cid, c3d in self.cluster_centroids.items():
            rx, ry = -c3d[1], c3d[0]
            pair_mi = _pair_for_cluster(self.pairs, cid)
            skipped  = cid in self.skipped_clusters
            selected = cid == self.selected_cluster

            if skipped:
                self.ax_rad.plot(rx, ry, "x", color="red", ms=14, mew=3, zorder=4)
                self.ax_rad.text(rx+0.4, ry+0.4, f"C{cid}✕",
                                 color="red", fontsize=10, fontweight="bold")
            elif pair_mi is not None:
                c = CMAP(pair_mi % 10)
                self.ax_rad.plot(rx, ry, "^", color=c, ms=14,
                                 mew=1.5, mec="white", zorder=4)
                self.ax_rad.text(rx+0.4, ry+0.4, f"C{cid}→M{pair_mi}",
                                 color=c, fontsize=10, fontweight="bold")
            elif selected:
                self.ax_rad.plot(rx, ry, "^", color="yellow", ms=18,
                                 mew=2.5, mec="darkorange", zorder=5)
                self.ax_rad.text(rx+0.4, ry+0.4, f"C{cid} ◀",
                                 color="darkorange", fontsize=11, fontweight="bold")
            else:
                self.ax_rad.plot(rx, ry, "^", color="white", ms=14,
                                 mew=2, mec="#888888", zorder=3)
                self.ax_rad.text(rx+0.4, ry+0.4, f"C{cid}",
                                 color="#555555", fontsize=10, fontweight="bold")

        self.ax_rad.set_xlim(-25, 25); self.ax_rad.set_ylim(0, 70)
        self.ax_rad.set_xlabel("−Y  (lateral, m)"); self.ax_rad.set_ylabel("X  (forward, m)")
        self.ax_rad.set_title("Radar top-down  ▲=cluster（點選） | 灰=未配對 黃=選取 色=已配對 紅✕=跳過")
        self.ax_rad.grid(True, alpha=0.3); self.ax_rad.axvline(0, color="k", lw=0.8, ls="--")
        self.fig.canvas.draw_idle()

    # ── 結果 ─────────────────────────────────────────────────────────────────

    def get_correspondences(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """回傳 [(pts3d, pts2d), ...] 格式供 PnP 使用。"""
        return [
            (self.cluster_centroids[cid], self.mask_centroids[mi])
            for cid, mi in self.pairs
        ]


# ── PnP ──────────────────────────────────────────────────────────────────────

def solve_pnp(correspondences: list[tuple[np.ndarray, np.ndarray]],
              K: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(correspondences) < 4:
        print(f"[PnP] 對應點不足（{len(correspondences)} < 4），無法求解")
        return None
    pts3d = np.array([c[0] for c in correspondences], dtype=np.float64)
    pts2d = np.array([c[1] for c in correspondences], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        confidence=0.99, reprojectionError=20.0, iterationsCount=2000,
    )
    if not ok:
        print("[PnP] solvePnPRansac 失敗")
        return None
    n_in = len(inliers) if inliers is not None else 0
    print(f"[PnP] {n_in}/{len(pts3d)} inliers")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


# ── 驗證投影 ─────────────────────────────────────────────────────────────────

def show_reprojection(image_dir, frame_names, frame_indices,
                      mask_dir, radar_dir, K, R, t, img_w, img_h):
    n = len(frame_indices)
    fig, axes = plt.subplots(1, n, figsize=(10*n, 7))
    if n == 1:
        axes = [axes]
    for ax, fi in zip(axes, frame_indices):
        img_path = image_dir / f"{fi:04d}.jpg"
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        centroids = get_cluster_centroids(mask_dir, radar_dir, frame_names, fi)
        pts3d = np.array(list(centroids.values()))
        if len(pts3d):
            uv, in_img = project_pts(pts3d, K, R, t, img_w, img_h)
            for i, (cid, c3d) in enumerate(centroids.items()):
                if in_img[i]:
                    u, v = uv[i]
                    color = CMAP(cid % 10)
                    ax.plot(u, v, "o", color=color, ms=12, mew=1.5, mec="white")
                    ax.text(u+5, v-5, f"C{cid}", color=color, fontsize=9, fontweight="bold")
        ax.set_title(f"Reprojection  frame {fi:04d}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show(block=True)


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2 互動標注 + 手動配對 + PnP Extrinsics 擬合")
    parser.add_argument("--seq",    required=True, choices=["seq_1","seq_2","seq_3","seq_4"])
    parser.add_argument("--K",      required=True, help="intrinsics .npy 路徑")
    parser.add_argument("--ckpt",   default=str(CKPT_DEFAULT))
    parser.add_argument("--cfg",    default=CFG_DEFAULT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 200])
    parser.add_argument("--pitch",   type=float, default=0.0)
    parser.add_argument("--yaw",     type=float, default=0.0)
    parser.add_argument("--roll",    type=float, default=0.0)
    parser.add_argument("--height",  type=float, default=1.3)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--out",     default=None, help="輸出 .npz 路徑")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    seq_dir   = data_root / args.seq
    mask_dir  = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"
    image_dir = seq_dir / "image"

    K = np.load(args.K)
    sample_img = cv2.imread(str(image_dir / "0000.jpg"))
    img_h, img_w = sample_img.shape[:2]
    print(f"[K] f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  img={img_w}x{img_h}")

    R_init, t_init = build_RT(args.pitch, args.yaw, args.roll,
                               args.height, args.forward, args.lateral)
    print(f"[初始外參] pitch={args.pitch}°  height={args.height}m")

    print("\n[SAM2] 載入模型 ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"       device = {device}")
    sam2_model = build_sam2(args.cfg, args.ckpt, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    print("       OK")

    frame_names = list_frame_names(mask_dir)
    all_correspondences: list[tuple[np.ndarray, np.ndarray]] = []

    for fi in args.frames:
        if fi >= len(frame_names):
            print(f"[skip] frame {fi} 超出範圍（max={len(frame_names)-1}）")
            continue

        print(f"\n── Frame {fi:04d} ──────────────────────────────────")
        img_rgb = cv2.cvtColor(cv2.imread(str(image_dir / f"{fi:04d}.jpg")), cv2.COLOR_BGR2RGB)
        cluster_centroids = get_cluster_centroids(mask_dir, radar_dir, frame_names, fi)

        # 第一步：SAM2 標注
        annotator = FrameAnnotator(img_rgb, predictor, fi, cluster_centroids)
        if annotator.skip:
            print("  [quit] 使用者終止標注流程")
            break
        if not annotator.centroids:
            print("  [skip] 未標注任何 mask")
            continue

        print(f"  SAM2 masks: {len(annotator.centroids)} 個")

        # 第二步：手動配對
        matcher = ManualMatcher(
            img_rgb, fi,
            annotator.confirmed_masks,
            annotator.centroids,
            cluster_centroids,
            K, R_init, t_init,
        )
        if matcher.skip_frame:
            print("  [skip] 使用者跳過此幀配對")
            continue

        pairs = matcher.get_correspondences()
        print(f"  手動配對 {len(pairs)} 組：")
        for (cid, mi), (p3d, p2d) in zip(matcher.pairs, pairs):
            uv = matcher.cluster_uv.get(cid)
            print(f"    C{cid} → M{mi}  3D={p3d.round(3)}  "
                  f"proj={uv.round(1) if uv is not None else 'N/A'}  sam2={p2d.round(1)}")
        if matcher.skipped_clusters:
            print(f"  跳過 cluster：{sorted(matcher.skipped_clusters)}")

        all_correspondences.extend(pairs)

    # PnP
    print(f"\n[PnP] 共 {len(all_correspondences)} 組對應點")
    result = solve_pnp(all_correspondences, K)
    if result is None:
        print("[ERROR] PnP 求解失敗，請增加標注幀數")
        sys.exit(1)

    R_pnp, t_pnp = result
    print("[PnP 結果]")
    print(f"  R =\n{R_pnp}")
    print(f"  t = {t_pnp}")

    out_path = Path(args.out) if args.out else OUTPUT_ROOT / f"{args.seq}_extrinsics.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, R=R_pnp, t=t_pnp, K=K)
    print(f"[DONE] 儲存至 {out_path}")

    print("\n[驗證] 顯示 PnP 後的 cluster 投影 ...")
    show_reprojection(image_dir, frame_names, args.frames[:4],
                      mask_dir, radar_dir, K, R_pnp, t_pnp, img_w, img_h)


if __name__ == "__main__":
    main()
