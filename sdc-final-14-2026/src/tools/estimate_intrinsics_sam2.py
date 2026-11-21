#!/usr/bin/env python3
"""
使用 SAM2 標注 + Cluster 對應點，以最小二乘法估算相機內參（f, cx, cy）。

外參（R, t）固定不動，只求解內參矩陣 K = [[f,0,cx],[0,f,cy],[0,0,1]]。

原理
----
設第 i 組對應點（3D 雷達 cluster p3d_i，2D mask centroid [u_i, v_i]）：

    p_cam = R @ p3d + t
    x_n   = p_cam[0] / p_cam[2]     (歸一化 x)
    y_n   = p_cam[1] / p_cam[2]     (歸一化 y)

    u = f * x_n + cx    →  [x_n, 1, 0] @ [f, cx, cy]ᵀ = u
    v = f * y_n + cy    →  [y_n, 0, 1] @ [f, cx, cy]ᵀ = v

對所有對應點拼成線性系統 A @ [f, cx, cy]ᵀ = b，以 lstsq 求解。

流程
----
1. 在影像上點選車輛 → SAM2 產生 mask → 取 centroid（2D）
2. 手動配對雷達 cluster（3D）與影像 centroid（2D）
3. 收集跨幀對應點後求解 f, cx, cy
4. 顯示更新前後的重投影誤差並存檔

用法
----
    python tools/estimate_intrinsics_sam2.py \\
        --seq seq_4 \\
        --extrinsics data/outputs/seq_4_extrinsics.npz \\
        --frames 10 20 30 \\
        --out data/outputs/seq_4_camera_K.npy

標注模式操作（同 annotate_extrinsics_sam2.py）
----------------------------------------------
    左鍵          新增前景點，SAM2 自動生成 mask
    右鍵          新增背景點，排除雜訊
    Space / C     確認當前 mask，準備標注下一台車
    Z             撤銷當前物件最後一個點
    Enter         完成標注，進入配對模式
    Q / Esc       放棄此幀

配對模式操作
------------
    左鍵點 cluster 三角 (C0/C1...)    選取 cluster
    左鍵點 mask 方塊 (M0/M1...)       將 cluster 配對至此 mask
    S                                 跳過選取的 cluster
    Z                                 撤銷最後一對配對
    Enter                             完成此幀
    Q / Esc                           跳過此幀
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sdc_tracking_utils import list_frame_names
from utils.paths import DATA_ROOT, OUTPUT_ROOT

# 直接從外參標注腳本匯入 UI 類別與工具函式，避免重複實作
from annotate_extrinsics_sam2 import (
    CKPT_DEFAULT, CFG_DEFAULT, CMAP,
    FrameAnnotator, ManualMatcher,
    get_cluster_centroids, project_pts,
    show_reprojection,
)


# ── 最小二乘求解內參 ──────────────────────────────────────────────────────────

def solve_intrinsics(
    correspondences: list[tuple[np.ndarray, np.ndarray]],
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """
    最小二乘求解 f, cx, cy。

    Parameters
    ----------
    correspondences : list of (p3d, uv)
        p3d : (3,) 雷達座標；uv : (2,) 影像像素座標
    R, t : 固定外參

    Returns
    -------
    K_new : (3, 3) 解出的內參矩陣
    errors : 每個對應點的重投影誤差（像素）
    """
    rows_A: list[list[float]] = []
    rows_b: list[float] = []

    valid: list[tuple[np.ndarray, np.ndarray]] = []
    for p3d, uv in correspondences:
        p_cam = R @ p3d + t
        if p_cam[2] < 0.1:
            print(f"  [skip] 負深度 p_cam={p_cam}")
            continue
        x_n = p_cam[0] / p_cam[2]
        y_n = p_cam[1] / p_cam[2]
        u, v = float(uv[0]), float(uv[1])
        rows_A.extend([[x_n, 1.0, 0.0], [y_n, 0.0, 1.0]])
        rows_b.extend([u, v])
        valid.append((p3d, uv))

    if len(valid) < 2:
        raise ValueError(f"有效對應點不足（{len(valid)} < 2），無法求解內參")

    A = np.array(rows_A, dtype=float)
    b = np.array(rows_b, dtype=float)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    f, cx, cy = params

    K_new = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    # 計算每點重投影誤差
    errors: list[float] = []
    for p3d, uv in valid:
        p_cam = R @ p3d + t
        proj  = K_new @ p_cam
        u_hat = proj[0] / proj[2]
        v_hat = proj[1] / proj[2]
        errors.append(float(np.hypot(u_hat - uv[0], v_hat - uv[1])))

    return K_new, errors


# ── 驗證視窗（前後對比）────────────────────────────────────────────────────────

def show_comparison(
    correspondences: list[tuple[np.ndarray, np.ndarray]],
    K_old: np.ndarray, K_new: np.ndarray,
    R: np.ndarray, t: np.ndarray,
    errors_new: list[float],
) -> None:
    fig, (ax_old, ax_new) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("內參更新前後重投影誤差對比", fontsize=12)

    for ax, K, label in [(ax_old, K_old, "Before"), (ax_new, K_new, "After")]:
        us_gt, vs_gt = [], []
        us_pr, vs_pr = [], []
        for p3d, uv in correspondences:
            p_cam = R @ p3d + t
            if p_cam[2] < 0.1:
                continue
            proj  = K @ p_cam
            u_hat = proj[0] / proj[2]
            v_hat = proj[1] / proj[2]
            us_gt.append(uv[0]); vs_gt.append(uv[1])
            us_pr.append(u_hat);  vs_pr.append(v_hat)

        ax.scatter(us_gt, vs_gt, c="limegreen", s=60, zorder=5, label="SAM2 mask centroid")
        ax.scatter(us_pr, vs_pr, c="tomato",    s=60, marker="x", zorder=5, label="Projected cluster")
        for ug, vg, up, vp in zip(us_gt, vs_gt, us_pr, vs_pr):
            ax.plot([ug, up], [vg, vp], "gray", lw=0.8, alpha=0.6)

        errs = []
        for ug, vg, up, vp in zip(us_gt, vs_gt, us_pr, vs_pr):
            errs.append(np.hypot(ug - up, vg - vp))
        mean_e = np.mean(errs) if errs else 0.0

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title(
            f"{label}  f={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n"
            f"mean error = {mean_e:.2f} px",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show(block=True)


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM2 標注 + Cluster 對應點最小二乘求解內參"
    )
    parser.add_argument("--seq",         required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--extrinsics",  required=True, help="extrinsics .npz 路徑（含 R, t, K）")
    parser.add_argument("--frames",      type=int, nargs="+", default=[10, 30, 60],
                        help="要標注的幀索引（預設 10 30 60）")
    parser.add_argument("--out",         default=None,
                        help="輸出 K.npy 路徑（預設 data/outputs/<seq>_camera_K.npy）")
    parser.add_argument("--ckpt",        default=str(CKPT_DEFAULT))
    parser.add_argument("--cfg",         default=CFG_DEFAULT)
    parser.add_argument("--no-save-ext", action="store_true",
                        help="不更新 extrinsics.npz 裡的 K（只存 K.npy）")
    args = parser.parse_args()

    ext_path = Path(args.extrinsics)
    ext      = np.load(ext_path)
    R, t     = ext["R"].astype(float), ext["t"].astype(float)
    K_init   = ext["K"].astype(float) if "K" in ext else np.eye(3)
    print(f"[extrinsics] {ext_path}")
    print(f"  R=\n{R}")
    print(f"  t={t}")
    print(f"  K_init  f={K_init[0,0]:.1f}  cx={K_init[0,2]:.1f}  cy={K_init[1,2]:.1f}")

    seq_dir   = DATA_ROOT / args.seq
    mask_dir  = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"
    image_dir = seq_dir / "image"

    sample_img = cv2.imread(str(image_dir / "0000.jpg"))
    if sample_img is None:
        sys.exit(f"[error] 找不到影像：{image_dir / '0000.jpg'}")
    img_h, img_w = sample_img.shape[:2]

    frame_names = list_frame_names(mask_dir)

    # ── 載入 SAM2 ─────────────────────────────────────────────────────────────
    print(f"\n[SAM2] ckpt={args.ckpt}  cfg={args.cfg}")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    sam2_model = build_sam2(args.cfg, args.ckpt, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    print(f"  device={device}")

    # ── 逐幀標注 + 配對 ────────────────────────────────────────────────────────
    all_correspondences: list[tuple[np.ndarray, np.ndarray]] = []

    for fi in args.frames:
        if fi >= len(frame_names):
            print(f"[skip] frame {fi} 超出序列長度 {len(frame_names)}")
            continue

        img_bgr = cv2.imread(str(image_dir / f"{fi:04d}.jpg"))
        if img_bgr is None:
            print(f"[skip] 找不到影像 frame {fi:04d}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cluster_centroids = get_cluster_centroids(mask_dir, radar_dir, frame_names, fi)
        if not cluster_centroids:
            print(f"[skip] frame {fi:04d} 無 cluster")
            continue

        print(f"\n── frame {fi:04d} ─────────────────────────────")
        print(f"  clusters: {list(cluster_centroids.keys())}")

        # Phase 1: SAM2 標注 → mask centroids
        annotator = FrameAnnotator(img_rgb, predictor, fi, cluster_centroids)
        if annotator.skip or not annotator.centroids:
            print("  [skip] 標注放棄或無 mask")
            continue

        print(f"  masks confirmed: {len(annotator.centroids)}")

        # Phase 2: 手動配對 cluster ↔ mask
        matcher = ManualMatcher(
            img_rgb, fi,
            annotator.confirmed_masks, annotator.centroids,
            cluster_centroids,
            K_init, R, t,
        )
        if matcher.skip_frame or not matcher.pairs:
            print("  [skip] 配對放棄或無有效配對")
            continue

        # 收集此幀對應點
        frame_corr = 0
        for cid, mid in matcher.pairs:
            p3d = cluster_centroids[cid]
            uv  = annotator.centroids[mid]
            all_correspondences.append((p3d, uv))
            print(f"  C{cid} → M{mid}  cluster=({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})  "
                  f"uv=({uv[0]:.1f}, {uv[1]:.1f})")
            frame_corr += 1
        print(f"  本幀新增 {frame_corr} 對應點")

    # ── 求解內參 ──────────────────────────────────────────────────────────────
    print(f"\n[LS] 共 {len(all_correspondences)} 組對應點")
    if len(all_correspondences) < 2:
        sys.exit("[error] 對應點不足，至少需要 2 組才能求解")

    K_new, errors = solve_intrinsics(all_correspondences, R, t)

    print(f"\n[結果]")
    print(f"  舊 K:  f={K_init[0,0]:.2f}  cx={K_init[0,2]:.2f}  cy={K_init[1,2]:.2f}")
    print(f"  新 K:  f={K_new[0,0]:.2f}  cx={K_new[0,2]:.2f}  cy={K_new[1,2]:.2f}")
    print(f"  重投影誤差（像素）：")
    for i, e in enumerate(errors):
        print(f"    對應點 {i:02d}: {e:.2f} px")
    print(f"  mean={np.mean(errors):.2f} px  max={np.max(errors):.2f} px")

    # ── 前後對比視覺化 ────────────────────────────────────────────────────────
    show_comparison(all_correspondences, K_init, K_new, R, t, errors)

    # 驗證新 K 的全圖投影
    show_reprojection(
        image_dir, frame_names, args.frames[:3],
        mask_dir, radar_dir,
        K_new, R, t, img_w, img_h,
    )

    # ── 存檔 ──────────────────────────────────────────────────────────────────
    out_k = Path(args.out) if args.out else OUTPUT_ROOT / f"{args.seq}_camera_K.npy"
    out_k.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_k), K_new)
    print(f"\n[saved] K.npy → {out_k}")

    if not args.no_save_ext:
        # 就地更新 extrinsics.npz 裡的 K（保留 R, t）
        np.savez(str(ext_path), R=R, t=t, K=K_new)
        print(f"[saved] extrinsics K updated → {ext_path}")


if __name__ == "__main__":
    main()
