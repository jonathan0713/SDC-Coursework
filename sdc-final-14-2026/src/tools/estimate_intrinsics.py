#!/usr/bin/env python3
"""
Estimate camera intrinsics from a monocular driving image sequence.

Two methods are available (--method):

  vp  — Vanishing-Point method (default, works for straight-road footage)
        Detects line segments via Canny + probabilistic Hough, finds their
        intersections, and clusters them into vanishing points.
        • Single forward VP  → refined principal point only
        • Two orthogonal VPs → focal length via f² = –(VP1–PP)·(VP2–PP)
        • With --lane-width + --camera-height → focal length from geometry

  fm  — Fundamental-Matrix method (works for footage with turns/lateral motion)
        ORB feature matching + RANSAC F, then autocalibration for focal length.
        Unreliable for pure straight-line driving.

Assumptions
-----------
- Pinhole camera, no lens distortion, square pixels (fx = fy = f).
- Principal point near image centre (refined by VP method).

Usage
-----
    python tools/estimate_intrinsics.py --seq seq_4
    python tools/estimate_intrinsics.py --seq seq_4 --method vp --lane-width 3.7 --camera-height 1.2
    python tools/estimate_intrinsics.py --seq seq_4 --method fm --step 3 --frames 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import DATA_ROOT, OUTPUT_ROOT


# ── Image loading ─────────────────────────────────────────────────────────────

def load_images(image_dir: Path, max_frames: int, step: int,
                gray: bool = True) -> list[np.ndarray]:
    paths = sorted(image_dir.glob("*.jpg"))
    selected = paths[::step][:max_frames]
    mode = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return [img for p in selected if (img := cv2.imread(str(p), mode)) is not None]


# ── Vanishing-point method ────────────────────────────────────────────────────

def _hough_segments(gray: np.ndarray) -> np.ndarray:
    """Return line segments detected via Canny + probabilistic Hough."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    segs = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=60, maxLineGap=15,
    )
    return segs.reshape(-1, 4) if segs is not None else np.empty((0, 4))


def _segment_vp_candidates(segs: np.ndarray, h: int, w: int,
                            angle_band: tuple[float, float] = (10, 80),
                            horizon_frac: float = 0.65,
                            ) -> list[tuple[float, float]]:
    """
    Intersect all pairs of line segments whose angle to horizontal is within
    `angle_band` degrees.  Keep intersections in the upper `horizon_frac`
    fraction of the image (where VPs plausibly lie for road scenes).
    """
    if len(segs) < 2:
        return []

    # Convert to line equations ax + by + c = 0
    lines = []
    for x1, y1, x2, y2 in segs:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        if angle_band[0] <= angle <= angle_band[1]:
            a, b, c = y2 - y1, -(x2 - x1), (x2 - x1) * y1 - (y2 - y1) * x1
            norm = np.hypot(a, b)
            if norm > 1e-6:
                lines.append((a / norm, b / norm, c / norm))

    pts: list[tuple[float, float]] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, c1 = lines[i]
            a2, b2, c2 = lines[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue  # parallel
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            if -w < x < 2 * w and 0 < y < horizon_frac * h:
                pts.append((x, y))
    return pts


def _cluster_vps(pts: list[tuple[float, float]],
                 bandwidth: float = 80.0) -> list[np.ndarray]:
    """
    Mean-shift-style clustering of intersection points.
    Returns cluster centres sorted by cluster size (largest first).
    """
    if not pts:
        return []
    arr = np.array(pts, dtype=float)
    labels = np.full(len(arr), -1, dtype=int)
    cluster_id = 0
    for i, p in enumerate(arr):
        if labels[i] != -1:
            continue
        dists = np.linalg.norm(arr - p, axis=1)
        mask = dists < bandwidth
        labels[mask] = cluster_id
        cluster_id += 1

    centres = []
    for cid in range(cluster_id):
        members = arr[labels == cid]
        centres.append((members.mean(axis=0), len(members)))
    centres.sort(key=lambda x: -x[1])
    return [c for c, _ in centres]


def estimate_vps(images: list[np.ndarray],
                 max_pts_per_frame: int = 300,
                 bandwidth: float = 60.0,
                 ) -> list[np.ndarray]:
    """
    Aggregate vanishing-point candidates over all frames, then cluster.
    Returns list of VP locations in pixel coordinates (sorted by support).
    """
    h, w = images[0].shape
    all_pts: list[tuple[float, float]] = []

    for img in images:
        segs = _hough_segments(img)
        pts = _segment_vp_candidates(segs, h, w)
        # Sub-sample to avoid one frame dominating
        if len(pts) > max_pts_per_frame:
            idx = np.random.choice(len(pts), max_pts_per_frame, replace=False)
            pts = [pts[i] for i in idx]
        all_pts.extend(pts)

    return _cluster_vps(all_pts, bandwidth=bandwidth)


def focal_from_two_vps(vp1: np.ndarray, vp2: np.ndarray,
                        cx: float, cy: float) -> float | None:
    """
    Orthogonality constraint: for two vanishing points of orthogonal directions
    in the same horizontal plane,
        f² = –(VP1 – PP) · (VP2 – PP)

    Returns None if the result is non-positive (VPs not orthogonal in 3D).
    """
    d1 = vp1 - np.array([cx, cy])
    d2 = vp2 - np.array([cx, cy])
    val = -float(d1 @ d2)
    return float(np.sqrt(val)) if val > 0 else None


def focal_from_lane_geometry(vp: np.ndarray, cy: float,
                              camera_height: float,
                              lane_width: float,
                              image_bottom_y: int,
                              lane_px_width_at_bottom: float,
                              ) -> float:
    """
    Given:
      vp         — forward vanishing point in pixels (x, y)
      cy         — principal-point y (pixels)
      camera_height — camera height above road (metres)
      lane_width    — actual lane width (metres)
      image_bottom_y — y-coordinate of the bottom of visible lane markings
      lane_px_width_at_bottom — pixel width of lane at that y

    Geometry (pinhole, camera looking at horizon):
      y_bottom = image_bottom_y,  y_vp = vp[1]
      distance to bottom strip:  d = f * camera_height / (image_bottom_y - vp[1])
      lane_px_width = f * lane_width / d
      → f² = lane_px_width * (image_bottom_y – vp_y) * camera_height / lane_width … (approx)
    """
    dy = float(image_bottom_y) - float(vp[1])
    if dy <= 0:
        return None
    # Solve: lane_px_width_at_bottom = f * lane_width / (f * camera_height / (f * dy / f))
    # Simplified (paraxial): f = lane_px_width_at_bottom * dy / (lane_width / camera_height * f)
    # Exact: f = sqrt(lane_px_width * dy / lane_width * camera_height)   [when vp_y << image_bottom_y]
    return float(np.sqrt(lane_px_width_at_bottom * dy * camera_height / lane_width))


# ── Fundamental-Matrix method ─────────────────────────────────────────────────

def compute_fundamental_matrices(images: list[np.ndarray],
                                  min_inliers: int = 60,
                                  ) -> list[np.ndarray]:
    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    Fs: list[np.ndarray] = []

    for i in range(len(images) - 1):
        kp1, des1 = orb.detectAndCompute(images[i], None)
        kp2, des2 = orb.detectAndCompute(images[i + 1], None)
        if des1 is None or des2 is None or len(kp1) < 8:
            continue
        raw = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < min_inliers:
            continue
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        if F is None or F.shape != (3, 3):
            continue
        if int(mask.ravel().sum()) >= min_inliers:
            Fs.append(F)
    return Fs


def focal_from_F(F: np.ndarray, cx: float, cy: float) -> float | None:
    """Search f s.t. E = K(f)^T F K(f) has two equal singular values."""
    def objective(f: float) -> float:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        E = K.T @ F @ K
        sv = sorted(np.linalg.svd(E, compute_uv=False), reverse=True)
        return abs(sv[0] - sv[1]) / (sv[0] + sv[1] + 1e-9)

    cands = np.linspace(200, 3000, 300)
    errs = [objective(f) for f in cands]
    coarse = float(cands[int(np.argmin(errs))])
    res = minimize_scalar(objective, bounds=(max(100.0, coarse * 0.6),
                                             min(5000.0, coarse * 1.6)),
                          method="bounded", options={"xatol": 0.5})
    return float(res.x) if res.fun < 0.08 else None


# ── Robust aggregation ────────────────────────────────────────────────────────

def robust_stats(values: np.ndarray, sigma: float = 2.5,
                 ) -> tuple[float, float, int]:
    """Return (mean, std, n_inliers) after MAD-based outlier rejection."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) + 1e-9
    inliers = values[np.abs(values - med) < sigma * mad]
    return float(np.mean(inliers)), float(np.std(inliers)), len(inliers)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics from monocular driving footage."
    )
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--method", choices=["vp", "fm", "both"], default="vp",
                        help="vp=vanishing-point (default), fm=fundamental-matrix, both=combine")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--step",   type=int, default=3)
    parser.add_argument("--lane-width",     type=float, default=None,
                        help="Known lane width in metres (e.g. 3.7). "
                             "Enables geometry-based focal length estimate.")
    parser.add_argument("--camera-height",  type=float, default=None,
                        help="Camera height above road in metres (e.g. 1.2).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    image_dir = data_root / args.seq / "image"
    if not image_dir.exists():
        print(f"[ERROR] image directory not found: {image_dir}")
        sys.exit(1)

    print(f"[1] Loading images  (step={args.step}, max={args.frames})")
    gray_imgs = load_images(image_dir, args.frames, args.step, gray=True)
    if len(gray_imgs) < 2:
        print("[ERROR] Need at least 2 images.")
        sys.exit(1)
    h, w = gray_imgs[0].shape
    cx, cy = w / 2.0, h / 2.0
    print(f"    {len(gray_imgs)} frames, {w}×{h} px, assumed cx={cx:.0f} cy={cy:.0f}")

    focal_estimates: list[float] = []
    vp_principal: np.ndarray | None = None

    # ── Vanishing-point method ────────────────────────────────────────────────
    if args.method in ("vp", "both"):
        print("\n[2] Vanishing-point detection ...")
        np.random.seed(42)
        vps = estimate_vps(gray_imgs)

        if not vps:
            print("    No vanishing points found.")
        else:
            print(f"    {len(vps)} VP cluster(s) found:")
            for i, vp in enumerate(vps[:4]):
                print(f"      VP{i}: ({vp[0]:.1f}, {vp[1]:.1f}) px")

            vp_principal = vps[0]  # most supported VP = forward direction
            print(f"    → Refined principal point: ({vp_principal[0]:.1f}, {vp_principal[1]:.1f})")

            # Try orthogonal VP pair for focal length
            if len(vps) >= 2:
                f_vp = focal_from_two_vps(vps[0], vps[1], cx, cy)
                if f_vp is not None:
                    print(f"    → f from orthogonal VPs: {f_vp:.1f} px")
                    focal_estimates.append(f_vp)
                else:
                    print("    → VP pair not orthogonal in 3D (no focal estimate)")

            # Geometry-based estimate if lane width + camera height provided
            if args.lane_width is not None and args.camera_height is not None:
                # Measure lane pixel width at bottom of image by examining
                # the spread of Hough lines at y = h-1
                # Approximation: use interquartile range of x-intercepts at y=h-1
                all_x_bottom: list[float] = []
                for img in gray_imgs[:20]:
                    segs = _hough_segments(img)
                    for x1, y1, x2, y2 in segs:
                        if abs(y2 - y1) > 10:
                            x_bot = x1 + (x2 - x1) * (h - 1 - y1) / (y2 - y1)
                            if 0 < x_bot < w:
                                all_x_bottom.append(x_bot)
                if len(all_x_bottom) >= 4:
                    q25, q75 = np.percentile(all_x_bottom, [25, 75])
                    lane_px = q75 - q25
                    f_geo = focal_from_lane_geometry(
                        vp_principal, cy, args.camera_height,
                        args.lane_width, h - 1, lane_px,
                    )
                    if f_geo:
                        print(f"    → f from lane geometry:   {f_geo:.1f} px  "
                              f"(lane_px≈{lane_px:.0f}, h={args.camera_height}m, w={args.lane_width}m)")
                        focal_estimates.append(f_geo)

    # ── Fundamental-matrix method ─────────────────────────────────────────────
    if args.method in ("fm", "both"):
        print("\n[3] Fundamental-Matrix autocalibration ...")
        Fs = compute_fundamental_matrices(gray_imgs)
        print(f"    {len(Fs)} valid F matrices")
        fm_estimates: list[float] = []
        for i, F in enumerate(Fs):
            f = focal_from_F(F, cx, cy)
            if f is not None:
                fm_estimates.append(f)
        if fm_estimates:
            f_fm, _, n = robust_stats(np.array(fm_estimates))
            print(f"    → f from FM method: {f_fm:.1f} px  ({n}/{len(fm_estimates)} inliers)")
            focal_estimates.append(f_fm)
        else:
            print("    → No reliable FM estimates (likely straight-line motion)")

    # ── Aggregate and output ──────────────────────────────────────────────────
    print()
    if not focal_estimates:
        print("[WARNING] Could not estimate focal length from data alone.")
        print("  Possible fixes:")
        print("    --method fm          (if footage has turns)")
        print("    --lane-width 3.7 --camera-height 1.2  (if lane width known)")
        print("  Falling back to 90° horizontal FoV assumption: f = W/2")
        final_f = w / 2.0
        f_std, n_inliers = 0.0, 0
    else:
        arr = np.array(focal_estimates)
        final_f, f_std, n_inliers = robust_stats(arr)

    # Use refined principal point from VP if available
    if vp_principal is not None:
        final_cx = float(vp_principal[0])
        final_cy = float(vp_principal[1])
        print(f"  Principal point refined by VP: ({final_cx:.1f}, {final_cy:.1f})")
    else:
        final_cx, final_cy = cx, cy

    K = np.array([
        [final_f, 0.0,      final_cx],
        [0.0,     final_f,  final_cy],
        [0.0,     0.0,      1.0     ],
    ])

    print("=" * 52)
    print(f"  Focal length  : {final_f:.1f} ± {f_std:.1f} px  ({n_inliers} estimates)")
    print(f"  Principal pt  : ({final_cx:.1f}, {final_cy:.1f})")
    print(f"  Horiz. FoV    : {2 * np.degrees(np.arctan(w / 2 / final_f)):.1f}°")
    print(f"  K =")
    for row in K:
        print(f"      [{row[0]:9.2f}  {row[1]:9.2f}  {row[2]:9.2f}]")
    print("=" * 52)

    out_path = Path(args.out) if args.out else OUTPUT_ROOT / f"{args.seq}_camera_K.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, K)
    print(f"[DONE] K saved → {out_path}")


if __name__ == "__main__":
    main()
