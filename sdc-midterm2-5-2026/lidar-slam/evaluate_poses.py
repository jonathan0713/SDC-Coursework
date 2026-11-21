import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils.io import load_folder, load_pcd


def load_poses(path):
    df = pd.read_csv(path)
    timestamps = df.iloc[:, 0].astype(str).tolist()
    poses = df.iloc[:, 1:].to_numpy(dtype=float).reshape(-1, 4, 4)
    return timestamps, poses


def downsample(points, voxel_size):
    if voxel_size <= 0:
        return points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)


def transform_points(points, pose):
    return (pose[:3, :3] @ points[:, :3].T).T + pose[:3, 3]


def symmetric_nn_stats(a, b, overlap_thresh):
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan

    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1, workers=-1)
    tree_a = cKDTree(a)
    d_ba, _ = tree_a.query(b, k=1, workers=-1)
    d = np.concatenate([d_ab, d_ba])
    return float(np.median(d)), float(np.percentile(d, 90)), float(np.mean(d < overlap_thresh))


def trajectory_stats(poses):
    xyz = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    accel = np.abs(np.diff(steps)) if len(steps) > 1 else np.array([0.0])

    rel_rots = []
    for i in range(1, len(poses)):
        dR = poses[i, :3, :3] @ poses[i - 1, :3, :3].T
        rel_rots.append(R.from_matrix(dR).magnitude())
    rel_rots = np.asarray(rel_rots) if rel_rots else np.array([0.0])

    return {
        "frames": len(poses),
        "path_len": float(np.sum(steps)),
        "net_xy": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
        "z_range": float(np.ptp(xyz[:, 2])),
        "step_mean": float(np.mean(steps)),
        "step_p95": float(np.percentile(steps, 95)),
        "step_max": float(np.max(steps)),
        "accel_p95": float(np.percentile(accel, 95)),
        "rot_deg_p95": float(np.degrees(np.percentile(rel_rots, 95))),
    }


def evaluate_pose_file(pose_path, raw_files, args):
    _, poses = load_poses(pose_path)
    n = min(len(raw_files), len(poses))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    frame_indices = list(range(0, n, args.stride))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    clouds = {}
    for i in tqdm(frame_indices, desc=f"load {pose_path.name}", leave=False):
        pts = load_pcd(raw_files[i])
        pts = downsample(pts, args.voxel)
        clouds[i] = transform_points(pts, poses[i])

    medians = []
    p90s = []
    overlaps = []
    pairs = list(zip(frame_indices[:-1], frame_indices[1:]))
    for i, j in tqdm(pairs, desc=f"nn {pose_path.name}", leave=False):
        med, p90, overlap = symmetric_nn_stats(clouds[i], clouds[j], args.overlap_thresh)
        medians.append(med)
        p90s.append(p90)
        overlaps.append(overlap)

    stats = trajectory_stats(poses[:n])
    stats.update(
        {
            "file": pose_path.name,
            "nn_median": float(np.nanmedian(medians)),
            "nn_p90": float(np.nanmedian(p90s)),
            "overlap": float(np.nanmean(overlaps)),
        }
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Rank Track pose CSVs with local map-consistency proxies.")
    parser.add_argument("--target", default="Track1")
    parser.add_argument("--poses", nargs="+", default=["Track1/lidar_poses*.csv"])
    parser.add_argument("--voxel", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--overlap-thresh", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data" / args.target / "data" / "raw_pcd"
    raw_files = load_folder(data_dir)

    pose_paths = []
    for pattern in args.poses:
        matches = sorted(root.glob(pattern))
        if matches:
            pose_paths.extend(matches)
        else:
            pose_paths.append((root / pattern).resolve())
    pose_paths = sorted(set(p for p in pose_paths if p.exists()))

    rows = [evaluate_pose_file(path, raw_files, args) for path in pose_paths]
    df = pd.DataFrame(rows)

    columns = [
        "file",
        "frames",
        "nn_median",
        "nn_p90",
        "overlap",
        "path_len",
        "net_xy",
        "z_range",
        "step_p95",
        "step_max",
        "accel_p95",
        "rot_deg_p95",
    ]
    df = df[columns].sort_values(["nn_p90", "nn_median", "accel_p95"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
