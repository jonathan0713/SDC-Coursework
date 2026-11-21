import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

from utils.io import load_folder, load_pcd
from utils.viz import Visualizer


def load_poses(path):
    df = pd.read_csv(path)
    timestamps = df.iloc[:, 0].astype(str).tolist()
    poses = df.iloc[:, 1:].to_numpy(dtype=float).reshape(-1, 4, 4)
    return timestamps, [p.copy() for p in poses]


def transform_points(pts, pose):
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    return (pose[:3, :3] @ pts.T).T + pose[:3, 3]


def downsample_points(points, voxel_size):
    if points is None or len(points) == 0 or voxel_size <= 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)


def resolve_pose_path(current_dir, pose_arg):
    pose_path = Path(pose_arg)
    if pose_path.is_absolute():
        return pose_path
    return current_dir / pose_path


def main():
    parser = argparse.ArgumentParser(description="Replay raw PCD frames using a lidar_poses CSV.")
    parser.add_argument("--target", default="Track1", help="Dataset target folder under data/.")
    parser.add_argument("--poses", required=True, help="Pose CSV path, relative to lidar-slam/ or absolute.")
    parser.add_argument("--delay-ms", type=int, default=20, help="Viewer delay per frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit replay frames for quick checks.")
    parser.add_argument("--map-stride", type=int, default=5, help="Add every Nth scan to the gray map.")
    parser.add_argument("--map-voxel", type=float, default=0.5, help="Voxel size for accumulated map.")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "data"
    pose_path = resolve_pose_path(current_dir, args.poses)
    files = load_folder(data_dir / args.target / "data" / "raw_pcd")
    timestamps, poses = load_poses(pose_path)

    n = min(len(files), len(poses))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    files = files[:n]
    poses = poses[:n]

    print(f"Loaded {n} frames from target={args.target}")
    print(f"Using poses: {pose_path}")

    viz = Visualizer(delay_ms=args.delay_ms)
    viz.set_total_frames(n)

    global_map = np.zeros((0, 3))
    map_stride = max(args.map_stride, 1)

    for i, (file_path, pose) in enumerate(tqdm(zip(files, poses), total=n, desc="Replay")):
        if viz.should_quit:
            break

        pts = load_pcd(file_path)
        pts_w = transform_points(pts, pose)

        if i % map_stride == 0:
            global_map = np.concatenate([global_map, pts_w], axis=0)
            global_map = downsample_points(global_map, args.map_voxel)

        viz.update(pts_w, global_map, poses[:i + 1], [])


if __name__ == "__main__":
    main()
