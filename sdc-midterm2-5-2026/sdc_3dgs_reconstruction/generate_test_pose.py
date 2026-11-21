import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import load_pose_file


def read_frame_list(path):
    with Path(path).open("r") as f:
        return [line.strip().split(".")[0] for line in f if line.strip()]


def interpolate_pose(target_ts, pose_ts, poses):
    pose_ts = np.asarray(pose_ts, dtype=np.int64)
    idx = np.searchsorted(pose_ts, target_ts)

    if idx < len(pose_ts) and pose_ts[idx] == target_ts:
        return poses[idx]
    if idx == 0 or idx >= len(pose_ts):
        raise ValueError(f"Timestamp {target_ts} is outside the known pose range")

    t0, t1 = pose_ts[idx - 1], pose_ts[idx]
    alpha = (target_ts - t0) / float(t1 - t0)

    p0 = poses[idx - 1][:3, 3]
    p1 = poses[idx][:3, 3]
    position = (1.0 - alpha) * p0 + alpha * p1

    rotations = R.from_matrix([poses[idx - 1][:3, :3], poses[idx][:3, :3]])
    rotation = Slerp([float(t0), float(t1)], rotations)(float(target_ts))

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = position
    return transform


def default_paths(track, data_root):
    script_dir = Path(__file__).resolve().parent
    return {
        "camera_poses": data_root / track / "result" / "camera_pose.csv",
        "frame_list": script_dir / track.lower() / f"{track.lower()}_test_frame_list.txt",
        "output": script_dir / track.lower() / "test_pose_list.txt",
    }


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interpolate camera poses for official test frames.")
    parser.add_argument("--track", default="Track2", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--camera-poses", type=Path)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = default_paths(args.track, args.data_root)
    camera_poses_path = args.camera_poses or paths["camera_poses"]
    frame_list_path = args.frame_list or paths["frame_list"]
    output_path = args.output or paths["output"]

    poses, pose_ts = load_pose_file(camera_poses_path)
    if pose_ts is None:
        raise ValueError(f"{camera_poses_path} must include timestamps for interpolation")

    order = np.argsort(pose_ts)
    pose_ts = pose_ts[order]
    poses = [poses[i] for i in order]

    frame_ids = read_frame_list(frame_list_path)
    results = []
    skipped = []

    print(f"Interpolating poses for {len(frame_ids)} {args.track} test frames.")
    for frame_id in frame_ids:
        timestamp = int(frame_id)
        try:
            results.append(interpolate_pose(timestamp, pose_ts, poses).reshape(-1))
        except ValueError as exc:
            skipped.append((frame_id, str(exc)))

    if skipped:
        for frame_id, reason in skipped[:5]:
            print(f"[Warning] Skipped {frame_id}: {reason}")
        raise RuntimeError(f"Could not interpolate {len(skipped)} test poses")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.asarray(results), fmt="%.18e")
    print(f"Saved {len(results)} test poses to {output_path}")


if __name__ == "__main__":
    main()
