import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


POSE_COLUMNS = [
    "m00",
    "m01",
    "m02",
    "m03",
    "m10",
    "m11",
    "m12",
    "m13",
    "m20",
    "m21",
    "m22",
    "m23",
    "m30",
    "m31",
    "m32",
    "m33",
]


def make_transform(x, y, z, qx, qy, qz, qw):
    """Create a 4x4 transform from translation and quaternion."""
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    transform[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return transform


def get_extrinsic_matrix(track):
    """Return T_lidar_camera for the requested track."""
    track = track.lower()

    if track == "track3":
        t_lidar_camera = make_transform(
            -0.170, -0.115, -0.100, -0.7071, 0.0, 0.0, 0.7071
        )

        pitch = R.from_euler("x", np.radians(7.0)).as_matrix()
        yaw = R.from_euler("z", np.radians(-3.25)).as_matrix()

        adjusted = np.eye(4)
        adjusted[:3, :3] = t_lidar_camera[:3, :3] @ yaw @ pitch
        adjusted[:3, 3] = t_lidar_camera[:3, 3]
        return adjusted

    if track == "track2":
        t_top_to_30f = make_transform(
            0.079, -0.082, -0.112, -0.502, 0.506, -0.501, 0.490
        )
        t_30f_to_60f = make_transform(
            -0.162, -0.003, 0.003, -0.001, -0.002, -0.000, 1.000
        )
        return t_top_to_30f @ t_30f_to_60f

    if track != "track1":
        raise ValueError(f"Unsupported track: {track}")

    t_lidar_camera = make_transform(
        -0.190, -0.115, 0.070, -0.7071, 0.0, 0.0, 0.7071
    )

    pitch = R.from_euler("x", np.radians(6.5)).as_matrix()
    yaw = R.from_euler("z", np.radians(-3.25)).as_matrix()

    adjusted = np.eye(4)
    adjusted[:3, :3] = yaw @ t_lidar_camera[:3, :3] @ pitch
    adjusted[:3, 3] = t_lidar_camera[:3, 3]
    return adjusted


def load_pose_csv(file_path):
    """
    Load a pose CSV in either matrix or quaternion format.

    Supported formats:
    - timestamp,m00,...,m33
    - id,timestamp,x,y,z,qx,qy,qz,qw
    - timestamp,x,y,z,qx,qy,qz,qw
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Pose CSV is empty: {file_path}")

    timestamp_names = {"timestamp", "timestamps", "ts", "time"}
    ts_col = next((col for col in df.columns if col.lower() in timestamp_names), None)
    if ts_col is None:
        ts_col = df.columns[0]

    timestamps = df[ts_col].astype(np.int64).to_numpy()
    poses = []

    if all(col in df.columns for col in POSE_COLUMNS):
        matrix_values = df[POSE_COLUMNS].to_numpy(dtype=float)
        poses = [row.reshape(4, 4) for row in matrix_values]
    else:
        quat_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if all(col in df.columns for col in quat_cols):
            quat_values = df[quat_cols].to_numpy(dtype=float)
        elif df.shape[1] == 9:
            quat_values = df.iloc[:, 2:].to_numpy(dtype=float)
        elif df.shape[1] == 8:
            quat_values = df.iloc[:, 1:].to_numpy(dtype=float)
        else:
            raise ValueError(
                f"Unknown pose CSV format with {df.shape[1]} columns: {file_path}"
            )

        for row in quat_values:
            transform = np.eye(4)
            transform[:3, 3] = row[:3]
            transform[:3, :3] = R.from_quat(row[3:]).as_matrix()
            poses.append(transform)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    poses = [poses[i] for i in order]
    return poses, timestamps


def interpolate_pose(target_ts, lidar_ts, lidar_poses):
    """Interpolate translation linearly and rotation with SLERP."""
    lidar_ts = np.asarray(lidar_ts, dtype=np.int64)
    idx = np.searchsorted(lidar_ts, target_ts)

    if idx < len(lidar_ts) and lidar_ts[idx] == target_ts:
        return lidar_poses[idx].copy()

    if idx == 0 or idx >= len(lidar_ts):
        return None

    t0, t1 = lidar_ts[idx - 1], lidar_ts[idx]
    if t0 == t1:
        return lidar_poses[idx].copy()

    alpha = (target_ts - t0) / float(t1 - t0)
    p0 = lidar_poses[idx - 1][:3, 3]
    p1 = lidar_poses[idx][:3, 3]
    position = (1.0 - alpha) * p0 + alpha * p1

    rotations = R.from_matrix(
        [lidar_poses[idx - 1][:3, :3], lidar_poses[idx][:3, :3]]
    )
    rotation = Slerp([float(t0), float(t1)], rotations)(float(target_ts))

    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = position
    return transform


def default_paths(track, data_root):
    process_dir = Path(__file__).resolve().parent
    slam_dir = process_dir.parent

    lidar_candidates = [
        data_root / track / "result" / "lidar_poses.csv",
        data_root / track / "lidar_poses.csv",
        slam_dir / track / "lidar_poses.csv",
    ]

    return {
        "image_dir": data_root / track / "data" / "image",
        "lidar_poses": next((p for p in lidar_candidates if p.exists()), lidar_candidates[0]),
        "output": data_root / track / "result" / "camera_pose.csv",
    }


def parse_args():
    process_dir = Path(__file__).resolve().parent
    repo_dir = process_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Interpolate LiDAR poses to image timestamps and export camera poses."
    )
    parser.add_argument("--track", default="Track1", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--lidar-poses", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = default_paths(args.track, args.data_root)

    lidar_pose_file = args.lidar_poses or paths["lidar_poses"]
    image_dir = args.image_dir or paths["image_dir"]
    output_file = args.output or paths["output"]

    lidar_poses, lidar_ts = load_pose_csv(lidar_pose_file)
    image_files = sorted(image_dir.glob("*.jpg"))
    image_ts = [np.int64(path.stem) for path in image_files if path.stem.isdigit()]

    if len(image_ts) == 0:
        raise FileNotFoundError(f"No timestamped JPG images found in {image_dir}")

    t_lidar_camera = get_extrinsic_matrix(args.track)
    rows = []
    skipped = 0

    print(
        f"Calculating {args.track} camera poses for {len(image_ts)} images "
        f"from {len(lidar_ts)} LiDAR poses."
    )

    for timestamp in image_ts:
        t_world_lidar = interpolate_pose(timestamp, lidar_ts, lidar_poses)
        if t_world_lidar is None:
            skipped += 1
            continue

        t_world_camera = t_world_lidar @ t_lidar_camera
        rows.append([int(timestamp)] + t_world_camera.reshape(-1).tolist())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows, columns=["timestamp"] + POSE_COLUMNS)
    df_out["timestamp"] = df_out["timestamp"].astype(np.int64)
    df_out.to_csv(output_file, index=False, float_format="%.10f")

    print(f"Saved {len(rows)} camera poses to {output_file}.")
    if skipped:
        print(f"Skipped {skipped} images outside the LiDAR pose time range.")


if __name__ == "__main__":
    main()
