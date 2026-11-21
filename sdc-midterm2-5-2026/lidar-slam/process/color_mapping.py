import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


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

INTRINSICS = {
    "Track1": {
        "width": 640,
        "height": 480,
        "fx": 653.143433778113,
        "fy": 657.670567367976,
        "cx": 299.1738577337179,
        "cy": 236.60674857178367,
    },
    "Track2": {
        "width": 1440,
        "height": 928,
        "fx": 1040.18078,
        "fy": 1038.55506,
        "cx": 720.04463,
        "cy": 464.33648,
    },
    "Track3": {
        "width": 960,
        "height": 720,
        "fx": 979.71515067,
        "fy": 986.50585105,
        "cx": 448.7607866,
        "cy": 354.91012286,
    },
}


def camera_matrix(track):
    intr = INTRINSICS[track]
    return np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=float,
    )


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


def filter_lidar_points(points, min_range=0.0, max_range=0.0):
    if points.size == 0:
        return points

    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = np.ones(points.shape[0], dtype=bool)
    if min_range > 0:
        mask &= distances >= min_range
    if max_range > 0:
        mask &= distances <= max_range
    return points[mask, :3]


def build_map(lidar_poses, pose_ts, pcd_dir, voxel_size=0.2, min_range=0.0, max_range=0.0):
    """Build a global point cloud by matching each PCD timestamp to the closest pose."""
    pcd_dir = Path(pcd_dir)
    pcd_files = sorted(pcd_dir.glob("*.pcd"))
    if not pcd_files:
        raise FileNotFoundError(f"No PCD files found in {pcd_dir}")

    pose_ts = np.asarray(pose_ts, dtype=np.int64)
    if len(pose_ts) == 0:
        raise ValueError("No LiDAR poses were loaded.")

    combined_pcd = o3d.geometry.PointCloud()
    print(f"Merging {len(pcd_files)} PCD frames from {pcd_dir}.")

    for pcd_path in tqdm(pcd_files):
        if not pcd_path.stem.isdigit():
            continue

        file_ts = np.int64(pcd_path.stem)
        closest_idx = int(np.argmin(np.abs(pose_ts - file_ts)))

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        if points.size == 0:
            continue

        points = filter_lidar_points(points, min_range=min_range, max_range=max_range)
        if points.size == 0:
            continue

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.transform(lidar_poses[closest_idx])
        combined_pcd += pcd

    if len(combined_pcd.points) == 0:
        raise ValueError("The merged point cloud is empty.")

    print(f"Downsampling merged map with voxel size {voxel_size}.")
    return combined_pcd.voxel_down_sample(voxel_size=voxel_size)


def colorize_map(
    pcd,
    image_dir,
    image_poses,
    image_pose_ts,
    intrinsics,
    z_tolerance=1.05,
    default_color=(0.2, 0.2, 0.2),
):
    """Project image colors onto the point cloud with per-image z-buffering."""
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        print(f"Skipping colorization: no JPG images found in {image_dir}.")
        return pcd

    points = np.asarray(pcd.points)
    n_points = points.shape[0]
    if n_points == 0:
        return pcd

    pose_by_ts = {int(ts): pose for ts, pose in zip(image_pose_ts, image_poses)}
    points_h = np.hstack((points, np.ones((n_points, 1))))
    color_sum = np.zeros((n_points, 3), dtype=float)
    color_counts = np.zeros(n_points, dtype=float)

    print(f"Colorizing point cloud with {len(image_files)} images.")

    for img_path in tqdm(image_files):
        if not img_path.stem.isdigit():
            continue

        pose = pose_by_ts.get(int(img_path.stem))
        if pose is None:
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        height, width = image.shape[:2]

        t_camera_world = np.linalg.inv(pose)
        points_cam = (t_camera_world @ points_h.T).T[:, :3]

        valid_z = points_cam[:, 2] > 0.1
        if not np.any(valid_z):
            continue

        valid_indices = np.where(valid_z)[0]
        projected = (intrinsics @ points_cam[valid_z].T).T
        z = projected[:, 2]
        u = np.floor(projected[:, 0] / z).astype(np.int32)
        v = np.floor(projected[:, 1] / z).astype(np.int32)

        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_image):
            continue

        valid_indices = valid_indices[in_image]
        u = u[in_image]
        v = v[in_image]
        z = z[in_image]

        depth_buffer = np.full((height, width), np.inf, dtype=float)
        np.minimum.at(depth_buffer, (v, u), z)

        visible = z <= depth_buffer[v, u] * z_tolerance
        if not np.any(visible):
            continue

        visible_indices = valid_indices[visible]
        color_sum[visible_indices] += image[v[visible], u[visible]]
        color_counts[visible_indices] += 1.0

    colors = np.tile(np.asarray(default_color, dtype=float), (n_points, 1))
    colored = color_counts > 0
    colors[colored] = color_sum[colored] / color_counts[colored, None]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Colored {int(np.count_nonzero(colored))}/{n_points} map points.")
    return pcd


def draw_camera_trajectory_and_axes(poses, step=20, scale=0.5):
    if not poses:
        return []

    points = [pose[:3, 3] for pose in poses]
    geometries = []

    if len(points) >= 2:
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in lines])
        geometries.append(line_set)

    for i in range(0, len(poses), max(1, step)):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        axis.transform(poses[i])
        geometries.append(axis)

    return geometries


def default_paths(track, data_root):
    process_dir = Path(__file__).resolve().parent
    slam_dir = process_dir.parent

    lidar_candidates = [
        data_root / track / "result" / "lidar_poses.csv",
        data_root / track / "lidar_poses.csv",
        slam_dir / track / "lidar_poses.csv",
    ]
    camera_candidates = [
        data_root / track / "result" / "camera_pose.csv",
        data_root / track / "camera_pose.csv",
        slam_dir / track / "camera_pose.csv",
    ]

    return {
        "pcd_dir": data_root / track / "data" / "raw_pcd",
        "image_dir": data_root / track / "data" / "image",
        "lidar_poses": next((p for p in lidar_candidates if p.exists()), lidar_candidates[0]),
        "camera_poses": next(
            (p for p in camera_candidates if p.exists()), camera_candidates[0]
        ),
        "output": data_root / track / "result" / f"{track}.pcd",
    }


def parse_args():
    process_dir = Path(__file__).resolve().parent
    repo_dir = process_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Build a LiDAR map and color it by projecting camera images."
    )
    parser.add_argument("--track", default="Track1", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--lidar-poses", type=Path)
    parser.add_argument("--camera-poses", type=Path)
    parser.add_argument("--pcd-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--map-voxel", type=float, default=0.1)
    parser.add_argument("--final-voxel", type=float, default=0.2)
    parser.add_argument("--min-range", type=float, default=0.0)
    parser.add_argument("--max-range", type=float, default=0.0)
    parser.add_argument("--outlier-neighbors", type=int, default=80)
    parser.add_argument("--outlier-std", type=float, default=2.0)
    parser.add_argument("--z-tolerance", type=float, default=1.05)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--camera-step", type=int, default=20)
    parser.add_argument("--camera-axis-scale", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = default_paths(args.track, args.data_root)

    lidar_pose_path = args.lidar_poses or paths["lidar_poses"]
    camera_pose_path = args.camera_poses or paths["camera_poses"]
    pcd_dir = args.pcd_dir or paths["pcd_dir"]
    image_dir = args.image_dir or paths["image_dir"]
    output_path = args.output or paths["output"]

    lidar_poses, lidar_ts = load_pose_csv(lidar_pose_path)
    full_map = build_map(
        lidar_poses,
        lidar_ts,
        pcd_dir,
        voxel_size=args.map_voxel,
        min_range=args.min_range,
        max_range=args.max_range,
    )

    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.raw_output), full_map)

    print("Removing statistical outliers.")
    full_map, _ = full_map.remove_statistical_outlier(
        nb_neighbors=args.outlier_neighbors,
        std_ratio=args.outlier_std,
    )
    full_map = full_map.voxel_down_sample(voxel_size=args.final_voxel)

    camera_poses = []
    if not args.no_color:
        camera_poses, camera_ts = load_pose_csv(camera_pose_path)
        full_map = colorize_map(
            full_map,
            image_dir,
            camera_poses,
            camera_ts,
            camera_matrix(args.track),
            z_tolerance=args.z_tolerance,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), full_map)
    print(f"Saved colored map to {output_path}.")

    if args.visualize:
        geometries = [full_map] + draw_camera_trajectory_and_axes(
            camera_poses,
            step=args.camera_step,
            scale=args.camera_axis_scale,
        )
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Colored LiDAR Map with Camera Path",
            width=1280,
            height=720,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()
