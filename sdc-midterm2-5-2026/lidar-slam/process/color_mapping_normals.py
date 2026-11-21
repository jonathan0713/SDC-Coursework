import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
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


def build_map_with_normals(
    lidar_poses,
    pose_ts,
    pcd_dir,
    voxel_size=0.1,
    min_range=0.0,
    max_range=0.0,
    chunk_size=5,
    normal_radius=0.5,
    normal_max_nn=30,
    outlier_neighbors=50,
    outlier_std=1.0,
):
    """Build a global map and estimate normals oriented toward local sensor origins."""
    pcd_dir = Path(pcd_dir)
    pcd_files = sorted(pcd_dir.glob("*.pcd"))
    if not pcd_files:
        raise FileNotFoundError(f"No PCD files found in {pcd_dir}")

    pose_ts = np.asarray(pose_ts, dtype=np.int64)
    if len(pose_ts) == 0:
        raise ValueError("No LiDAR poses were loaded.")

    combined_pcd = o3d.geometry.PointCloud()
    chunk_pcd = o3d.geometry.PointCloud()
    chunk_origins = []

    print(f"Merging {len(pcd_files)} PCD frames with chunk normals (chunk={chunk_size}).")
    for local_idx, pcd_path in tqdm(enumerate(pcd_files), total=len(pcd_files)):
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

        pose = lidar_poses[closest_idx]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.transform(pose)

        chunk_pcd += pcd
        chunk_origins.append(pose[:3, 3])

        is_chunk_end = (local_idx + 1) % chunk_size == 0 or (local_idx + 1) == len(pcd_files)
        if not is_chunk_end or len(chunk_pcd.points) == 0:
            continue

        chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size=voxel_size)
        chunk_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn,
            )
        )
        chunk_pcd.orient_normals_towards_camera_location(np.mean(chunk_origins, axis=0))

        combined_pcd += chunk_pcd
        chunk_pcd = o3d.geometry.PointCloud()
        chunk_origins = []

    if len(combined_pcd.points) == 0:
        raise ValueError("The merged point cloud is empty.")

    print("Downsampling and cleaning normal-aware map.")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(
        nb_neighbors=outlier_neighbors,
        std_ratio=outlier_std,
    )
    return combined_pcd


def colorize_map_with_normals(
    pcd,
    image_dir,
    image_poses,
    image_pose_ts,
    intrinsics,
    facing_threshold=-0.2,
    depth_dilation_size=3,
    depth_slack=0.8,
    closest_reset_ratio=0.90,
    closest_accum_ratio=1.15,
    fill_radius=0.5,
    default_color=(0.2, 0.2, 0.2),
):
    """Colorize with z-buffering, depth dilation, normal backface culling, and neighbor filling."""
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        print(f"Skipping colorization: no JPG images found in {image_dir}.")
        return pcd

    if not pcd.has_normals():
        raise ValueError("Point cloud missing normals. Use build_map_with_normals first.")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_points = points.shape[0]
    if n_points == 0:
        return pcd

    pose_by_ts = {int(ts): pose for ts, pose in zip(image_pose_ts, image_poses)}
    points_h = np.hstack((points, np.ones((n_points, 1))))

    color_sum = np.zeros((n_points, 3), dtype=float)
    color_counts = np.zeros(n_points, dtype=float)
    min_depths = np.full(n_points, np.inf, dtype=float)

    print(f"Normal-aware colorization with {len(image_files)} images.")
    for image_path in tqdm(image_files):
        if not image_path.stem.isdigit():
            continue

        pose = pose_by_ts.get(int(image_path.stem))
        if pose is None:
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        height, width = image.shape[:2]

        t_camera_world = np.linalg.inv(pose)
        points_cam = (t_camera_world @ points_h.T).T[:, :3]
        normals_cam = (t_camera_world[:3, :3] @ normals.T).T

        valid_z = points_cam[:, 2] > 0.1
        if not np.any(valid_z):
            continue

        valid_indices_z = np.where(valid_z)[0]
        points_cam_valid = points_cam[valid_z]
        normals_cam_valid = normals_cam[valid_z]

        view_dirs = -points_cam_valid
        view_norm = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs = view_dirs / np.maximum(view_norm, 1e-12)
        facing = np.sum(view_dirs * normals_cam_valid, axis=1) > facing_threshold

        projected = (intrinsics @ points_cam_valid.T).T
        z = projected[:, 2]
        u = np.floor(projected[:, 0] / z).astype(np.int32)
        v = np.floor(projected[:, 1] / z).astype(np.int32)

        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        final = facing & in_image
        if not np.any(final):
            continue

        valid_indices = valid_indices_z[final]
        u = u[final]
        v = v[final]
        z = z[final]

        depth_buffer = np.full((height, width), np.inf, dtype=float)
        np.minimum.at(depth_buffer, (v, u), z)

        dense_depth_buffer = minimum_filter(depth_buffer, size=depth_dilation_size)
        visible = z <= dense_depth_buffer[v, u] + depth_slack
        if not np.any(visible):
            continue

        for idx, vv, uu, zz in zip(valid_indices[visible], v[visible], u[visible], z[visible]):
            if zz < min_depths[idx] * closest_reset_ratio:
                min_depths[idx] = zz
                color_sum[idx] = image[vv, uu]
                color_counts[idx] = 1.0
            elif zz <= min_depths[idx] * closest_accum_ratio:
                color_sum[idx] += image[vv, uu]
                color_counts[idx] += 1.0
                if zz < min_depths[idx]:
                    min_depths[idx] = zz

    colors = np.tile(np.asarray(default_color, dtype=float), (n_points, 1))
    colored = color_counts > 0
    colors[colored] = color_sum[colored] / color_counts[colored, None]

    if fill_radius > 0 and np.any(colored) and np.any(~colored):
        print(f"Filling blind spots from colored neighbors within {fill_radius}m.")
        colored_points = points[colored]
        colored_colors = colors[colored]
        tree = cKDTree(colored_points)

        uncolored_indices = np.where(~colored)[0]
        neighbors = tree.query_ball_point(points[uncolored_indices], r=fill_radius)
        for i, neighbor_indices in enumerate(neighbors):
            if neighbor_indices:
                colors[uncolored_indices[i]] = np.mean(colored_colors[neighbor_indices], axis=0)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Colored {int(np.count_nonzero(colored))}/{n_points} points before neighbor filling.")
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
        description="Build and color a LiDAR map using normal-aware image projection."
    )
    parser.add_argument("--track", default="Track3", choices=["Track1", "Track2", "Track3"])
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
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--normal-radius", type=float, default=0.5)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--outlier-neighbors", type=int, default=80)
    parser.add_argument("--outlier-std", type=float, default=2.0)
    parser.add_argument("--chunk-outlier-neighbors", type=int, default=50)
    parser.add_argument("--chunk-outlier-std", type=float, default=1.0)
    parser.add_argument("--facing-threshold", type=float, default=-0.2)
    parser.add_argument("--depth-dilation-size", type=int, default=3)
    parser.add_argument("--depth-slack", type=float, default=0.8)
    parser.add_argument("--fill-radius", type=float, default=0.5)
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
    full_map = build_map_with_normals(
        lidar_poses,
        lidar_ts,
        pcd_dir,
        voxel_size=args.map_voxel,
        min_range=args.min_range,
        max_range=args.max_range,
        chunk_size=args.chunk_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        outlier_neighbors=args.chunk_outlier_neighbors,
        outlier_std=args.chunk_outlier_std,
    )

    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.raw_output), full_map)

    print("Removing final statistical outliers.")
    full_map, _ = full_map.remove_statistical_outlier(
        nb_neighbors=args.outlier_neighbors,
        std_ratio=args.outlier_std,
    )
    full_map = full_map.voxel_down_sample(voxel_size=args.final_voxel)

    camera_poses = []
    if not args.no_color:
        camera_poses, camera_ts = load_pose_csv(camera_pose_path)
        full_map = colorize_map_with_normals(
            full_map,
            image_dir,
            camera_poses,
            camera_ts,
            camera_matrix(args.track),
            facing_threshold=args.facing_threshold,
            depth_dilation_size=args.depth_dilation_size,
            depth_slack=args.depth_slack,
            fill_radius=args.fill_radius,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), full_map)
    print(f"Saved normal-aware colored map to {output_path}.")

    if args.visualize:
        geometries = [full_map] + draw_camera_trajectory_and_axes(
            camera_poses,
            step=args.camera_step,
            scale=args.camera_axis_scale,
        )
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Normal-Aware Colored LiDAR Map",
            width=1280,
            height=720,
            mesh_show_back_face=True,
        )


if __name__ == "__main__":
    main()
