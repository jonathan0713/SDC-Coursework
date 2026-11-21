"""
Dataset reader for 3DGS training.

It supports both the TA sample layout:
    dataset/itri58_image
    dataset/camera_gt_pose.txt
    dataset/camera_intrinsics.json
    dataset/itri58_full_color_map.pcd
    dataset/sky_masks

and this repository's Track layout:
    data/TrackX/data/image
    data/TrackX/result/camera_pose.csv
    data/TrackX/result/TrackX.pcd
    data/TrackX/result/sky_masks
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch


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

TRACK_INTRINSICS = {
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def infer_track(data_dir, explicit_track=None):
    if explicit_track:
        return explicit_track

    for part in Path(data_dir).parts[::-1]:
        if part.lower() in {"track1", "track2", "track3"}:
            return part[0].upper() + part[1:]
    return None


def load_pose_file(path):
    """Load camera-to-world poses from CSV/TXT. Returns (poses, timestamps_or_none)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fields = reader.fieldnames
                ts_field = next(
                    (
                        name
                        for name in fields
                        if name.lower() in {"timestamp", "timestamps", "ts", "time"}
                    ),
                    None,
                )
                if all(col in fields for col in POSE_COLUMNS):
                    poses = []
                    timestamps = []
                    for row in reader:
                        poses.append(
                            np.array([float(row[col]) for col in POSE_COLUMNS], dtype=np.float32).reshape(4, 4)
                        )
                        if ts_field is not None:
                            timestamps.append(int(row[ts_field]))
                    return poses, np.array(timestamps, dtype=np.int64) if ts_field else None

        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    else:
        raw = np.loadtxt(path, delimiter=",")

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    timestamps = None
    if raw.shape[1] == 17:
        timestamps = raw[:, 0].astype(np.int64)
        raw = raw[:, 1:]
    elif raw.shape[1] != 16:
        raise ValueError(f"Unsupported pose format in {path}: {raw.shape[1]} columns")

    poses = [row.reshape(4, 4) for row in raw]
    return poses, timestamps


class Dataset:
    """Loads images, camera poses, intrinsics, sky masks, and the colored map."""

    def __init__(
        self,
        data_dir,
        track=None,
        image_dir=None,
        poses_file=None,
        intrinsics_file=None,
        pointcloud_file=None,
        mask_dir=None,
    ):
        self.data_dir = Path(data_dir)
        self.track = infer_track(self.data_dir, track)

        self.image_dir = Path(image_dir) if image_dir else self._first_existing(
            self.data_dir / "itri58_image",
            self.data_dir / "data" / "image",
        )
        self.poses_file = Path(poses_file) if poses_file else self._first_existing(
            self.data_dir / "camera_gt_pose.txt",
            self.data_dir / "result" / "camera_pose.csv",
            self.data_dir / "camera_pose.csv",
        )
        self.intrinsics_file = Path(intrinsics_file) if intrinsics_file else self._first_existing(
            self.data_dir / "camera_intrinsics.json",
            self.data_dir / "result" / "camera_intrinsics.json",
            required=False,
        )
        self.pointcloud_file = Path(pointcloud_file) if pointcloud_file else self._find_pointcloud()
        self.mask_dir = Path(mask_dir) if mask_dir else self._first_existing(
            self.data_dir / "sky_masks",
            self.data_dir / "result" / "sky_masks",
            required=False,
        )

        self._load_intrinsics()
        self._load_image_list()
        self._load_poses()
        self._load_pointcloud()
        self._load_mask_list()

    def _first_existing(self, *paths, required=True):
        for path in paths:
            if path.exists():
                return path
        if required:
            raise FileNotFoundError(f"None of these paths exists: {', '.join(map(str, paths))}")
        return paths[0]

    def _find_pointcloud(self):
        candidates = [
            self.data_dir / "itri58_full_color_map.pcd",
            self.data_dir / "result" / f"{self.track}.pcd" if self.track else None,
            self.data_dir / "result" / "Track1.pcd",
            self.data_dir / "result" / "Track2.pcd",
            self.data_dir / "result" / "Track3.pcd",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate

        result_dir = self.data_dir / "result"
        pcd_files = sorted(result_dir.glob("*.pcd")) if result_dir.exists() else []
        if pcd_files:
            return pcd_files[0]
        raise FileNotFoundError(f"No colored PCD found under {self.data_dir}")

    def _load_intrinsics(self):
        if self.intrinsics_file.exists():
            with self.intrinsics_file.open("r") as f:
                data = json.load(f)
        elif self.track in TRACK_INTRINSICS:
            data = TRACK_INTRINSICS[self.track]
            print(f"[Dataset] Using built-in intrinsics for {self.track}")
        else:
            raise FileNotFoundError(
                f"No camera_intrinsics.json found and no built-in intrinsics for {self.track}"
            )

        fx = data["fx"]
        fy = data["fy"]
        cx = data["cx"]
        cy = data["cy"]
        self.image_width = int(data["width"])
        self.image_height = int(data["height"])
        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        print(f"[Dataset] Image size: {self.image_width}x{self.image_height}")
        print(f"[Dataset] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    def _load_image_list(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        self.image_filenames = sorted(
            path.name for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_filenames:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        print(f"[Dataset] Found {len(self.image_filenames)} images in {self.image_dir}")

    def _load_poses(self):
        poses, pose_ts = load_pose_file(self.poses_file)

        if pose_ts is None:
            if len(poses) != len(self.image_filenames):
                raise ValueError(
                    f"Pose count ({len(poses)}) does not match image count ({len(self.image_filenames)})"
                )
            self.poses = poses
        else:
            pose_by_ts = {int(ts): pose for ts, pose in zip(pose_ts, poses)}
            aligned_images = []
            aligned_poses = []
            missing = 0

            for name in self.image_filenames:
                stem = Path(name).stem
                if not stem.isdigit():
                    missing += 1
                    continue
                pose = pose_by_ts.get(int(stem))
                if pose is None:
                    missing += 1
                    continue
                aligned_images.append(name)
                aligned_poses.append(pose)

            if not aligned_poses:
                raise ValueError(
                    f"No image timestamps matched poses in {self.poses_file}"
                )

            self.image_filenames = aligned_images
            self.poses = aligned_poses
            if missing:
                print(f"[Dataset] Skipped {missing} images without matching camera poses")

        print(f"[Dataset] Loaded {len(self.poses)} camera poses from {self.poses_file}")

    def _load_pointcloud(self):
        pcd = o3d.io.read_point_cloud(str(self.pointcloud_file))
        self.points_map = np.asarray(pcd.points, dtype=np.float32)
        self.colors_map = np.asarray(pcd.colors, dtype=np.float32)

        if self.points_map.size == 0:
            raise ValueError(f"Point cloud is empty: {self.pointcloud_file}")
        if self.colors_map.shape[0] != self.points_map.shape[0]:
            self.colors_map = np.full_like(self.points_map, 0.5, dtype=np.float32)

        self.colors_map = np.clip(self.colors_map, 0.0, 1.0)
        print(f"[Dataset] Loaded point cloud with {len(self.points_map)} points")
        print(f"[Dataset] Point cloud: {self.pointcloud_file}")

    def _load_mask_list(self):
        if self.mask_dir.exists():
            self.mask_filenames = sorted(
                path.name for path in self.mask_dir.iterdir() if path.suffix.lower() == ".png"
            )
        else:
            self.mask_filenames = []
        print(f"[Dataset] Found {len(self.mask_filenames)} sky masks")

    def get_poses_torch(self):
        return torch.stack([torch.from_numpy(pose).float() for pose in self.poses])

    def get_intrinsics_torch(self):
        return torch.from_numpy(self.K).float()

    def get_image(self, frame_idx):
        if frame_idx >= len(self.image_filenames):
            raise IndexError(f"Frame {frame_idx} out of range")

        image_path = self.image_dir / self.image_filenames[frame_idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def get_image_batch(self, frame_indices):
        return torch.stack([self.get_image(i) for i in frame_indices])

    def get_pointcloud(self):
        points = torch.from_numpy(self.points_map).float()
        colors = torch.from_numpy(self.colors_map).float()
        return points, colors

    def get_mask(self, frame_idx):
        mask_name = Path(self.image_filenames[frame_idx]).with_suffix(".png").name
        mask_path = self.mask_dir / mask_name
        if not mask_path.exists():
            return torch.zeros((self.image_height, self.image_width), dtype=torch.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return torch.zeros((self.image_height, self.image_width), dtype=torch.uint8)
        return torch.from_numpy(mask)

    def get_lidar_depth(self, frame_idx, device="cuda"):
        points = torch.from_numpy(self.points_map).float().to(device)
        pose_c2w = torch.from_numpy(self.poses[frame_idx]).float().to(device)
        K = torch.from_numpy(self.K).float().to(device)
        pose_w2c = torch.linalg.inv(pose_c2w)

        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=-1)
        p_cam = (pose_w2c @ points_h.T).T

        z = p_cam[:, 2]
        mask = z > 0.1
        p_cam = p_cam[mask]
        z = z[mask]

        p_pix = (K @ (p_cam[:, :3] / z.unsqueeze(-1)).T).T
        u = p_pix[:, 0].long()
        v = p_pix[:, 1].long()

        valid_mask = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]

        depth_map = torch.zeros((self.image_height, self.image_width), device=device)
        indices = torch.argsort(z, descending=True)
        depth_map[v[indices], u[indices]] = z[indices]
        return depth_map

    def __len__(self):
        return len(self.poses)

    def __repr__(self):
        return (
            f"Dataset(frames={len(self)}, points={len(self.points_map)}, "
            f"size={self.image_width}x{self.image_height})"
        )
