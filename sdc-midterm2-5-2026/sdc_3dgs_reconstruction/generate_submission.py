import argparse
import csv
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from gsplat.rendering import rasterization
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import TRACK_INTRINSICS


def parse_background_color(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"none", "black"}:
        return None
    if value == "white":
        return [1.0, 1.0, 1.0]
    if value == "auto":
        return "auto"

    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--background-color must be auto, black, white, or R,G,B")
    if max(parts) > 1.0:
        parts = [part / 255.0 for part in parts]
    return [float(np.clip(part, 0.0, 1.0)) for part in parts]


def estimate_sky_background(track, data_root, sample_count=80):
    image_dir = Path(data_root) / track / "data" / "image"
    mask_dir = Path(data_root) / track / "result" / "sky_masks"
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        return None

    colors = []
    indices = np.linspace(0, len(mask_files) - 1, min(sample_count, len(mask_files)), dtype=int)
    for idx in indices:
        mask_path = mask_files[idx]
        image_path = next(
            (
                image_dir / f"{mask_path.stem}{suffix}"
                for suffix in (".jpg", ".jpeg", ".png")
                if (image_dir / f"{mask_path.stem}{suffix}").exists()
            ),
            None,
        )
        if image_path is None:
            continue
        if not image_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(image_path))
        if mask is None or image is None:
            continue
        sky = mask >= 128
        if not np.any(sky):
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        colors.append(image[sky])

    if not colors:
        return None
    return np.concatenate(colors, axis=0).mean(axis=0).clip(0.0, 1.0).tolist()


def validate_gsplat_runtime(device):
    if torch.device(device).type != "cuda":
        raise RuntimeError("gsplat rendering requires CUDA. Use a CUDA-capable environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU. Check `nvidia-smi` and your driver.")

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and device_arch not in supported_arches:
        raise RuntimeError(
            f"Your GPU requires {device_arch}, but this PyTorch build only supports "
            f"{', '.join(sorted(supported_arches))}. Install a newer PyTorch/CUDA build "
            "that supports this GPU, or run on an older supported GPU."
        )

    try:
        from gsplat.cuda import _backend
    except Exception as exc:
        raise RuntimeError(f"Could not import gsplat CUDA backend: {exc}") from exc
    if getattr(_backend, "_C", None) is None:
        raise RuntimeError(
            "gsplat's CUDA extension is disabled. Install the CUDA toolkit/nvcc and rebuild gsplat."
        )


class StandaloneRenderer:
    """Renderer for RGB novel view synthesis from exported 3DGS PLY files."""

    def __init__(self, ply_path, intrinsics, device="cuda", background_color=None):
        self.device = torch.device(device)
        self.K = intrinsics.to(self.device)
        self.background_color = (
            torch.tensor(background_color, dtype=torch.float32, device=self.device)
            if background_color is not None
            else None
        )
        self.splats = self.load_ply(ply_path)

    def load_ply(self, path):
        print(f"Loading splats from {path}...")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} does not exist. Train failed or saved the PLY to a different output directory."
            )
        with Path(path).open("rb") as f:
            num_points = None
            properties = []
            while True:
                line = f.readline().decode("ascii")
                if line.startswith("element vertex"):
                    num_points = int(line.split()[-1])
                elif line.startswith("property float"):
                    properties.append(line.split()[-1])
                if line.strip() == "end_header":
                    break

            if num_points is None:
                raise ValueError(f"PLY header does not contain vertex count: {path}")
            if len(properties) < 14:
                raise ValueError(f"PLY has too few vertex properties: {len(properties)}")

            point_format = f"{len(properties)}f"
            size = struct.calcsize(point_format)
            data = [struct.unpack(point_format, f.read(size)) for _ in range(num_points)]

        data = torch.tensor(data, device=self.device)
        splats = nn.ParameterDict(
            {
                "means": nn.Parameter(data[:, 0:3]),
                "scales": nn.Parameter(data[:, 3:6]),
                "quats": nn.Parameter(data[:, 6:10]),
                "opacities": nn.Parameter(data[:, 10:11]),
                "sh0": nn.Parameter(data[:, 11:14].unsqueeze(1)),
            }
        )

        rest_count = len(properties) - 14
        if rest_count:
            if rest_count % 3 != 0:
                raise ValueError(f"SH rest property count must be divisible by 3, got {rest_count}")
            num_rest_bases = rest_count // 3
            num_bases = num_rest_bases + 1
            sh_degree = int(round(math.sqrt(num_bases) - 1))
            if (sh_degree + 1) ** 2 != num_bases:
                raise ValueError(f"Could not infer SH degree from {num_bases} bases")
            splats["shN"] = nn.Parameter(data[:, 14:].reshape(num_points, num_rest_bases, 3))
            self.sh_degree = sh_degree
        else:
            self.sh_degree = 0

        print(f"Loaded {num_points} Gaussians with sh_degree={self.sh_degree}.")
        return splats

    def render_rgb(self, camtoworld, width, height):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        if self.sh_degree > 0 and "shN" in self.splats:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
            sh_degree = self.sh_degree
        else:
            c0 = 0.28209479177387814
            colors = self.splats["sh0"].squeeze(1) * c0 + 0.5
            sh_degree = None
        viewmat = torch.linalg.inv(camtoworld)

        features, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=sh_degree,
            backgrounds=(
                self.background_color.unsqueeze(0)
                if self.background_color is not None
                else None
            ),
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
        )

        image_rgb = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image_rgb


def read_frame_list(path):
    with Path(path).open("r") as f:
        return [line.strip().split(".")[0] for line in f if line.strip()]


def load_intrinsics(track, json_path=None):
    if json_path and Path(json_path).exists():
        with Path(json_path).open("r") as f:
            data = json.load(f)
    else:
        data = TRACK_INTRINSICS[track]

    intrinsics = torch.tensor(
        [
            [data["fx"], 0.0, data["cx"]],
            [0.0, data["fy"], data["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return intrinsics, int(data["width"]), int(data["height"])


def copy_or_create_submission_csv(track, output_dir, source_csv=None):
    output_csv = output_dir / "submission.csv"
    if source_csv and Path(source_csv).exists():
        shutil.copyfile(source_csv, output_csv)
        return

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
    print(f"Wrote placeholder submission CSV to {output_csv}")


def default_paths(track, data_root):
    script_dir = Path(__file__).resolve().parent
    track_dir = script_dir / track.lower()
    return {
        "ply": script_dir / "output" / "gaussian_reconstruction.ply",
        "intrinsics": data_root / track / "result" / "camera_intrinsics.json",
        "pose_list": track_dir / "test_pose_list.txt",
        "frame_list": track_dir / f"{track.lower()}_test_frame_list.txt",
        "output_dir": track_dir / "test_submission",
        "submission_csv": track_dir / "submission.csv",
    }


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render official test images from a trained 3DGS PLY.")
    parser.add_argument("--track", default="Track2", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--pose-list", type=Path)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--background-color",
        default="black",
        help="Background for uncovered pixels: black, white, auto, or R,G,B. Use the same value as training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = default_paths(args.track, args.data_root)

    ply_path = args.ply or paths["ply"]
    intrinsics_path = args.intrinsics or paths["intrinsics"]
    pose_list_path = args.pose_list or paths["pose_list"]
    frame_list_path = args.frame_list or paths["frame_list"]
    output_dir = args.output_dir or paths["output_dir"]
    submission_csv = args.submission_csv or paths["submission_csv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = read_frame_list(frame_list_path)
    raw_poses = np.loadtxt(pose_list_path)
    raw_poses = np.asarray(raw_poses).reshape(-1, 16)
    test_poses = [torch.from_numpy(row.reshape(4, 4)).float() for row in raw_poses]

    if len(frame_ids) != len(test_poses):
        raise ValueError(
            f"Frame count ({len(frame_ids)}) does not match pose count ({len(test_poses)})"
        )

    intrinsics, width, height = load_intrinsics(args.track, intrinsics_path)
    validate_gsplat_runtime(args.device)
    background_color = parse_background_color(args.background_color)
    if background_color == "auto":
        background_color = estimate_sky_background(args.track, args.data_root)
        print(f"Auto background RGB: {background_color}")
    elif background_color is not None:
        print(f"Background RGB: {background_color}")

    renderer = StandaloneRenderer(ply_path, intrinsics, device=args.device, background_color=background_color)

    print(f"Rendering {len(frame_ids)} {args.track} views to {output_dir}.")
    for frame_id, pose in tqdm(zip(frame_ids, test_poses), total=len(frame_ids)):
        rgb = renderer.render_rgb(pose.to(renderer.device), width, height)
        rgb_np = (
            rgb.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .clip(0.0, 1.0)
            * 255.0
        ).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{frame_id}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    copy_or_create_submission_csv(args.track, output_dir, source_csv=submission_csv)
    print(f"Done. Renders saved to {output_dir}")


if __name__ == "__main__":
    main()
