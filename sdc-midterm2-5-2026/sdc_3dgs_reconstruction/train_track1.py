"""
Minimal Track1 trainer.

This intentionally keeps Track1 away from the extra training features in train.py:
- no SH rest coefficients
- no sky loss
- no learned/auto background
- no depth loss
- old 14-float PLY export
"""

import argparse
import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from train import create_gaussians_with_optimizers, generate_sky_points, validate_gsplat_runtime
from utils.general_utils import mkdir_p


class Track1Trainer:
    """Small, legacy-style trainer for Track1 only."""

    def __init__(self, data_dir: Path, output_dir: Path, config: dict):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Track1Trainer] Device: {self.device}")
        validate_gsplat_runtime(self.device)

        print("[Track1Trainer] Loading Track1 dataset...")
        self.dataset = Dataset(self.data_dir, track="Track1")
        pc_points, pc_colors = self.dataset.get_pointcloud()
        num_pc = pc_points.shape[0]
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)
        print(f"[Track1Trainer] Points: {num_pc}, Cameras: {len(self.poses_c2w)}")

        sky_points, sky_colors = generate_sky_points(
            self.dataset,
            num_points=config["sky_points"],
            device=self.device,
        )
        num_sky = sky_points.shape[0]
        print(f"[Track1Trainer] Sky Gaussians: {num_sky}")

        pc_scales = torch.ones((num_pc, 3), device=self.device) * config["init_scale_lidar"]
        sky_scales = torch.ones((num_sky, 3), device=self.device) * config["init_scale_sky"]

        if num_sky > 0:
            points = torch.cat([pc_points.to(self.device), sky_points], dim=0)
            colors = torch.cat([pc_colors.to(self.device), sky_colors], dim=0)
            scales = torch.cat([pc_scales, sky_scales], dim=0)
        else:
            points = pc_points.to(self.device)
            colors = pc_colors.to(self.device)
            scales = pc_scales

        self.splats, self.optimizers = create_gaussians_with_optimizers(
            points=points,
            rgbs=colors,
            init_scales=scales,
            init_opacity=config["init_opacity"],
            sh_degree=0,
            means_lr=config["lr_xyz"],
            scale_lr=config["lr_scaling"],
            opacity_lr=config["lr_opacity"],
            quat_lr=config["lr_rotation"],
            sh0_lr=config["lr_color"],
            device=str(self.device),
        )
        print(f"[Track1Trainer] Initialized {len(self.splats['means'])} Gaussians")

        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=config["lr_decay"])
        self.strategy = DefaultStrategy(
            prune_opa=config["prune_opa"],
            grow_grad2d=config["grow_grad2d"],
            grow_scale3d=config["grow_scale3d"],
            grow_scale2d=config["grow_scale2d"],
            prune_scale3d=config["prune_scale3d"],
            prune_scale2d=config["prune_scale2d"],
            refine_start_iter=config["refine_start_iter"],
            refine_stop_iter=config["refine_stop_iter"],
            refine_every=config["refine_every"],
            reset_every=config["reset_every"],
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state()
        print(
            "[Track1Trainer] Strategy: "
            f"{self.strategy.refine_start_iter}-{self.strategy.refine_stop_iter}, "
            f"every {self.strategy.refine_every}"
        )

        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        self.iteration = 0
        self.last_info = None

    def rasterize(self, camtoworld: torch.Tensor):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        c0 = 0.28209479177387814
        rgb = self.splats["sh0"].squeeze(1) * c0 + 0.5
        viewmat = torch.linalg.inv(camtoworld)

        features, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgb,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=self.dataset.image_width,
            height=self.dataset.image_height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
        )
        self.last_info = info
        image = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image

    def train_step(self, frame_indices: np.ndarray):
        for opt in self.optimizers.values():
            opt.zero_grad()

        total_loss = 0.0
        for frame_idx in frame_indices:
            target = self.dataset.get_image(int(frame_idx)).to(self.device)
            mask = self.dataset.get_mask(int(frame_idx)).to(self.device)
            object_mask = (mask == 0).float().unsqueeze(0)

            rendered = self.rasterize(self.poses_c2w[int(frame_idx)])
            loss = F.l1_loss(rendered * object_mask, target * object_mask)
            total_loss = total_loss + loss

        total_loss = total_loss / len(frame_indices)

        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
        )
        total_loss.backward()
        for opt in self.optimizers.values():
            opt.step()
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
            packed=False,
        )
        return total_loss.item()

    def train(self, epochs: int, batch_size: int):
        print(f"[Track1Trainer] Training: epochs={epochs}, batch_size={batch_size}")
        num_batches = (len(self.dataset) + batch_size - 1) // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(len(self.dataset))
            epoch_loss = 0.0
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]

                loss = self.train_step(batch_indices)
                epoch_loss += loss
                self.tb_writer.add_scalar("loss/train", loss, self.iteration)
                self.tb_writer.add_scalar("gs_count", len(self.splats["means"]), self.iteration)
                pbar.set_postfix({"loss": f"{loss:.6f}"})
                self.iteration += 1

            self.scheduler.step()
            print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss / num_batches:.6f}")
            if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch: int):
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "iteration": self.iteration,
                "splats": {k: v.data for k, v in self.splats.items()},
            },
            path,
        )
        print(f"[Track1Trainer] Saved checkpoint: {path}")

    def save_ply(self, filename: str):
        means = self.splats["means"].detach().cpu().numpy()
        scales = self.splats["scales"].detach().cpu().numpy()
        quats = self.splats["quats"].detach().cpu().numpy()
        opacities = self.splats["opacities"].squeeze(-1).detach().cpu().numpy()
        sh0 = self.splats["sh0"].squeeze(1).detach().cpu().numpy()

        path = self.output_dir / filename
        with path.open("wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(means)}\n".encode())
            for prop in (
                "x",
                "y",
                "z",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
                "opacity",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
            ):
                f.write(f"property float {prop}\n".encode())
            f.write(b"end_header\n")

            for i in range(len(means)):
                values = [
                    means[i, 0],
                    means[i, 1],
                    means[i, 2],
                    scales[i, 0],
                    scales[i, 1],
                    scales[i, 2],
                    quats[i, 0],
                    quats[i, 1],
                    quats[i, 2],
                    quats[i, 3],
                    opacities[i],
                    sh0[i, 0],
                    sh0[i, 1],
                    sh0[i, 2],
                ]
                f.write(struct.pack("14f", *values))
        print(f"[Track1Trainer] Saved PLY: {path}")


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Minimal legacy-style Track1 3DGS trainer.")
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/track1_legacy"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sky-points", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--init-scale-lidar", type=float, default=0.15)
    parser.add_argument("--init-scale-sky", type=float, default=0.5)
    parser.add_argument("--init-opacity", type=float, default=0.7)
    parser.add_argument("--refine-start-iter", type=int, default=199)
    parser.add_argument("--refine-stop-iter", type=int, default=2001)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=3000)
    parser.add_argument("--ply-name", default="gaussian_reconstruction.ply")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or (args.data_root / "Track1")
    config = {
        "lr_xyz": 0.00016,
        "lr_color": 0.0025,
        "lr_opacity": 0.05,
        "lr_scaling": 0.005,
        "lr_rotation": 0.001,
        "lr_decay": 0.9999,
        "checkpoint_interval": args.checkpoint_interval,
        "init_scale_lidar": args.init_scale_lidar,
        "init_scale_sky": args.init_scale_sky,
        "init_opacity": args.init_opacity,
        "refine_start_iter": args.refine_start_iter,
        "refine_stop_iter": args.refine_stop_iter,
        "refine_every": args.refine_every,
        "reset_every": args.reset_every,
        "sky_points": args.sky_points,
        "prune_opa": 0.005,
        "grow_grad2d": 0.0001,
        "grow_scale3d": 0.01,
        "grow_scale2d": 0.05,
        "prune_scale3d": 0.15,
        "prune_scale2d": 0.15,
    }

    trainer = Track1Trainer(data_dir, args.output_dir, config)
    trainer.train(args.epochs, args.batch_size)
    trainer.save_ply(args.ply_name)
    print("\nTrack1 legacy training complete!")


if __name__ == "__main__":
    main()
