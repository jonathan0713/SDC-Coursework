import argparse
import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[Warning] pytorch_msssim not installed. Fallback to pure L1 loss.")

from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from train import create_gaussians_with_optimizers, generate_sky_points, validate_gsplat_runtime
from utils.general_utils import mkdir_p


def compute_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: torch.Tensor,
    ssim_weight: float = 0.2,
    scale_reg_weight: float = 0.005,
) -> torch.Tensor:
    rendered_masked = rendered * mask
    target_masked = target * mask

    valid_pixels = mask.sum() + 1e-5
    l1_loss = F.l1_loss(rendered_masked, target_masked, reduction="sum") / valid_pixels

    if HAS_SSIM and ssim_weight > 0.0:
        r4d = rendered_masked.unsqueeze(0).clamp(0, 1)
        t4d = target_masked.unsqueeze(0).clamp(0, 1)
        ssim_val = ssim_fn(r4d, t4d, data_range=1.0, size_average=True)
        photometric_loss = (1.0 - ssim_weight) * l1_loss + ssim_weight * (1.0 - ssim_val)
    else:
        photometric_loss = l1_loss

    scale_penalty = torch.mean(torch.exp(scales)) * scale_reg_weight

    return photometric_loss + scale_penalty


class Track1Trainer:
    def __init__(self, data_dir: Path, output_dir: Path, config: dict):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validate_gsplat_runtime(self.device)

        self.dataset = Dataset(self.data_dir, track="Track1")
        pc_points, pc_colors = self.dataset.get_pointcloud()
        num_pc = pc_points.shape[0]
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)

        sky_points, sky_colors = generate_sky_points(
            self.dataset,
            num_points=config["sky_points"],
            device=self.device,
        )
        num_sky = sky_points.shape[0]

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
            sh_degree=config["sh_degree"],
            means_lr=config["lr_xyz"],
            scale_lr=config["lr_scaling"],
            opacity_lr=config["lr_opacity"],
            quat_lr=config["lr_rotation"],
            sh0_lr=config["lr_color"],
            device=str(self.device),
        )

        self.scheduler = ExponentialLR(
            self.optimizers["means"], gamma=config["lr_decay"]
        )

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
            verbose=False,
        )
        self.strategy_state = self.strategy.initialize_state()

        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        self.iteration = 0
        self.last_info = None

    def rasterize(self, camtoworld: torch.Tensor) -> torch.Tensor:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        c0 = 0.28209479177387814
        rgb = self.splats["sh0"].squeeze(1) * c0 + 0.5
        viewmat = torch.linalg.inv(camtoworld)

        colors_input = rgb
        sh_degree = self.config["sh_degree"]
        if sh_degree > 0:
            sh_rest = self.splats["shN"]
            colors_input = torch.cat([self.splats["sh0"], sh_rest], dim=1)

        features, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_input,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=self.dataset.image_width,
            height=self.dataset.image_height,
            sh_degree=sh_degree,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
        )
        self.last_info = info
        image = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image

    def train_step(self, frame_indices: np.ndarray) -> float:
        for opt in self.optimizers.values():
            opt.zero_grad()

        total_loss = 0.0
        for frame_idx in frame_indices:
            target = self.dataset.get_image(int(frame_idx)).to(self.device)
            mask = self.dataset.get_mask(int(frame_idx)).to(self.device)
            object_mask = (mask == 0).float().unsqueeze(0)

            rendered = self.rasterize(self.poses_c2w[int(frame_idx)])

            loss = compute_loss(
                rendered=rendered,
                target=target,
                mask=object_mask,
                scales=self.splats["scales"],
                ssim_weight=self.config["ssim_weight"],
                scale_reg_weight=self.config["scale_reg_weight"]
            )
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
                pbar.set_postfix({"loss": f"{loss:.5f}", "gs": len(self.splats["means"])})
                self.iteration += 1

            self.scheduler.step()

            torch.cuda.empty_cache()

            if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch: int):
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "iteration": self.iteration,
            "splats": {k: v.data for k, v in self.splats.items()},
        }, path)

    def save_ply(self, filename: str):
        means = self.splats["means"].detach().cpu().numpy()
        scales = self.splats["scales"].detach().cpu().numpy()
        quats = self.splats["quats"].detach().cpu().numpy()
        opacities = self.splats["opacities"].squeeze(-1).detach().cpu().numpy()
        sh0 = self.splats["sh0"].squeeze(1).detach().cpu().numpy()

        has_shN = "shN" in self.splats
        if has_shN:
            shN = self.splats["shN"].view(means.shape[0], -1).detach().cpu().numpy()

        path = self.output_dir / filename
        with path.open("wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(means)}\n".encode())

            props = [
                "x", "y", "z",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
                "opacity",
                "f_dc_0", "f_dc_1", "f_dc_2"
            ]

            if has_shN:
                for i in range(shN.shape[1]):
                    props.append(f"f_rest_{i}")

            for prop in props:
                f.write(f"property float {prop}\n".encode())
            f.write(b"end_header\n")

            for i in range(len(means)):
                values = [
                    means[i, 0], means[i, 1], means[i, 2],
                    scales[i, 0], scales[i, 1], scales[i, 2],
                    quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3],
                    opacities[i],
                    sh0[i, 0], sh0[i, 1], sh0[i, 2],
                ]
                if has_shN:
                    values.extend(shN[i].tolist())
                f.write(struct.pack(f"{len(values)}f", *values))


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/track1_hq"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--ply-name", default="gaussian_reconstruction.ply")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or (args.data_root / "Track1")

    config = {
        "sh_degree":    3,
        "lr_xyz":       0.00016,
        "lr_color":     0.005,
        "lr_opacity":   0.05,
        "lr_scaling":   0.005,
        "lr_rotation":  0.001,
        "lr_decay":     0.99995,

        "checkpoint_interval": args.checkpoint_interval,

        "init_scale_lidar": 0.01,
        "init_scale_sky":   0.5,
        "init_opacity":     0.7,

        "refine_start_iter": 100,
        "refine_stop_iter":  30000,
        "refine_every":      200,
        "reset_every":       4000,
        "sky_points":        20000,

        "prune_opa":    0.03,
        "grow_grad2d":  0.0008,
        "grow_scale3d": 0.01,
        "grow_scale2d": 0.05,
        "prune_scale3d": 0.1,
        "prune_scale2d": 0.15,

        "ssim_weight": 0.2,
        "scale_reg_weight": 0.005,
    }

    trainer = Track1Trainer(data_dir, args.output_dir, config)
    trainer.train(args.epochs, args.batch_size)
    trainer.save_ply(args.ply_name)


if __name__ == "__main__":
    main()
