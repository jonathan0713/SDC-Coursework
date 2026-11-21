"""
3D Gaussian Splatting Trainer - Rewritten following gsplat official example.
Uses step_pre_backward / step_post_backward pattern for proper densification.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import gsplat
from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from utils.general_utils import mkdir_p
from utils.sh_utils import RGB2SH


def validate_gsplat_runtime(device):
    if device.type != "cuda":
        raise RuntimeError("gsplat training requires CUDA. Re-run on a machine with CUDA, or install CUDA correctly.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see a CUDA GPU. Check `nvidia-smi`, NVIDIA driver, and your PyTorch CUDA build."
        )

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
        nvcc = shutil.which("nvcc")
        hint = (
            "gsplat's CUDA extension is disabled. It needs the CUDA toolkit, especially `nvcc`, "
            "not only a CUDA-enabled PyTorch wheel."
        )
        if nvcc is None:
            hint += " `nvcc` is not on PATH."
        raise RuntimeError(hint)


def create_gaussians_with_optimizers(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scales: torch.Tensor,
    init_opacity: float = 0.5,
    sh_degree: int = 0,  # 0 means only DC component (RGB colors)
    means_lr: float = 1.6e-4,
    scale_lr: float = 5e-3,
    opacity_lr: float = 5e-2,
    quat_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 1.25e-4,
    device: str = "cuda",
) -> tuple:
    """
    Initialize Gaussians from point cloud with fixed scale.
    Returns: (splats ParameterDict, optimizers dict)
    """
    # Ensure tensors are on device
    points = points.to(device).float()
    rgbs = rgbs.to(device).float()
    rgbs = torch.clamp(rgbs, 0, 1)

    N = points.shape[0]

    # Initialize means (positions)
    means = points  # [N, 3]

    # Use the per-point scales and convert to log space
    # gsplat expects scales in log space: s_log = log(s_actual)
    scales = torch.log(init_scales.to(device))  # [N, 3]

    # Initialize rotations (identity quaternions: [0, 0, 0, 1])
    quats = torch.zeros((N, 4), device=device)
    quats[:, 3] = 1.0  # [N, 4]

    # Initialize opacities (logit space)
    init_opacity = np.clip(init_opacity, 0.001, 0.999)
    logit_opacity = np.log(init_opacity / (1 - init_opacity))
    opacities = torch.ones((N, 1), device=device) * logit_opacity  # [N, 1]

    # Initialize colors (DC SH component for RGB)
    # Convert RGB to SH DC space
    colors_sh = RGB2SH(rgbs)  # [N, 3]
    sh0 = colors_sh.unsqueeze(1)  # [N, 1, 3]
    num_sh_bases = (sh_degree + 1) ** 2

    # Create ParameterDict (mimics gsplat's approach)
    splats = nn.ParameterDict({
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
    })
    if num_sh_bases > 1:
        shN = torch.zeros((N, num_sh_bases - 1, 3), device=device)
        splats["shN"] = nn.Parameter(shN)

    # Create optimizers for each parameter with different learning rates
    optimizers = {
        "means": Adam([{"params": splats["means"], "lr": means_lr}], eps=1e-15),
        "scales": Adam([{"params": splats["scales"], "lr": scale_lr}], eps=1e-15),
        "quats": Adam([{"params": splats["quats"], "lr": quat_lr}], eps=1e-15),
        "opacities": Adam([{"params": splats["opacities"], "lr": opacity_lr}], eps=1e-15),
        "sh0": Adam([{"params": splats["sh0"], "lr": sh0_lr}], eps=1e-15),
    }
    if "shN" in splats:
        optimizers["shN"] = Adam([{"params": splats["shN"], "lr": shN_lr}], eps=1e-15)

    return splats, optimizers


def create_optimizers_for_splats(splats: nn.ParameterDict, config: dict) -> dict:
    optimizers = {
        "means": Adam([{"params": splats["means"], "lr": config['lr_xyz']}], eps=1e-15),
        "scales": Adam([{"params": splats["scales"], "lr": config['lr_scaling']}], eps=1e-15),
        "quats": Adam([{"params": splats["quats"], "lr": config['lr_rotation']}], eps=1e-15),
        "opacities": Adam([{"params": splats["opacities"], "lr": config['lr_opacity']}], eps=1e-15),
        "sh0": Adam([{"params": splats["sh0"], "lr": config.get('lr_color', 2.5e-3)}], eps=1e-15),
    }
    if "shN" in splats:
        optimizers["shN"] = Adam(
            [{"params": splats["shN"], "lr": config.get('lr_color_rest', 1.25e-4)}],
            eps=1e-15,
        )
    return optimizers

def generate_sky_points(dataset, num_points=100000, depth_range=(50.0, 80.0), device="cuda"):
    """Generates 3D points for sky regions using unprojection."""
    if num_points <= 0:
        return (
            torch.empty((0, 3), device=device),
            torch.empty((0, 3), device=device),
        )

    all_points = []
    all_colors = []

    # Sample from frames to ensure coverage
    sample_count = min(100, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    K_inv = torch.inverse(dataset.get_intrinsics_torch().to(device))

    for idx in indices:
        mask = dataset.get_mask(idx)

        image = dataset.get_image(idx).to(device)
        pose_c2w = dataset.get_poses_torch()[idx].to(device)

        # Identify sky pixels
        sky_coords = torch.where(mask == 255)
        if len(sky_coords[0]) == 0: continue

        # Sample points from the sky
        num_to_sample = max(1, num_points // len(indices))
        sel = torch.randint(0, len(sky_coords[0]), (num_to_sample,))
        y, x = sky_coords[0][sel].to(device), sky_coords[1][sel].to(device)

        # Random depth initialization
        depths = torch.rand(num_to_sample, device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]

        # Unproject: P_world = R * (K_inv * p_pix * depth) + t
        pix_h = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1)
        p_cam = (K_inv @ pix_h.unsqueeze(-1)).squeeze(-1) * depths.unsqueeze(-1)
        p_world = (pose_c2w[:3, :3] @ p_cam.unsqueeze(-1)).squeeze(-1) + pose_c2w[:3, 3]

        all_points.append(p_world)
        all_colors.append(image[:, y, x].T)

    if not all_points:
        return (
            torch.empty((0, 3), device=device),
            torch.empty((0, 3), device=device),
        )

    return torch.cat(all_points), torch.cat(all_colors)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss normalized by the number of supervised pixels, not the full image."""
    valid_pixels = mask.sum()
    if valid_pixels.item() <= 0:
        return pred.new_tensor(0.0)
    denom = valid_pixels * pred.shape[1]
    return ((pred - target).abs() * mask).sum() / denom.clamp_min(1.0)


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


def estimate_sky_background(dataset, sample_count=80, device="cuda"):
    """Estimate a single RGB background from sky-masked training pixels."""
    colors = []
    count = min(sample_count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    for idx in indices:
        mask = dataset.get_mask(idx)
        sky = mask >= 128
        if sky.sum() == 0:
            continue
        image = dataset.get_image(idx).to(device)
        colors.append(image[:, sky].T)

    if not colors:
        return torch.zeros(3, device=device)
    return torch.cat(colors, dim=0).mean(dim=0).clamp(0.0, 1.0)

class Trainer:
    """3DGS Trainer following gsplat official pattern."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: dict,
        force_cpu: bool = False,
        track: str = None,
        pointcloud_file: str = None,
    ):
        """Initialize trainer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))

        self.config = config
        self.sh_degree = int(config.get("sh_degree", 0))
        self.device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Trainer] Device: {self.device}")
        validate_gsplat_runtime(self.device)

        # Load dataset
        print("[Trainer] Loading dataset...")
        self.dataset = Dataset(data_dir, track=track, pointcloud_file=pointcloud_file)
        background_color = parse_background_color(config.get("background_color"))
        if background_color == "auto":
            self.background_color = estimate_sky_background(self.dataset, device=self.device)
            print(f"[Trainer] Auto background RGB: {self.background_color.detach().cpu().tolist()}")
        elif background_color is None:
            self.background_color = None
        else:
            self.background_color = torch.tensor(background_color, device=self.device, dtype=torch.float32)
            print(f"[Trainer] Background RGB: {self.background_color.detach().cpu().tolist()}")

        # Pointcloud data
        pc_points, pc_colors = self.dataset.get_pointcloud()
        num_pc = pc_points.shape[0]
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)

        print(f"[Trainer] Points: {len(pc_points)}, Cameras: {len(self.poses_c2w)}")

        # Add sky points at random depth
        print("[Trainer] Initializing additional sky Gaussians...")
        # Generate Sky data
        sky_points, sky_colors = generate_sky_points(
            self.dataset,
            num_points=config.get('sky_points', 10000),
            device=self.device,
        )
        num_sky = sky_points.shape[0]

        init_scale_lidar = config.get('init_scale_lidar', 0.1)
        init_scale_sky = config.get('init_scale_sky', 10.0)
        pc_scales = torch.ones((num_pc, 3)) * init_scale_lidar
        sky_scales = torch.ones((num_sky, 3)) * init_scale_sky

        # Combine LiDAR map points with optional sky points.
        if num_sky > 0:
            combined_points = torch.cat([pc_points.to(self.device), sky_points], dim=0)
            combined_colors = torch.cat([pc_colors.to(self.device), sky_colors], dim=0)
            combined_scales = torch.cat(
                [pc_scales.to(self.device), sky_scales.to(self.device)], dim=0
            )
        else:
            print("[Trainer] No sky masks/sky pixels found; training with LiDAR map points only.")
            combined_points = pc_points.to(self.device)
            combined_colors = pc_colors.to(self.device)
            combined_scales = pc_scales.to(self.device)

        # Create Gaussians and optimizers
        init_opacity = config.get('init_opacity', 0.5)
        self.splats, self.optimizers = create_gaussians_with_optimizers(
            points=combined_points,
            rgbs=combined_colors,
            init_scales=combined_scales,
            init_opacity=init_opacity,
            means_lr=config['lr_xyz'],
            scale_lr=config['lr_scaling'],
            opacity_lr=config['lr_opacity'],
            quat_lr=config['lr_rotation'],
            sh0_lr=config.get('lr_color', 2.5e-3),
            shN_lr=config.get('lr_color_rest', 1.25e-4),
            sh_degree=self.sh_degree,
            device=str(self.device),
        )

        print(f"[Trainer] Initialized {len(self.splats['means'])} Gaussians")

        # Setup learning rate scheduler (only for means which has schedule)
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=config['lr_decay'])

        # Setup strategy
        self.strategy = DefaultStrategy(
            prune_opa=config.get('prune_opa', 0.005),
            grow_grad2d=config.get('grow_grad2d', 0.0001),
            grow_scale3d=config.get('grow_scale3d', 0.01),
            grow_scale2d=config.get('grow_scale2d', 0.05),
            prune_scale3d=config.get('prune_scale3d', 0.15),
            prune_scale2d=config.get('prune_scale2d', 0.15),
            refine_start_iter=config.get('refine_start_iter', 500),
            refine_stop_iter=config.get('refine_stop_iter', 15000),
            refine_every=config.get('refine_every', 100),
            reset_every=config.get('reset_every', 3000),
            verbose=True,
        )

        # Initialize strategy state
        self.strategy_state = self.strategy.initialize_state()
        print(f"[Trainer] Strategy: densification {self.strategy.refine_start_iter}-{self.strategy.refine_stop_iter} iters, every {self.strategy.refine_every}")

        # Tensorboard
        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        self.iteration = 0

    def rasterize_splats(self, camtoworlds: torch.Tensor, K: torch.Tensor, width: int, height: int):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)

        if camtoworlds.ndim == 2:
            camtoworlds = camtoworlds.unsqueeze(0)

        viewmats = torch.linalg.inv(camtoworlds) # [C, 4, 4]

        use_sh = self.sh_degree > 0 and "shN" in self.splats
        include_depth = self.config.get("lambda_depth", 0.0) > 0.0 and camtoworlds.shape[0] == 1 and not use_sh
        sh_degree = None

        if use_sh:
            render_colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
            sh_degree = self.sh_degree
        else:
            # C0 is the standard SH constant (1 / (2 * sqrt(pi))).
            C0 = 0.28209479177387814
            sh0 = self.splats["sh0"].squeeze(1) # [N, 3]
            rgb = sh0 * C0 + 0.5
            render_colors = rgb

        if include_depth:
            means_h = torch.cat([means, torch.ones((means.shape[0], 1), device=self.device)], dim=-1)
            p_cam = (viewmats[0] @ means_h.T).T
            gauss_depths = p_cam[:, 2:3] # [N, 1]
            render_colors = torch.cat([render_colors, gauss_depths], dim=-1) # [N, 4]

        K_batch = K.unsqueeze(0).expand(camtoworlds.shape[0], -1, -1).contiguous()
        
        render_features, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=render_colors,
            viewmats=viewmats,
            Ks=K_batch,
            width=width,
            height=height,
            sh_degree=sh_degree,
            backgrounds=(
                self.background_color.unsqueeze(0).expand(camtoworlds.shape[0], -1).contiguous()
                if self.background_color is not None and not include_depth
                else None
            ),
            packed=False,
            near_plane=0.01,
            far_plane=1000.0, # CRITICAL: Increase for sky coverage
        )

        image_rgb = render_features[..., :3]
        image_depth = render_features[..., 3] if include_depth else None
        
        # Composite with background
        image_rgb = image_rgb + (1.0 - render_alphas) * 0.0 # Black background
        
        # Return both for the train_step to use
        return image_rgb.permute(0, 3, 1, 2), image_depth, info

    def train_step(self, frame_indices: np.ndarray) -> float:
        """Run one training step."""
        # Zero gradients for all optimizers
        for opt in self.optimizers.values():
            opt.zero_grad()

        target_images = self.dataset.get_image_batch(frame_indices).to(self.device)
        masks = torch.stack([self.dataset.get_mask(i) for i in frame_indices]).to(self.device)
        object_mask = (masks < 128).float().unsqueeze(1)
        sky_mask = (masks >= 128).float().unsqueeze(1)
        pose_indices = torch.as_tensor(frame_indices, device=self.device, dtype=torch.long)
        pose_c2w = self.poses_c2w.index_select(0, pose_indices)

        rendered_rgb, rendered_depth, info = self.rasterize_splats(
            pose_c2w,
            self.K,
            self.dataset.image_width,
            self.dataset.image_height,
        )

        object_l1 = masked_l1_loss(rendered_rgb, target_images, object_mask)
        sky_l1 = masked_l1_loss(rendered_rgb, target_images, sky_mask)

        lambda_sky = self.config.get("lambda_sky", 0.0)
        lambda_depth = self.config.get("lambda_depth", 0.0)
        depth_l1 = rendered_rgb.new_tensor(0.0)

        if lambda_depth > 0.0:
            if len(frame_indices) == 1 and rendered_depth is not None:
                target_depth = self.dataset.get_lidar_depth(int(frame_indices[0]), device=self.device)
                depth_mask = ((target_depth > 0.0).float() * object_mask[0, 0]).unsqueeze(0)
                denom = depth_mask.sum().clamp_min(1.0)
                depth_l1 = ((rendered_depth[0] - target_depth).abs() * depth_mask).sum() / denom
            elif not getattr(self, "_warned_depth_batch", False):
                print("[Trainer] Depth loss is skipped for SH rendering or batch_size > 1.")
                self._warned_depth_batch = True

        total_loss = object_l1 + lambda_sky * sky_l1 + lambda_depth * depth_l1
        self.last_loss_stats = {
            "object_l1": object_l1.item(),
            "sky_l1": sky_l1.item(),
            "depth_l1": depth_l1.item(),
            "total": total_loss.item(),
        }
        
        # Pre-backward step
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=info,
        )

        # Backward
        total_loss.backward()

        # Optimizer steps for all parameters
        for opt in self.optimizers.values():
            opt.step()

        # Post-backward step (handles split/clone/prune densification)
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=info,
            packed=False,
        )

        return total_loss.item()

    def train(self, num_epochs: int, batch_size: int = 4):
        """Run training loop."""
        print(f"\n[Trainer] Starting training: {num_epochs} epochs, batch_size={batch_size}\n")

        num_batches = (len(self.dataset) + batch_size - 1) // batch_size

        for epoch in range(num_epochs):
            indices = np.random.permutation(len(self.dataset))

            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0

            for batch_idx in pbar:
                # Get batch
                start = batch_idx * batch_size
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]

                # Train step
                loss = self.train_step(batch_indices)
                epoch_loss += loss

                # Log
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                self.tb_writer.add_scalar('loss/train', loss, self.iteration)
                for name, value in self.last_loss_stats.items():
                    self.tb_writer.add_scalar(f'loss/{name}', value, self.iteration)
                self.tb_writer.add_scalar('gs_count', len(self.splats["means"]), self.iteration)
                self.tb_writer.add_scalar('lr/means', self.optimizers["means"].param_groups[0]["lr"], self.iteration)
                
                self.scheduler.step()
                self.iteration += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch + 1)


    def save_checkpoint(self, epoch: int):
        """Save checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'iteration': self.iteration,
            'sh_degree': self.sh_degree,
            'splats': {k: v.data for k, v in self.splats.items()},
            'optimizers': {k: opt.state_dict() for k, opt in self.optimizers.items()},
            'scheduler': self.scheduler.state_dict(),
            'strategy_state': self.strategy_state,
        }, ckpt_path)
        print(f"[Trainer] Saved checkpoint: {ckpt_path}")

    def save_ply(self, filename: str = "gaussians.ply"):
        """Export as PLY file."""
        import struct

        means = self.splats["means"].detach().cpu().numpy()
        scales = self.splats["scales"].detach().cpu().numpy()
        quats = self.splats["quats"].detach().cpu().numpy()
        opacities = self.splats["opacities"].squeeze(-1).detach().cpu().numpy()
        sh0 = self.splats["sh0"].squeeze(1).detach().cpu().numpy()
        shN = self.splats["shN"].detach().cpu().numpy() if "shN" in self.splats else None
        rest_count = 0 if shN is None else shN.shape[1] * shN.shape[2]

        num_points = len(means)

        ply_path = self.output_dir / filename

        with open(ply_path, 'wb') as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float scale_0\n")
            f.write(b"property float scale_1\n")
            f.write(b"property float scale_2\n")
            f.write(b"property float rot_0\n")
            f.write(b"property float rot_1\n")
            f.write(b"property float rot_2\n")
            f.write(b"property float rot_3\n")
            f.write(b"property float opacity\n")
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            for j in range(rest_count):
                f.write(f"property float f_rest_{j}\n".encode())
            f.write(b"end_header\n")

            # Data
            for i in range(num_points):
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
                if shN is not None:
                    values.extend(shN[i].reshape(-1))
                f.write(struct.pack(f'{len(values)}f', *values))

        print(f"[Trainer] Saved PLY: {ply_path}")

    def load_checkpoint(self, ckpt_path: str):
        """Load a saved checkpoint and resume training."""
        print(f"[Trainer] Loading checkpoint from: {ckpt_path}")
        # Load the data to the current device
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Restore the iteration count
        self.iteration = checkpoint.get('iteration', 0)

        # Restore parameter data
        # We wrap the saved tensors back into nn.Parameters to maintain gradient flow
        splat_data = checkpoint['splats']
        for k in self.splats.keys():
            if k in splat_data:
                self.splats[k] = nn.Parameter(splat_data[k].to(self.device))
            else:
                print(f"[Warning] Key {k} not found in checkpoint.")
        if "shN" in splat_data and "shN" not in self.splats:
            self.splats["shN"] = nn.Parameter(splat_data["shN"].to(self.device))
            inferred_bases = 1 + self.splats["shN"].shape[1]
            self.sh_degree = int(round(np.sqrt(inferred_bases) - 1))
            self.config["sh_degree"] = self.sh_degree
            print(f"[Trainer] Inferred sh_degree={self.sh_degree} from checkpoint.")

        # CRITICAL: Re-initialize optimizers
        # Old optimizers are tied to the memory addresses of the old parameters
        self.optimizers = create_optimizers_for_splats(self.splats, self.config)

        if "optimizers" in checkpoint:
            for name, state in checkpoint["optimizers"].items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(state)

        # Restore the scheduler state for the new 'means' optimizer
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=self.config['lr_decay'])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Older checkpoints did not store scheduler state.
            gamma = self.config['lr_decay']
            for group in self.optimizers["means"].param_groups:
                base_lr = group.get("initial_lr", self.config['lr_xyz'])
                group["lr"] = base_lr * (gamma ** self.iteration)
            self.scheduler.last_epoch = self.iteration

        if "strategy_state" in checkpoint:
            self.strategy_state = checkpoint["strategy_state"]

        print(f"[Trainer] Resuming from iteration {self.iteration}")


def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting on a Track dataset.")
    parser.add_argument("--track", default="Track2", choices=["Track1", "Track2", "Track3"])
    parser.add_argument("--data-root", type=Path, default=repo_dir / "data")
    parser.add_argument("--data-dir", type=Path, help="Dataset root. Defaults to data-root/track.")
    parser.add_argument("--pointcloud-file", type=Path, help="Colored PCD to initialize Gaussians from.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sky-points", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--init-scale-lidar", type=float, default=0.15)
    parser.add_argument("--init-scale-sky", type=float, default=0.5)
    parser.add_argument("--init-opacity", type=float, default=0.7)
    parser.add_argument("--refine-start-iter", type=int, default=199)
    parser.add_argument("--refine-stop-iter", type=int, default=2001)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=3000)
    parser.add_argument("--lambda-sky", type=float, default=0.1)
    parser.add_argument("--lambda-depth", type=float, default=0.0)
    parser.add_argument(
        "--background-color",
        default="black",
        help="Background for uncovered pixels: black, white, auto, or R,G,B. Use auto for Track1 simple mode.",
    )
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--lr-xyz", type=float, default=0.00016)
    parser.add_argument("--lr-color", type=float, default=0.0025)
    parser.add_argument("--lr-color-rest", type=float, default=0.000125)
    parser.add_argument("--lr-opacity", type=float, default=0.02)
    parser.add_argument("--lr-scaling", type=float, default=0.003)
    parser.add_argument("--lr-rotation", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9999)
    parser.add_argument("--prune-opa", type=float, default=0.005)
    parser.add_argument("--grow-grad2d", type=float, default=0.0001)
    parser.add_argument("--grow-scale3d", type=float, default=0.01)
    parser.add_argument("--grow-scale2d", type=float, default=0.05)
    parser.add_argument("--prune-scale3d", type=float, default=0.15)
    parser.add_argument("--prune-scale2d", type=float, default=0.15)
    parser.add_argument("--ply-name", default="gaussian_reconstruction.ply")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir or (args.data_root / args.track)

    config = {
        'lr_xyz': args.lr_xyz,
        'lr_color': args.lr_color,
        'lr_opacity': args.lr_opacity,
        'lr_scaling': args.lr_scaling,
        'lr_rotation': args.lr_rotation,
        'lr_decay': args.lr_decay,
        'checkpoint_interval': args.checkpoint_interval,
        'init_scale_lidar': args.init_scale_lidar,
        'init_scale_sky': args.init_scale_sky,
        'init_opacity': args.init_opacity,
        'refine_start_iter': args.refine_start_iter,
        'refine_stop_iter': args.refine_stop_iter,
        'refine_every': args.refine_every,
        'reset_every': args.reset_every,
        'lambda_sky': args.lambda_sky,
        'lambda_depth': args.lambda_depth,
        'background_color': args.background_color,
        'sh_degree': args.sh_degree,
        'prune_opa': args.prune_opa,
        'lr_color_rest': args.lr_color_rest,
        'grow_grad2d': args.grow_grad2d,
        'grow_scale3d': args.grow_scale3d,
        'grow_scale2d': args.grow_scale2d,
        'prune_scale3d': args.prune_scale3d,
        'prune_scale2d': args.prune_scale2d,
        'sky_points': args.sky_points,
    }

    trainer = Trainer(
        data_dir,
        args.output_dir,
        config,
        force_cpu=args.cpu,
        track=args.track,
        pointcloud_file=args.pointcloud_file,
    )

    if args.checkpoint and args.checkpoint.exists():
        trainer.load_checkpoint(str(args.checkpoint))

    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)
    trainer.save_ply(args.ply_name)
    print("\nTraining complete!")
