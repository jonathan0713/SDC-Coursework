# How To Run: SLAM Data Preparation + 3DGS Training

This file is the end-to-end checklist for preparing a Track dataset, training 3D Gaussian Splatting, and rendering a Kaggle submission.

The expected repository root is:

```bash
~/Documents/ROS1/sdc-midterm2-5-2026
```

Most examples below use `Track2`. Replace `Track2` with `Track1` when needed.

## 0. Data Layout

Raw competition data should already look like this:

```text
data/TrackX/
├── data/
│   ├── image/
│   └── raw_pcd/
└── result/
```

The full pipeline creates these important files:

```text
data/TrackX/result/
├── lidar_poses.csv
├── camera_pose.csv
├── TrackX.pcd
└── sky_masks/
```

3DGS training then creates:

```text
sdc_3dgs_reconstruction/output/trackX/
├── gaussian_reconstruction.ply
└── runs/
```

## 1. LiDAR SLAM Environment

Use the SLAM environment for `lidar-slam`. This environment should use `numpy<2` because several robotics/vision wheels can have NumPy ABI issues.

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/lidar-slam

uv venv --python 3.9 .venv
source .venv/bin/activate

uv pip install "numpy<2" pandas scipy tqdm opencv-python open3d gtsam small_gicp
```

Quick check:

```bash
python - <<'PY'
import numpy, open3d, small_gicp
print("numpy", numpy.__version__)
print("slam env ok")
PY
```

## 2. Generate LiDAR Poses

Run SLAM from the `lidar-slam` directory.

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/lidar-slam
source .venv/bin/activate

SLAM_TARGET=Track2 \
SLAM_ENABLE_VIZ=0 \
python main.py
```

Expected output:

```text
lidar-slam/Track2/lidar_poses.csv
```

Copy it into the dataset result folder used by the 3DGS preparation scripts:

```bash
mkdir -p ../data/Track2/result
cp Track2/lidar_poses.csv ../data/Track2/result/lidar_poses.csv
```

Optional smoke run:

```bash
SLAM_TARGET=Track2 SLAM_MAX_FRAMES=20 SLAM_ENABLE_VIZ=0 python main.py
```

## 3. Generate Camera Poses

`camera_pose.py` interpolates LiDAR poses to image timestamps and applies the LiDAR-to-camera extrinsic.

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/lidar-slam
source .venv/bin/activate

python process/camera_pose.py --track Track2
```

Expected output:

```text
data/Track2/result/camera_pose.csv
```

## 4. Build Colored PCD Map

`color_mapping.py` merges raw PCD frames using `lidar_poses.csv`, removes outliers, and projects image colors onto the point cloud using `camera_pose.csv`.

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/lidar-slam
source .venv/bin/activate

python process/color_mapping.py --track Track2
```

Expected output:

```text
data/Track2/result/Track2.pcd
```

Optional visualization:

```bash
python process/color_mapping.py --track Track2 --visualize
```

For Track3, prefer the normal-aware color projection to reduce colors bleeding through walls:

```bash
python process/color_mapping_normals.py --track Track3
```

This also writes:

```text
data/Track3/result/Track3.pcd
```

To keep a separate output instead of replacing the default PCD:

```bash
python process/color_mapping_normals.py \
  --track Track3 \
  --output ../data/Track3/result/Track3_normals.pcd
```

Useful tuning knobs:

```bash
python process/color_mapping.py \
  --track Track2 \
  --map-voxel 0.1 \
  --final-voxel 0.2 \
  --min-range 0.0 \
  --max-range 0.0
```

## 5. 3DGS Environment With RTX 50 Series

RTX 50 series GPUs require `sm_120` support. The older prebuilt `gsplat` wheel for `torch 2.4.0 + cu124` does not support RTX 5070 Ti.

Use a newer PyTorch CUDA build and compile/JIT-build `gsplat` with CUDA Toolkit 12.8.

### 5.1 Install CUDA Toolkit 12.8

Only install the toolkit. Do not install the driver bundle.

```bash
cd /tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8
```

If apt cannot find `cuda-toolkit-12-8`, make sure the CUDA repo is enabled:

```bash
sudo cp /etc/apt/sources.list.d/disabled/cuda-ubuntu2204-x86_64.list \
  /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

sudo apt-get update
apt-cache policy cuda-toolkit-12-8
sudo apt-get install cuda-toolkit-12-8
```

Verify:

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

### 5.2 Create uv Environment

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/sdc_3dgs_reconstruction

uv venv --python 3.11 .venv-gsplat-src
source .venv-gsplat-src/bin/activate
```

Set CUDA variables for this shell:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
```

Install PyTorch CUDA 12.8:

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install project dependencies:

```bash
uv pip install opencv-python open3d scipy tensorboard numpy tqdm transformers timm pillow rich ninja packaging
```

Install `gsplat`:

```bash
uv pip install --no-build-isolation --no-binary gsplat "gsplat==1.5.3"
```

If the source build is not available from PyPI, install normally and let `gsplat` JIT-build on first import:

```bash
uv pip install "gsplat==1.5.3"
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("arch list:", torch.cuda.get_arch_list())

x = torch.randn(512, 512, device="cuda")
y = x @ x
print("matmul ok:", y.shape)

from gsplat.cuda import _backend
print("gsplat backend:", _backend._C is not None)
PY
```

Expected:

```text
cuda available: True
arch list includes sm_120 or compute_120
matmul ok
gsplat backend: True
```

## 6. Generate Sky Masks

This downloads a HuggingFace SegFormer model the first time it runs.

```bash
cd ~/Documents/ROS1/sdc-midterm2-5-2026/sdc_3dgs_reconstruction
source .venv-gsplat-src/bin/activate

python generate_sky_mask.py --track Track2
```

Expected output:

```text
data/Track2/result/sky_masks/*.png
```

## 7. Check Camera Poses Against Map

Create a PLY that contains the colored point cloud and red camera centers.

```bash
python test_camera_poses.py \
  --track Track2 \
  --output track2/visualization.ply
```

Open `track2/visualization.ply` in CloudCompare, MeshLab, or another PLY viewer. Check that:

- the trajectory follows the road/map
- camera centers are not floating far away
- the map does not look heavily twisted or mirrored

## 8. Train 3DGS

```bash
python train.py \
  --track Track2 \
  --output-dir output/track2 \
  --epochs 20 \
  --batch-size 8
```

Expected final output:

```text
output/track2/gaussian_reconstruction.ply
```

TensorBoard:

```bash
tensorboard --logdir output/track2/runs
```

If GPU memory is tight, reduce batch size:

```bash
python train.py --track Track2 --output-dir output/track2 --epochs 20 --batch-size 4
```

## 9. Render Submission Images

Generate poses for official test frames:

```bash
python generate_test_pose.py --track Track2
```

Expected output:

```text
track2/test_pose_list.txt
```

Render test images:

```bash
python generate_submission.py \
  --track Track2 \
  --ply output/track2/gaussian_reconstruction.ply \
  --output-dir track2/test_submission
```

Expected output:

```text
track2/test_submission/
├── *.png
└── submission.csv
```

Zip for Kaggle:

```bash
cd track2/test_submission
zip -r ../submission.zip .
```

## 10. Common Errors

### `gsplat: No CUDA toolkit found. gsplat will be disabled.`

`gsplat` cannot find `nvcc`. Fix:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
which nvcc
```

Then reinstall or re-import `gsplat`.

### `no kernel image is available for execution on the device`

PyTorch does not support the GPU architecture. RTX 5070 Ti needs `sm_120`.

Do not use:

```text
torch 2.4.0+cu124
gsplat 1.5.3+pt24cu124
```

Use a newer PyTorch CUDA 12.8 build and source/JIT-build `gsplat`.

### `FileNotFoundError: gaussian_reconstruction.ply`

Training did not finish, or `--ply` points to the wrong output directory.

Check:

```bash
find output -name 'gaussian_reconstruction.ply'
```

### `Unable to locate package cuda-toolkit-12-8`

The CUDA apt repo is not enabled. Enable it:

```bash
sudo cp /etc/apt/sources.list.d/disabled/cuda-ubuntu2204-x86_64.list \
  /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
sudo apt-get update
```

### NumPy / pandas / cv2 ABI errors

Use separate environments:

- SLAM: Python 3.9 and `numpy<2`
- 3DGS: Python 3.11 and CUDA/PyTorch environment

Do not reuse one venv for both.
