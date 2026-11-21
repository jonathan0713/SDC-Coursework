import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def smooth_translations(poses, window):
    if window <= 1:
        return poses.copy()
    if window % 2 == 0:
        raise ValueError("window must be odd so smoothing is centered")

    smoothed = poses.copy()
    xyz = poses[:, :3, 3]
    radius = window // 2

    for i in range(len(poses)):
        lo = max(0, i - radius)
        hi = min(len(poses), i + radius + 1)
        smoothed[i, :3, 3] = xyz[lo:hi].mean(axis=0)

    # Keep the origin exactly anchored for Kaggle submissions.
    smoothed[0, :3, 3] = poses[0, :3, 3]
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Conservative lidar pose CSV post-processing.")
    parser.add_argument("--input", required=True, help="Input lidar_poses CSV.")
    parser.add_argument("--output", required=True, help="Output lidar_poses CSV.")
    parser.add_argument("--translation-window", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    poses = df.iloc[:, 1:].to_numpy(dtype=float).reshape(-1, 4, 4)
    poses = smooth_translations(poses, args.translation_window)

    out = df.copy()
    out.iloc[:, 1:] = poses.reshape(len(poses), 16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
