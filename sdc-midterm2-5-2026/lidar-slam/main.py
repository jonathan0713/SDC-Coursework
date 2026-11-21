import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from utils.ICP import ICP
from utils.submap import SubmapManager
from utils.keyframe import KeyframeManager
from utils.posegraph import PoseGraph
from utils.loopclosure import LoopClosure
from utils.io import load_folder, load_pcd
from utils.viz import Visualizer


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


class NullVisualizer:
    should_quit = False

    def set_total_frames(self, n):
        pass

    def update(self, curr_pts, global_pts, all_poses, kf_poses):
        pass


def distribute_pose_corrections(poses, kf_indices, kf_opt_poses):
    """Interpolate pose graph corrections to all non-keyframe poses using SLERP."""
    new_poses = [p.copy() for p in poses]

    for k in range(len(kf_indices) - 1):
        idx_a = kf_indices[k]
        idx_b = kf_indices[k + 1]

        orig_a, orig_b = poses[idx_a], poses[idx_b]
        opt_a, opt_b = kf_opt_poses[k], kf_opt_poses[k + 1]

        corr_a = opt_a @ np.linalg.inv(orig_a)
        corr_b = opt_b @ np.linalg.inv(orig_b)

        rot_a = R.from_matrix(corr_a[:3, :3])
        rot_b = R.from_matrix(corr_b[:3, :3])
        trans_a = corr_a[:3, 3]
        trans_b = corr_b[:3, 3]

        key_rots = R.from_quat(np.vstack([rot_a.as_quat(), rot_b.as_quat()]))
        try:
            slerp = Slerp([0, 1], key_rots)
        except Exception:
            slerp = None

        for j in range(idx_a, idx_b):
            t = (j - idx_a) / float(idx_b - idx_a)

            rot_t = slerp(t).as_matrix() if slerp else rot_a.as_matrix()
            trans_t = trans_a + t * (trans_b - trans_a)

            corr_j = np.eye(4)
            corr_j[:3, :3] = rot_t
            corr_j[:3, 3] = trans_t
            new_poses[j] = corr_j @ poses[j]

    if len(kf_indices) > 0:
        last_kf_idx = kf_indices[-1]
        last_corr = kf_opt_poses[-1] @ np.linalg.inv(poses[last_kf_idx])
        for j in range(last_kf_idx, len(poses)):
            new_poses[j] = last_corr @ poses[j]

    return new_poses


# ==========================================
# Main SLAM Pipeline
# ==========================================
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

target = os.environ.get("SLAM_TARGET", "Track3")
folder = data_dir / target / "data" / "raw_pcd"
files = load_folder(folder)
max_frames = env_int("SLAM_MAX_FRAMES", 0)
if max_frames > 0:
    files = files[:max_frames]
print(f"Loaded {len(files)} frames from {folder}")

is_track3 = target.lower() == "track3"

# Track1 is sensitive to small changes; Track3 is meant to exercise loop closure.
ICP_VOXEL_SIZE = env_float("SLAM_ICP_VOXEL", 0.12 if is_track3 else 0.20)
ICP_COARSE_VOXEL = env_float("SLAM_ICP_COARSE_VOXEL", 0.6 if is_track3 else 0.5)
ICP_COARSE_MAX_CORR = env_float("SLAM_ICP_COARSE_MAX_CORR", 2.5 if is_track3 else 1.5)
ICP_FINE_CORR_MULT = env_float("SLAM_ICP_FINE_CORR_MULT", 8.0 if is_track3 else 2.5)
SUBMAP_WINDOW = env_int("SLAM_SUBMAP_WINDOW", 4 if is_track3 else 3)
KEYFRAME_DIST = env_float("SLAM_KEYFRAME_DIST", 0.8 if is_track3 else 1.0)
KEYFRAME_ANGLE = env_float("SLAM_KEYFRAME_ANGLE", 8.0 if is_track3 else 10.0)
INIT_KEYFRAME_MANAGER = env_bool("SLAM_INIT_KEYFRAME_MANAGER", True)
ENABLE_LOOP_CLOSURE = env_bool("SLAM_ENABLE_LOOP", is_track3)
ENABLE_VIZ = env_bool("SLAM_ENABLE_VIZ", True)
VIZ_DELAY_MS = env_int("SLAM_VIZ_DELAY_MS", 50)
OUTPUT_SUFFIX = os.environ.get("SLAM_OUTPUT_SUFFIX", "")

LOOP_DIST_THRESH = env_float("SLAM_LOOP_DIST_THRESH", 20.0)
LOOP_FITNESS_THRESH = env_float("SLAM_LOOP_FITNESS_THRESH", 0.35)
LOOP_MAX_CANDIDATES = env_int("SLAM_LOOP_MAX_CANDIDATES", 3)
LOOP_MIN_INTERVAL = env_int("SLAM_LOOP_MIN_INTERVAL", 25)
LOOP_HEADING_MIN_COS = env_float("SLAM_LOOP_HEADING_MIN_COS", -1.0)
LOOP_CYCLE_ROT_THRESH = env_float("SLAM_LOOP_CYCLE_ROT_THRESH", 0.08)
LOOP_CYCLE_TRANS_THRESH = env_float("SLAM_LOOP_CYCLE_TRANS_THRESH", 0.25)
LOOP_MAX_CORRECTION_TRANS = env_float("SLAM_LOOP_MAX_CORRECTION_TRANS", 8.0)
LOOP_MAX_CORRECTION_Z = env_float("SLAM_LOOP_MAX_CORRECTION_Z", 3.0)
LOOP_MAX_CORRECTION_ROLL_PITCH = env_float("SLAM_LOOP_MAX_CORRECTION_ROLL_PITCH", 15.0)
LOOP_MIN_ACCEPT_INTERVAL = env_int("SLAM_LOOP_MIN_ACCEPT_INTERVAL", 12)
LOOP_DETECT_INTERVAL = env_int("SLAM_LOOP_DETECT_INTERVAL", 5)
LOOP_VOXEL_SIZE = env_float("SLAM_LOOP_VOXEL", 0.6)
LOOP_SUBMAP_WINDOW = env_int("SLAM_LOOP_SUBMAP_WINDOW", 2)

print(
    "[Config] "
    f"target={target}, icp_voxel={ICP_VOXEL_SIZE}, "
    f"coarse=({ICP_COARSE_VOXEL}, {ICP_COARSE_MAX_CORR}), "
    f"fine_mult={ICP_FINE_CORR_MULT}, submap_window={SUBMAP_WINDOW}, "
    f"keyframe=({KEYFRAME_DIST}m, {KEYFRAME_ANGLE}deg), "
    f"init_kf_mgr={INIT_KEYFRAME_MANAGER}, loop={ENABLE_LOOP_CLOSURE}, "
    f"viz={ENABLE_VIZ}, output_suffix='{OUTPUT_SUFFIX}'"
)
if ENABLE_LOOP_CLOSURE:
    print(
        "[Loop] "
        f"dist={LOOP_DIST_THRESH}, fitness={LOOP_FITNESS_THRESH}, "
        f"candidates={LOOP_MAX_CANDIDATES}, min_interval={LOOP_MIN_INTERVAL}, "
        f"heading_cos={LOOP_HEADING_MIN_COS}, "
        f"cycle=({LOOP_CYCLE_ROT_THRESH}, {LOOP_CYCLE_TRANS_THRESH}), "
        f"max_corr=({LOOP_MAX_CORRECTION_TRANS}m, z={LOOP_MAX_CORRECTION_Z}m, "
        f"rp={LOOP_MAX_CORRECTION_ROLL_PITCH}deg), "
        f"min_accept_interval={LOOP_MIN_ACCEPT_INTERVAL}, "
        f"detect_interval={LOOP_DETECT_INTERVAL}, "
        f"loop_voxel={LOOP_VOXEL_SIZE}, loop_submap_window={LOOP_SUBMAP_WINDOW}"
    )

icp = ICP(
    voxel_size=ICP_VOXEL_SIZE,
    coarse_voxel_size=ICP_COARSE_VOXEL,
    coarse_max_corr=ICP_COARSE_MAX_CORR,
    fine_max_corr_multiplier=ICP_FINE_CORR_MULT,
)
submap_mgr = SubmapManager(window_size=SUBMAP_WINDOW)
kf_manager = KeyframeManager(dist_thresh=KEYFRAME_DIST, ang_thresh=KEYFRAME_ANGLE)
pg = PoseGraph()
lc = LoopClosure(
    icp,
    dist_thresh=LOOP_DIST_THRESH,
    fitness_thresh=LOOP_FITNESS_THRESH,
    max_candidates=LOOP_MAX_CANDIDATES,
    min_interval=LOOP_MIN_INTERVAL,
    heading_min_cos=LOOP_HEADING_MIN_COS,
    cycle_rot_thresh=LOOP_CYCLE_ROT_THRESH,
    cycle_trans_thresh=LOOP_CYCLE_TRANS_THRESH,
    max_correction_trans=LOOP_MAX_CORRECTION_TRANS,
    max_correction_z=LOOP_MAX_CORRECTION_Z,
    max_correction_roll_pitch_deg=LOOP_MAX_CORRECTION_ROLL_PITCH,
    min_loop_interval=LOOP_MIN_ACCEPT_INTERVAL,
    detect_interval=LOOP_DETECT_INTERVAL,
    loop_voxel_size=LOOP_VOXEL_SIZE,
    loop_submap_window=LOOP_SUBMAP_WINDOW,
)
viz = Visualizer(delay_ms=VIZ_DELAY_MS) if ENABLE_VIZ else NullVisualizer()
viz.set_total_frames(len(files))

pg.add_prior()

prev_pose = np.eye(4)
poses = []
timestamps = []

full_map = np.zeros((0, 3))
kf_indices = []
kf_poses_list = []
last_kf_pose = np.eye(4)
current_kf_idx = 0

for i, f in enumerate(tqdm(files, desc="SLAM")):
    if viz.should_quit:
        print(f"\n  [Quit] Stopped at frame {i}/{len(files)}")
        break

    pts = load_pcd(f)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    if i == 0:
        poses.append(prev_pose)
        timestamps.append(os.path.basename(f).replace(".pcd", ""))
        submap_mgr.add_keyframe(pts, prev_pose)
        lc.add_keyframe(pts, prev_pose)

        kf_indices.append(i)
        kf_poses_list.append(prev_pose.copy())
        if INIT_KEYFRAME_MANAGER:
            kf_manager.last_pose = prev_pose.copy()
        continue

    submap_pts = submap_mgr.get_latest_submap()
    if submap_pts is None:
        continue

    T, score = icp.align(pts, submap_pts, prev_pose)
    curr_pose = T.copy()

    poses.append(curr_pose)
    timestamps.append(os.path.basename(f).replace(".pcd", ""))

    if kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1

        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose)

        rel_pose = np.linalg.inv(last_kf_pose) @ curr_pose
        pg.add_node(current_kf_idx, curr_pose)
        pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose)

        last_kf_pose = curr_pose.copy()
        kf_indices.append(i)
        kf_poses_list.append(curr_pose.copy())

        pts_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        full_map = np.concatenate([full_map, pts_w], axis=0)
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(full_map)
        pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.5)
        full_map = np.asarray(pcd_tmp.points)

        loop_result = lc.detect(submap_mgr) if ENABLE_LOOP_CLOSURE else None
        if loop_result is not None:
            loop_idx, t_loop, loop_score = loop_result
            pg.add_loop(loop_idx, current_kf_idx, t_loop)
            tqdm.write(
                f"  [Loop Closure] KF {current_kf_idx} <-> KF {loop_idx} "
                f"(score={loop_score})"
            )

            kf_opt_dict = pg.optimize()
            kf_opt_poses = [kf_opt_dict[k] for k in sorted(kf_opt_dict.keys())]

            submap_mgr.update_poses(kf_opt_poses)
            lc.update_poses(kf_opt_poses)

            poses = distribute_pose_corrections(poses, kf_indices, kf_opt_poses)
            curr_pose = kf_opt_poses[-1].copy()
            last_kf_pose = curr_pose.copy()
            kf_poses_list = kf_opt_poses

    prev_pose = curr_pose.copy()

    pts_curr_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
    viz.update(pts_curr_w, full_map, poses, kf_poses_list)
    if viz.should_quit:
        print(f"\n  [Quit] Stopped at frame {i}/{len(files)}")
        break

output_dir = current_dir / target
output_dir.mkdir(parents=True, exist_ok=True)

header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
rows = [[str(t)] + pose.reshape(-1).tolist() for t, pose in zip(timestamps, poses)]

df = pd.DataFrame(rows, columns=header.split(","))
output_name = f"lidar_poses{OUTPUT_SUFFIX}.csv"
output_path = output_dir / output_name
df.to_csv(str(output_path), index=False)

print(f"SLAM process completed. Poses saved to {output_path}")
