import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

'''
Make sure you set the parameters.
'''
class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=20.0,
        fitness_thresh=0.35,
        max_candidates=8,
        min_interval=25,
        heading_min_cos=-1.0,
        cycle_rot_thresh=0.08,
        cycle_trans_thresh=0.25,
        max_correction_trans=8.0,
        max_correction_z=3.0,
        max_correction_roll_pitch_deg=15.0,
        min_loop_interval=8,
        detect_interval=5,
        loop_voxel_size=0.6,
        loop_submap_window=2,
    ):
        """
        icp: ICP instance (small_gicp wrapper)
        dist_thresh: max Euclidean distance to consider a candidate (metres)
        fitness_thresh: minimum normalized overlap score to accept a loop
        max_candidates: maximum number of candidates to evaluate
        min_interval: minimum keyframe index gap to avoid matching nearby frames
        """

        self.icp = icp

        # ===== parameter =====
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_interval = min_interval
        self.heading_min_cos = heading_min_cos
        self.cycle_rot_thresh = cycle_rot_thresh
        self.cycle_trans_thresh = cycle_trans_thresh
        self.max_correction_trans = max_correction_trans
        self.max_correction_z = max_correction_z
        self.max_correction_roll_pitch = np.deg2rad(max_correction_roll_pitch_deg)
        self.min_loop_interval = min_loop_interval
        self.detect_interval = detect_interval
        self.loop_voxel_size = loop_voxel_size
        self.loop_submap_window = loop_submap_window

        # ===== state =====
        self.keyframes = []
        self.poses = []

        self.last_loop_accum = 0.0
        self.last_loop_idx = -10**9

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose):
        # local frame points, downsampled for fast loop verification
        self.keyframes.append(self._downsample(pts))

        # world frame poses
        self.poses.append(pose)

    def _downsample(self, pts):
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        if self.loop_voxel_size <= 0:
            return pts

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=self.loop_voxel_size)
        return np.asarray(pcd.points)

    def _build_loop_submap(self, idx):
        start = max(0, idx - self.loop_submap_window)
        end = min(len(self.keyframes), idx + self.loop_submap_window + 1)

        pts_list = []
        for i in range(start, end):
            pts = self.keyframes[i]
            pose = self.poses[i]
            pts_w = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
            pts_list.append(pts_w)

        return np.concatenate(pts_list, axis=0)


    # ==========================
    #    Candidate selection
    # ==========================
    # TODO
    def find_candidates(self, curr_idx):
        """
        Select loop-closure candidates based on spatial proximity
        while enforcing a minimum temporal gap to avoid matching
        neighbouring frames.

        Returns list of candidate keyframe indices.
        """
        if curr_idx - self.last_loop_idx < self.min_loop_interval:
            return []

        candidates = []

        curr_pos = self.poses[curr_idx][:3, 3]
        curr_rot = self.poses[curr_idx][:3, :3]

        for i in range(curr_idx - self.min_interval):
            dist = np.linalg.norm(curr_pos - self.poses[i][:3, 3])
            if dist >= self.dist_thresh:
                continue

            if self.heading_min_cos > -1.0:
                fwd_curr = curr_rot[:, 0]
                fwd_cand = self.poses[i][:3, :3][:, 0]
                cos_sim = np.dot(fwd_curr, fwd_cand)
                if cos_sim < self.heading_min_cos:
                    continue

            candidates.append((dist, i))

        # Sort by distance (closest first) and take top-k
        candidates.sort(key=lambda x: x[0])
        candidates = [idx for _, idx in candidates[:self.max_candidates]]

        return candidates

    def _normalized_score(self, score, num_points):
        if score <= 1.0:
            return float(score)
        return float(score) / max(float(num_points), 1.0)

    def _passes_correction_gate(self, correction):
        trans = correction[:3, 3]
        if np.linalg.norm(trans) > self.max_correction_trans:
            return False
        if abs(trans[2]) > self.max_correction_z:
            return False

        try:
            roll, pitch, _ = R.from_matrix(correction[:3, :3]).as_euler("xyz")
        except ValueError:
            return False
        if max(abs(roll), abs(pitch)) > self.max_correction_roll_pitch:
            return False

        return True

    # =========================
    #       ICP matching
    # =========================
    # TODO
    def match(self, curr_idx, candidates, submap_mgr):
        '''
        curr_idx: current keyframe index
        candidates: candidate indices to verify loop closure
        submap_mgr: use submap_mgr.submaps[idx] for idx-th submap (world frame)
        '''

        if len(candidates) == 0:
            return None

        best_score = -1
        best_idx = None
        best_T = None

        curr_pts = self.keyframes[curr_idx]          # local frame
        curr_pose = self.poses[curr_idx]              # current world pose

        for idx in candidates:
            # Target: the downsampled submap around the candidate keyframe.
            tgt_pts = self._build_loop_submap(idx)

            # Align current scan (local) → candidate submap (world)
            # We need to transform source to world first via curr_pose,
            # but it's easier to just use the raw numpy align API
            # with src in world frame
            src_pts_w = (curr_pose[:3, :3] @ curr_pts.T).T + curr_pose[:3, 3]

            # source → target
            # inv(candidate) @ curr
            init_T = np.eye(4)
            # init_T = np.linalg.inv(self.poses[idx]) @ curr_pose

            T_fwd, score_fwd = self.icp.align(src_pts_w, tgt_pts, init_T=init_T)
            T_bwd, score_bwd = self.icp.align(tgt_pts, src_pts_w, init_T=np.linalg.inv(T_fwd))

            C = T_fwd @ T_bwd
            rot_err   = np.linalg.norm(C[:3, :3] - np.eye(3), 'fro')
            trans_err = np.linalg.norm(C[:3, 3])
            if rot_err > self.cycle_rot_thresh or trans_err > self.cycle_trans_thresh:
                continue

            if not self._passes_correction_gate(T_fwd):
                continue

            score_fwd_n = self._normalized_score(score_fwd, len(src_pts_w))
            score_bwd_n = self._normalized_score(score_bwd, len(tgt_pts))
            score_normalized = np.sqrt(score_fwd_n * score_bwd_n)

            if score_normalized > best_score:
                best_score = score_normalized
                best_idx = idx
                best_T = T_fwd

        # Check if the best match is good enough
        if best_score >= self.fitness_thresh and best_idx is not None:
            # Compute the relative transform between candidate and current keyframe
            # T_loop brings current keyframe to candidate keyframe's frame
            T_loop = np.linalg.inv(self.poses[best_idx]) @ best_T @ curr_pose
            self.last_loop_idx = curr_idx
            return best_idx, T_loop, best_score

        return None

    # =========================
    #        Pose Update
    # =========================
    def update_poses(self, new_kf_poses):
        """ Sync poses and recalculate accumulated distances after PGO """
        # Update poses list
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = new_kf_poses[i].copy()

    # =====    ====================
    #            API
    # =========================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        # Could block some too-early detections
        if curr_idx < self.min_interval:
            return None
        if self.detect_interval > 1 and curr_idx % self.detect_interval != 0:
            return None

        candidates = self.find_candidates(curr_idx)
        if len(candidates) == 0:
            return None

        result = self.match(curr_idx, candidates, submap_mgr)
        return result
