import small_gicp as sg
import numpy as np

class ICP:
    def __init__(
        self,
        voxel_size=0.25,
        coarse_voxel_size=0.5,
        coarse_max_corr=1.5,
        fine_max_corr_multiplier=2.5,
        registration_type="GICP",
        coarse_registration_type=None,
        fine_registration_type=None,
        voxel_resolution=1.0,
        coarse_iterations=100,
        fine_iterations=200,
        num_threads=4,
    ):
        self.voxel_size = voxel_size
        self.coarse_voxel_size = coarse_voxel_size
        self.coarse_max_corr = coarse_max_corr
        self.fine_max_corr_multiplier = fine_max_corr_multiplier
        self.registration_type = registration_type
        self.coarse_registration_type = coarse_registration_type or registration_type
        self.fine_registration_type = fine_registration_type or registration_type
        self.voxel_resolution = voxel_resolution
        self.coarse_iterations = coarse_iterations
        self.fine_iterations = fine_iterations
        self.num_threads = num_threads

    def align(self, src_pts, tgt_pts, init_T=np.eye(4)):
        """
        Align source point cloud to target point cloud using GICP.

        Parameters
        ----------
        src_pts : np.ndarray (Nx3)  – current scan (source)
        tgt_pts : np.ndarray (Mx3)  – submap / reference (target)
        init_T  : np.ndarray (4x4)  – initial guess for transformation

        Returns
        -------
        T     : np.ndarray (4x4)  – estimated transformation (target ← source)
        score : float             – number of inliers (higher = better)
        """
        result_coarse = sg.align(
            target_points=tgt_pts.astype(np.float64),
            source_points=src_pts.astype(np.float64),
            init_T_target_source=init_T.astype(np.float64),
            registration_type=self.coarse_registration_type,
            voxel_resolution=self.voxel_resolution,
            downsampling_resolution=self.coarse_voxel_size,
            max_correspondence_distance=self.coarse_max_corr,
            num_threads=self.num_threads,
            max_iterations=self.coarse_iterations,
        )

        result_fine = sg.align(
            target_points=tgt_pts.astype(np.float64),
            source_points=src_pts.astype(np.float64),
            init_T_target_source=result_coarse.T_target_source,
            registration_type=self.fine_registration_type,
            voxel_resolution=self.voxel_resolution,
            downsampling_resolution=self.voxel_size,
            max_correspondence_distance=self.voxel_size * self.fine_max_corr_multiplier,
            num_threads=self.num_threads,
            max_iterations=self.fine_iterations,
        )

        T = np.asarray(result_fine.T_target_source)
        score = result_fine.num_inliers / len(src_pts)

        if not result_fine.converged:
            return T, 0.0

        return T, score
