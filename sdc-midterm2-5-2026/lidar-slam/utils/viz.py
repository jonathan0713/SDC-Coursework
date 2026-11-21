import open3d as o3d
import numpy as np
import time

class Visualizer:
    def __init__(self, delay_ms=50):
        """
        Interactive SLAM Visualizer with keyboard controls.

        Controls:
            Space  — Pause / Resume
            N      — Step forward one frame (while paused)
            + / =  — Slow down (increase delay)
            -      — Speed up (decrease delay)
            Q      — Quit
        """
        self.delay_ms = delay_ms       # ms to wait between frames
        self.paused = False
        self.step_once = False
        self.quit_flag = False
        self.frame_idx = 0
        self.total_frames = 0
        self.num_kf = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            self.is_valid = self.vis.create_window(
                window_name="SLAM Viewer", width=1280, height=720
            )
        except Exception as e:
            print(f"[Warning] Failed to create window: {e}. Visualization is disabled.")
            self.is_valid = False
            return

        if not self.is_valid:
            print("[Warning] Failed to create window. Visualization is disabled.")
            return

        # Register key callbacks
        # Space (32) — toggle pause
        self.vis.register_key_callback(32, self._on_space)
        # N (78) — step forward
        self.vis.register_key_callback(78, self._on_step)
        # n (110)
        self.vis.register_key_callback(110, self._on_step)
        # + (61 '=') and (43 '+')
        self.vis.register_key_callback(61, self._on_slower)
        self.vis.register_key_callback(43, self._on_slower)
        # - (45)
        self.vis.register_key_callback(45, self._on_faster)
        # Q (81) and q (113)
        self.vis.register_key_callback(81, self._on_quit)
        self.vis.register_key_callback(113, self._on_quit)

        # Geometries
        self.global_map = o3d.geometry.PointCloud()
        self.current_scan = o3d.geometry.PointCloud()
        self.kf_markers = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()

        self.first_view = True
        self._render_opts_set = False

        print("[Visualizer] Keyboard controls:")
        print("  Space  = Pause/Resume")
        print("  N      = Step (while paused)")
        print("  +/-    = Slower/Faster")
        print("  Q      = Quit")

    # ===== Key callbacks =====
    def _on_space(self, vis):
        self.paused = not self.paused
        state = "PAUSED" if self.paused else "RUNNING"
        print(f"  [Viz] {state}")
        return False

    def _on_step(self, vis):
        if self.paused:
            self.step_once = True
        return False

    def _on_slower(self, vis):
        self.delay_ms = min(self.delay_ms + 50, 2000)
        print(f"  [Viz] delay = {self.delay_ms}ms")
        return False

    def _on_faster(self, vis):
        self.delay_ms = max(self.delay_ms - 50, 0)
        print(f"  [Viz] delay = {self.delay_ms}ms")
        return False

    def _on_quit(self, vis):
        self.quit_flag = True
        print("  [Viz] Quit requested")
        return False

    # ===== Render options =====
    def _setup_render_opts(self):
        """Set render options after first geometry is added (avoids segfault)."""
        if self._render_opts_set:
            return
        try:
            opt = self.vis.get_render_option()
            opt.background_color = np.array([0.05, 0.05, 0.05])
            opt.point_size = 2.0
            self._render_opts_set = True
        except Exception:
            pass

    # ===== Pause loop =====
    def _wait_if_paused(self):
        """Block the main loop while paused, keep polling events so the window stays responsive."""
        while self.paused and not self.step_once and not self.quit_flag:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.05)

        if self.step_once:
            self.step_once = False  # consumed

    def set_total_frames(self, n):
        self.total_frames = n

    # ===== Main update =====
    def update(self, curr_pts, global_pts, all_poses, kf_poses):
        if not self.is_valid or self.quit_flag:
            return

        # Handle pause / step
        self._wait_if_paused()
        if self.quit_flag:
            return

        self.frame_idx = len(all_poses)
        self.num_kf = len(kf_poses)

        # 1. Update Current Scan (Green)
        self.current_scan.points = o3d.utility.Vector3dVector(curr_pts)
        self.current_scan.paint_uniform_color([0.0, 1.0, 0.0])

        # 2. Update Global Map (Gray)
        if global_pts is not None and len(global_pts) > 0:
            self.global_map.points = o3d.utility.Vector3dVector(global_pts)
            self.global_map.paint_uniform_color([0.65, 0.65, 0.65])

        # 3. Update Trajectory (Yellow)
        if len(all_poses) > 1:
            pts_traj = np.array([p[:3, 3] for p in all_poses])
            lines = [[i, i+1] for i in range(len(pts_traj)-1)]
            self.traj.points = o3d.utility.Vector3dVector(pts_traj)
            self.traj.lines = o3d.utility.Vector2iVector(lines)
            self.traj.paint_uniform_color([1.0, 1.0, 0.0])

        # 4. Update Keyframe Markers (Red)
        if len(kf_poses) > 0:
            kf_pts = np.array([p[:3, 3] for p in kf_poses])
            self.kf_markers.points = o3d.utility.Vector3dVector(kf_pts)
            self.kf_markers.paint_uniform_color([1.0, 0.0, 0.0])

        # Add geometries to scene (first time only)
        if self.first_view:
            self.vis.add_geometry(self.global_map)
            self.vis.add_geometry(self.current_scan)
            self.vis.add_geometry(self.kf_markers)
            self.vis.add_geometry(self.traj)
            self.first_view = False
            self._setup_render_opts()

        # Update existing geometries
        self.vis.update_geometry(self.global_map)
        self.vis.update_geometry(self.current_scan)
        self.vis.update_geometry(self.kf_markers)
        self.vis.update_geometry(self.traj)

        # 5. Auto-adjust View
        if len(all_poses) > 0:
            view_ctl = self.vis.get_view_control()
            view_ctl.set_lookat(all_poses[-1][:3, 3])

            if len(all_poses) % 20 == 0:
                view_ctl.set_zoom(0.3)

        self.vis.poll_events()
        self.vis.update_renderer()

        # Frame delay
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

    @property
    def should_quit(self):
        return self.quit_flag
