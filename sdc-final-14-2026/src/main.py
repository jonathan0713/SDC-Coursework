from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# make runner/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from runner.configs import TrackerConfig
from sdc_tracker import Tracker
from utils.camera_gate import CameraGate
from tools.debug_logger import DebugLogger
from utils.paths import DATA_ROOT, DEFAULT_CONFIG, OUTPUT_ROOT
from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    save_tracking_mask_to_csv,
    separate_static_and_moving,
)
from utils.detection import build_detections, calculate_cluster_centroids
from utils.viz import (
    plot_camera_gate_panel,
    plot_camera_panel,
    plot_confirmed_tracks,
    plot_kf_predictions,
    plot_moving_clusters,
    plot_static_points,
    plot_unmatched_clusters,
    plot_unmatched_tracks,
    save_frames_as_video,
    _CMAP,
)



def generate_cluster_track_dict(
    confirmed_tracks: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    cluster_points_dict: dict[int, np.ndarray],
):
    """
    Convert confirmed tracks into cluster_id -> track_id mapping.

    Expected confirmed_tracks format from Tracker.update():
        [x, y, z, cluster_id, track_id]

    Returned format:
        {cluster_id: track_id}
    """
    cluster_track_dict = {}

    # TODO-2
    # ================================ TODO: Implementation Starts Here ================================
    #
    # Generate the dictionary needed by save_tracking_mask_to_csv().
    #
    # Suggested logic:
    #   1. Iterate over confirmed_tracks.
    #   2. Read cluster_id and track_id from each confirmed track.
    #   3. Store cluster_track_dict[cluster_id] = track_id.
    #
    # If you design a different track output format, modify this function.
    #
    for track in confirmed_tracks:
        cluster_id = int(track[3])
        track_id = int(track[4])
        cluster_track_dict[cluster_id] = track_id
    # ================================ TODO: Implementation Ends Here ================================

    return cluster_track_dict


def _setup_ax(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Lateral (–Y, m)")
    ax.set_ylabel("Forward (X, m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _plot_cluster_panel(
    ax: plt.Axes,
    static_radar_pc: np.ndarray,
    moving_radar_pc: np.ndarray,
    moving_mask: np.ndarray,
    frame_idx: int,
) -> None:
    plot_static_points(ax, static_radar_pc)
    for cid in np.unique(moving_mask):
        if cid < 0:
            continue
        pts = moving_radar_pc[moving_mask == cid]
        color = _CMAP(int(cid) % 20)
        ax.scatter(-pts[:, 1], pts[:, 0], s=14, color=color, alpha=0.85, linewidths=0)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ax.annotate(str(int(cid)), xy=(-cy, cx), xytext=(-cy + 0.6, cx + 0.6),
                    fontsize=9, fontweight="bold", color=color, va="bottom", ha="left")
    _setup_ax(ax, f"Mask clusters | frame {frame_idx:04d}")


def visualize_tracking(
    static_radar_pc: np.ndarray,
    moving_radar_pc: np.ndarray,
    moving_mask: np.ndarray,
    confirmed_tracks: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    frame_idx: int,
    output_folder: Path,
    matched_track_ids: set[int] | None = None,
    frame_img: np.ndarray | None = None,
    cam_viz: dict | None = None,
    unmatched_tracks: "np.ndarray | None" = None,
    kf_predictions: "np.ndarray | None" = None,
):
    """
    Render a 2×2 visualisation frame:
      top-left  : raw radar cluster mask
      top-right : tracker output (matched tracks + unmatched clusters)
      bot-left  : clean camera frame
      bot-right : YOLO bounding boxes + SAM2 frozen-track masks (camera gate)
    """
    output_img_folder = output_folder / "tracking_vis"
    output_img_folder.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 14))
    gs  = plt.GridSpec(2, 2, figure=fig,
                       height_ratios=[1.35, 1.0],
                       hspace=0.28, wspace=0.18)
    ax_tl = fig.add_subplot(gs[0, 0])   # top-left:   clusters
    ax_tr = fig.add_subplot(gs[0, 1])   # top-right:  tracking
    ax_bl = fig.add_subplot(gs[1, 0])   # bot-left:   camera
    ax_br = fig.add_subplot(gs[1, 1])   # bot-right:  YOLO + SAM2

    # ── top-left: raw cluster mask ────────────────────────────────────────────
    _plot_cluster_panel(ax_tl, static_radar_pc, moving_radar_pc, moving_mask, frame_idx)

    # ── top-right: tracker output ─────────────────────────────────────────────
    if matched_track_ids is not None:
        matched_tracks = np.array(
            [t for t in confirmed_tracks if int(t[4]) in matched_track_ids], dtype=float,
        ) if any(int(t[4]) in matched_track_ids for t in confirmed_tracks) else np.empty((0, 5))
    else:
        matched_tracks = confirmed_tracks

    plot_static_points(ax_tr, static_radar_pc)
    plot_moving_clusters(ax_tr, moving_radar_pc, moving_mask, matched_tracks)
    plot_unmatched_clusters(ax_tr, cluster_centroids, matched_tracks)
    if kf_predictions is not None and len(kf_predictions) > 0:
        plot_kf_predictions(ax_tr, kf_predictions)
    plot_confirmed_tracks(ax_tr, matched_tracks)
    if unmatched_tracks is not None and len(unmatched_tracks) > 0:
        plot_unmatched_tracks(ax_tr, unmatched_tracks)
    _setup_ax(ax_tr, f"Tracking | frame {frame_idx:04d}")

    # ── bot-left: clean camera frame ──────────────────────────────────────────
    # Prefer image from cam_viz (already loaded), then frame_img, then nothing.
    # Explicit None checks — numpy arrays don't support truth-value coercion.
    _cg_img = cam_viz.get("img") if cam_viz is not None else None
    disp_img = _cg_img if _cg_img is not None else frame_img
    plot_camera_panel(ax_bl, disp_img, frame_idx)

    # ── bot-right: YOLO detections + SAM2 frozen masks ────────────────────────
    plot_camera_gate_panel(ax_br, cam_viz, frame_idx)

    fig.savefig(output_img_folder / f"frame_{frame_idx:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)



def run_sequence(
    data_root: str,
    seq: str,
    output_root: str,
    save_csv: bool,
    plot_every: int,
    config: TrackerConfig,
    debug_log: bool = False,
):
    data_root = Path(data_root)
    seq_dir = data_root / seq

    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"
    image_dir = seq_dir / "image"

    if not mask_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {seq_dir}")

    output_root = Path(output_root)
    output_folder = output_root / f"{seq}_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)

    result_csv = output_folder / "result.csv"

    if result_csv.exists():
        result_csv.unlink()

    log_path = output_folder / "debug.jsonl"
    logger = DebugLogger(log_path) if debug_log else None
    tracker = Tracker.from_config(config)
    tracker.debug_logger = logger

    if config.camera_gate.enabled:
        tracker.camera_gate = CameraGate(config.camera_gate, seq_dir)

    frame_names = list_frame_names(mask_dir)

    print("============================================================")
    print("Run student tracking starter")
    print("============================================================")
    print(f"seq: {seq}")
    print(f"frames: {len(frame_names)}")
    print(f"output: {output_folder}")
    print(config)
    print("note: this is incomplete starter code. Implement TODO-1 to TODO-6.")

    for frame_idx, mask_name in enumerate(frame_names):
        # Load frame image when needed:
        #   - every frame when camera gate is enabled (needed for YOLO + SAM2)
        #   - plotted frames when camera gate is off (needed for camera panel display)
        is_plot_frame = plot_every > 0 and frame_idx % plot_every == 0
        need_img = (config.camera_gate.enabled or is_plot_frame) and image_dir.exists()
        frame_img = None
        if need_img:
            img_bgr = cv2.imread(str(image_dir / f"{frame_idx:04d}.jpg"))
            if img_bgr is not None:
                frame_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)

        moving_mask, static_mask, moving_radar_pc, static_radar_pc, moving_indices = separate_static_and_moving(
            mask,
            radar_pc,
        )

        cluster_centroids, cluster_velocities, cluster_points_dict = calculate_cluster_centroids(
            moving_mask,
            moving_radar_pc,
        )

        detections = build_detections(cluster_centroids, cluster_velocities, cluster_points_dict)

        # TODO-5
        # ================================ TODO: Implementation Starts Here ================================
        #
        # Update the tracker with current-frame detections.
        #
        # Suggested logic:
        #   confirmed_tracks = tracker.update(detections)
        #
        # Students may also replace Tracker with their own implementation.
        #
        confirmed_tracks = tracker.update(detections, frame_img=frame_img, frame_idx=frame_idx)
        # ================================ TODO: Implementation Ends Here ================================

        if save_csv:
            cluster_track_dict = generate_cluster_track_dict(
                confirmed_tracks,
                cluster_centroids,
                cluster_points_dict,
            )

            save_tracking_mask_to_csv(
                csv_path=result_csv,
                frame_id=frame_idx,
                mask=mask,
                cluster_track_dict=cluster_track_dict,
                moving_indices=moving_indices,
                radar_points=radar_pc,
            )

        if is_plot_frame:
            # Build camera gate visualization data when gate is active
            cam_viz = None
            if tracker.camera_gate is not None and frame_img is not None:
                # Pass ALL active tracks so SAM2 masks are shown continuously,
                # not only when radar loses the cluster.
                all_p3ds = [
                    (t.track_id, np.array([
                        t.filter.state[0], t.filter.state[1], float(t.det[2])
                    ]))
                    for t in tracker.tracks
                ]
                confirmed_p3ds = [
                    (int(t[4]), np.array([float(t[0]), float(t[1]), float(t[2])]))
                    for t in confirmed_tracks
                ]
                unmatched_p3ds = [
                    (t.track_id, np.array([
                        t.filter.state[0], t.filter.state[1], float(t.det[2])
                    ]))
                    for t in tracker.tracks
                    if t.track_id not in tracker.last_matched_track_ids
                ]
                cam_viz = tracker.camera_gate.render_viz(
                    all_p3ds, confirmed_p3ds, unmatched_p3ds
                )

            _unmatched_arr = np.array(
                [
                    [t.filter.state[0], t.filter.state[1],
                     float(t.det[2]), int(t.det[3]), t.track_id]
                    for t in tracker.tracks
                    if t.track_id not in tracker.last_matched_track_ids
                ],
                dtype=float,
            ) if any(
                t.track_id not in tracker.last_matched_track_ids
                for t in tracker.tracks
            ) else np.empty((0, 5), dtype=float)

            _kf_pred_arr = np.array(
                [
                    [t.last_pred[0], t.last_pred[1],
                     float(t.det[2]), int(t.det[3]), t.track_id]
                    for t in tracker.tracks
                    if t.last_pred is not None
                ],
                dtype=float,
            ) if any(t.last_pred is not None for t in tracker.tracks) else np.empty((0, 5), dtype=float)

            visualize_tracking(
                static_radar_pc=static_radar_pc,
                moving_radar_pc=moving_radar_pc,
                moving_mask=moving_mask,
                confirmed_tracks=confirmed_tracks,
                cluster_centroids=cluster_centroids,
                frame_idx=frame_idx,
                output_folder=output_folder,
                matched_track_ids=tracker.last_matched_track_ids,
                frame_img=frame_img,
                cam_viz=cam_viz,
                unmatched_tracks=_unmatched_arr,
                kf_predictions=_kf_pred_arr,
            )

        if frame_idx % 50 == 0:
            print(
                f"  frame {frame_idx:04d}: "
                f"detections={len(detections)}, confirmed={len(confirmed_tracks)}"
            )

    if plot_every > 0:
        save_frames_as_video(
            image_folder=output_folder / "tracking_vis",
            output_path=output_folder / "tracking.mp4",
        )

    if logger:
        logger.close()
        print(f"[DONE] debug log: {log_path}")

    print(f"[DONE] output folder: {output_folder}")

    if save_csv:
        print(f"[DONE] result csv: {result_csv}")

    return output_folder


_DEFAULT_DATA_ROOT   = DATA_ROOT
_DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT
_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument(
        "--data-root", default=None,
        help=f"Dataset root directory. Defaults to {_DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-root", default=None,
        help=f"Output root directory. Defaults to {_DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a TrackerConfig JSON (e.g. data/optuna/best_config.json). "
             "When given, all tracker parameters are loaded from this file.",
    )

    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Disable writing result.csv. By default, result.csv is always saved.",
    )
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument(
        "--no-kf", action="store_true",
        help="Disable Kalman filter: use raw detection centroid positions directly. "
             "Equivalent to setting filter.type=identity in config.",
    )
    parser.add_argument(
        "--debug-log", action="store_true",
        help="Write per-frame tracker state to debug.jsonl inside the output folder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = Path(args.data_root) if args.data_root else _DEFAULT_DATA_ROOT
    output_root = Path(args.output_root) if args.output_root else _DEFAULT_OUTPUT_ROOT

    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG_PATH
    config = TrackerConfig.load(config_path)
    if args.no_kf:
        config.filter.type = "identity"
    print(f"[config] loaded from {config_path}")
    print(f"[config] filter.type = {config.filter.type}")

    run_sequence(
        data_root=data_root,
        seq=args.seq,
        output_root=output_root,
        save_csv=not args.no_save_csv,
        plot_every=args.plot_every,
        config=config,
        debug_log=args.debug_log,  # bool
    )
