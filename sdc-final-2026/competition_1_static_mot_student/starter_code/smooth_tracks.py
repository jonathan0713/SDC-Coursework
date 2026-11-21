from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_values(text: str, dtype=float) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray([], dtype=dtype)

    return np.asarray([dtype(float(value)) for value in str(text).strip().split()])


def load_result_csv(csv_path: str | Path) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    csv_path = Path(csv_path)
    masks = {}
    coords = defaultdict(dict)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain id,mask columns: {csv_path}")

        for row in reader:
            row_id = str(row["id"])

            if row_id.endswith("x") or row_id.endswith("y") or row_id.endswith("z"):
                coords[int(row_id[:-1])][row_id[-1]] = parse_values(row["mask"], float)
            else:
                masks[int(row_id)] = parse_values(row["mask"], int).astype(np.int32)

    return masks, coords


def write_result_csv(path: str | Path, masks: dict[int, np.ndarray], coords: dict[int, dict[str, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mask"])
        writer.writeheader()

        for frame_id in sorted(masks):
            writer.writerow({"id": frame_id, "mask": " ".join(map(str, masks[frame_id].astype(int)))})

            for axis in ["x", "y", "z"]:
                if axis in coords.get(frame_id, {}):
                    values = coords[frame_id][axis]
                    writer.writerow({"id": f"{frame_id}{axis}", "mask": " ".join(map(str, values))})


def frame_xy(coords: dict[int, dict[str, np.ndarray]], frame_id: int) -> np.ndarray | None:
    axes = coords.get(frame_id, {})

    if "x" not in axes or "y" not in axes:
        return None

    return np.column_stack([axes["x"], axes["y"]])


def read_ego_positions(path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((0, 3), dtype=float)

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing ego pose file: {path}")

    positions = []

    with path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.asarray(positions, dtype=float)


def ego_offset(ego_positions: np.ndarray, frame_id: int, ego_mode: str) -> np.ndarray:
    if ego_mode == "none" or len(ego_positions) == 0 or frame_id >= len(ego_positions):
        return np.zeros(2, dtype=float)

    ego_x, ego_y = ego_positions[frame_id, :2]

    if ego_mode == "add_xy":
        return np.asarray([ego_x, ego_y], dtype=float)

    if ego_mode == "add_x_neg_y":
        return np.asarray([ego_x, -ego_y], dtype=float)

    if ego_mode == "add_yx":
        return np.asarray([ego_y, ego_x], dtype=float)

    if ego_mode == "add_y_neg_x":
        return np.asarray([ego_y, -ego_x], dtype=float)

    raise ValueError(f"Unknown ego_mode: {ego_mode}")


def split_observations(
    track_id: int,
    observations: list[dict],
    split_gap: int,
    split_jump_distance: float,
) -> list[list[dict]]:
    if not observations:
        return []

    segments = [[observations[0]]]

    for obs in observations[1:]:
        prev = segments[-1][-1]
        frame_gap = int(obs["frame"]) - int(prev["frame"])
        jump_distance = float(np.linalg.norm(obs["xy"] - prev["xy"]))
        should_split = False

        if split_gap > 0 and frame_gap > split_gap:
            should_split = True

        if split_jump_distance > 0 and jump_distance > split_jump_distance:
            should_split = True

        if should_split:
            segments.append([obs])
        else:
            segments[-1].append(obs)

    return segments


def build_tracklets(
    masks: dict[int, np.ndarray],
    coords: dict[int, dict[str, np.ndarray]],
    split_gap: int = 0,
    split_jump_distance: float = 0.0,
    ego_positions: np.ndarray | None = None,
    ego_mode: str = "none",
    trend_fit_tail: int = 8,
) -> dict[int, dict]:
    observations = defaultdict(list)
    ego_positions = np.zeros((0, 3), dtype=float) if ego_positions is None else ego_positions

    for frame_id in sorted(masks):
        mask = masks[frame_id]
        xy = frame_xy(coords, frame_id)

        if xy is None or len(xy) != len(mask):
            continue

        tracking_xy = xy + ego_offset(ego_positions, frame_id, ego_mode)

        for track_id in np.unique(mask[mask >= 0]):
            track_id = int(track_id)
            idx = mask == track_id
            observations[track_id].append(
                {
                    "frame": frame_id,
                    "count": int(idx.sum()),
                    "xy": tracking_xy[idx].mean(axis=0),
                    "points_xy": tracking_xy[idx].copy(),
                }
            )

    tracklets = {}
    segment_id = 0

    for track_id, obs in observations.items():
        obs = sorted(obs, key=lambda row: row["frame"])
        segments = split_observations(track_id, obs, split_gap, split_jump_distance)

        for segment_index, segment_obs in enumerate(segments):
            frames = [row["frame"] for row in segment_obs]
            trend_end_xy, trend_velocity = estimate_regression_motion(segment_obs, tail=trend_fit_tail, anchor="end")
            trend_start_xy, trend_start_velocity = estimate_regression_motion(segment_obs, tail=trend_fit_tail, anchor="start")
            tracklets[segment_id] = {
                "segment_id": segment_id,
                "track_id": track_id,
                "segment_index": segment_index,
                "observations": segment_obs,
                "start": frames[0],
                "end": frames[-1],
                "start_xy": segment_obs[0]["xy"],
                "end_xy": segment_obs[-1]["xy"],
                "start_count": segment_obs[0]["count"],
                "end_count": segment_obs[-1]["count"],
                "frames": set(frames),
                "length": len(frames),
                "velocity": estimate_velocity(segment_obs),
                "start_velocity": estimate_start_velocity(segment_obs),
                "trend_end_xy": trend_end_xy,
                "trend_start_xy": trend_start_xy,
                "trend_velocity": trend_velocity,
                "trend_start_velocity": trend_start_velocity,
            }
            segment_id += 1

    return tracklets


def estimate_velocity(observations: list[dict], tail: int = 4) -> np.ndarray:
    if len(observations) < 2:
        return np.zeros(2, dtype=float)

    obs = observations[-tail:]
    first = obs[0]
    last = obs[-1]
    dt = max(int(last["frame"]) - int(first["frame"]), 1)
    return (last["xy"] - first["xy"]) / dt


def estimate_start_velocity(observations: list[dict], tail: int = 4) -> np.ndarray:
    if len(observations) < 2:
        return np.zeros(2, dtype=float)

    obs = observations[:tail]
    first = obs[0]
    last = obs[-1]
    dt = max(int(last["frame"]) - int(first["frame"]), 1)
    return (last["xy"] - first["xy"]) / dt


def estimate_regression_motion(
    observations: list[dict],
    tail: int = 8,
    anchor: str = "end",
) -> tuple[np.ndarray, np.ndarray]:
    if not observations:
        return np.zeros(2, dtype=float), np.zeros(2, dtype=float)

    if anchor == "start":
        obs = observations[:tail] if tail > 0 else observations
        anchor_frame = int(obs[0]["frame"])
    elif anchor == "end":
        obs = observations[-tail:] if tail > 0 else observations
        anchor_frame = int(obs[-1]["frame"])
    else:
        raise ValueError(f"Unknown regression anchor: {anchor}")

    if len(obs) < 2:
        return np.asarray(obs[0]["xy"], dtype=float).copy(), np.zeros(2, dtype=float)

    frames = np.asarray([int(row["frame"]) for row in obs], dtype=float)
    points = np.asarray([row["xy"] for row in obs], dtype=float)

    if np.ptp(frames) <= 0:
        return points[-1].copy(), np.zeros(2, dtype=float)

    frame_center = float(frames.mean())
    design = np.column_stack([frames - frame_center, np.ones(len(frames), dtype=float)])
    coeff, *_ = np.linalg.lstsq(design, points, rcond=None)
    velocity = coeff[0]
    intercept = coeff[1]
    anchor_xy = (float(anchor_frame) - frame_center) * velocity + intercept
    return anchor_xy.astype(float), velocity.astype(float)


def has_frame_overlap(track_a: dict, track_b: dict) -> bool:
    return bool(track_a["frames"] & track_b["frames"])


def point_footprint(points_xy: np.ndarray) -> dict[str, float]:
    if len(points_xy) < 2:
        return {"diameter": 0.0, "pca_length": 0.0, "pca_width": 0.0}

    centered = points_xy - points_xy.mean(axis=0)
    diameter = float(np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2).max())
    covariance = (centered.T @ centered) / max(len(points_xy) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    pca_length = float(2.0 * np.sqrt(max(eigenvalues[0], 0.0)))
    pca_width = float(2.0 * np.sqrt(max(eigenvalues[-1], 0.0)))
    return {"diameter": diameter, "pca_length": pca_length, "pca_width": pca_width}


def vector_angle(vec_a: np.ndarray, vec_b: np.ndarray, min_norm: float) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a < min_norm or norm_b < min_norm:
        return 0.0

    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def trend_gate_metrics(
    source: dict,
    target: dict,
    predicted_xy: np.ndarray,
    source_direction: np.ndarray,
    source_anchor_xy: np.ndarray,
    target_direction: np.ndarray,
    min_trend_speed: float,
) -> dict:
    direction = np.asarray(source_direction, dtype=float)
    direction_norm = float(np.linalg.norm(direction))

    if direction_norm < min_trend_speed:
        return {
            "enabled": False,
            "lateral_error": 0.0,
            "forward_error": 0.0,
            "backward_error": 0.0,
            "link_angle": 0.0,
            "velocity_angle": 0.0,
        }

    unit = direction / direction_norm
    error = np.asarray(target["start_xy"], dtype=float) - predicted_xy
    forward_error = float(np.dot(error, unit))
    lateral_vec = error - forward_error * unit
    lateral_error = float(np.linalg.norm(lateral_vec))
    link_vec = np.asarray(target["start_xy"], dtype=float) - np.asarray(source_anchor_xy, dtype=float)

    return {
        "enabled": True,
        "lateral_error": lateral_error,
        "forward_error": max(forward_error, 0.0),
        "backward_error": max(-forward_error, 0.0),
        "link_angle": vector_angle(direction, link_vec, min_trend_speed),
        "velocity_angle": vector_angle(direction, np.asarray(target_direction, dtype=float), min_trend_speed),
    }


def find_merge_pairs(
    tracklets: dict[int, dict],
    max_gap: int,
    max_distance: float,
    low_point_threshold: int,
    min_source_length: int,
    max_cost: float,
    max_target_length: int,
    max_velocity_diff: float,
    velocity_weight: float,
    max_reconnect_lateral: float,
    max_reconnect_forward: float,
    max_reconnect_backward: float,
    max_reconnect_link_angle: float,
    max_reconnect_velocity_angle: float,
    min_trend_speed: float,
    reconnect_motion_model: str,
    cross_id_only: bool,
) -> list[dict]:
    pairs = []
    active_tracklets = sorted(tracklets.values(), key=lambda row: (row["start"], row["segment_id"]))

    for target in active_tracklets:
        if target["start_count"] > low_point_threshold:
            continue

        if max_target_length > 0 and target["length"] > max_target_length:
            continue

        best = None

        for source in active_tracklets:
            if source["segment_id"] == target["segment_id"]:
                continue

            if cross_id_only and source["track_id"] == target["track_id"]:
                continue

            if source["length"] < min_source_length:
                continue

            gap = target["start"] - source["end"]

            if gap <= 0 or gap > max_gap:
                continue

            if has_frame_overlap(source, target):
                continue

            if reconnect_motion_model == "regression":
                source_anchor_xy = source["trend_end_xy"]
                source_direction = source["trend_velocity"]
                target_direction = target["trend_start_velocity"]
            elif reconnect_motion_model == "endpoint":
                source_anchor_xy = source["end_xy"]
                source_direction = source["velocity"]
                target_direction = target["start_velocity"]
            else:
                raise ValueError(f"Unknown reconnect motion model: {reconnect_motion_model}")

            predicted_xy = source_anchor_xy + source_direction * gap
            distance = float(np.linalg.norm(target["start_xy"] - predicted_xy))

            if distance > max_distance:
                continue

            trend = trend_gate_metrics(
                source,
                target,
                predicted_xy=predicted_xy,
                source_direction=source_direction,
                source_anchor_xy=source_anchor_xy,
                target_direction=target_direction,
                min_trend_speed=min_trend_speed,
            )

            if trend["enabled"]:
                if max_reconnect_lateral > 0 and trend["lateral_error"] > max_reconnect_lateral:
                    continue

                if max_reconnect_forward > 0 and trend["forward_error"] > max_reconnect_forward:
                    continue

                if max_reconnect_backward > 0 and trend["backward_error"] > max_reconnect_backward:
                    continue

                if max_reconnect_link_angle > 0 and trend["link_angle"] > max_reconnect_link_angle:
                    continue

                if max_reconnect_velocity_angle > 0 and trend["velocity_angle"] > max_reconnect_velocity_angle:
                    continue

            velocity_diff = float(np.linalg.norm(source_direction - target_direction))

            if max_velocity_diff > 0 and velocity_diff > max_velocity_diff:
                continue

            cost = (
                distance
                + 0.08 * gap
                + 0.02 * max(target["length"] - 1, 0)
                + velocity_weight * velocity_diff
            )

            if max_cost > 0 and cost > max_cost:
                continue

            if best is None or cost < best["cost"]:
                best = {
                    "source_segment_id": source["segment_id"],
                    "target_segment_id": target["segment_id"],
                    "source_id": source["track_id"],
                    "target_id": target["track_id"],
                    "source_segment_index": source["segment_index"],
                    "target_segment_index": target["segment_index"],
                    "source_end": source["end"],
                    "target_start": target["start"],
                    "gap": gap,
                    "distance": round(distance, 4),
                    "lateral_error": round(trend["lateral_error"], 4),
                    "forward_error": round(trend["forward_error"], 4),
                    "backward_error": round(trend["backward_error"], 4),
                    "link_angle": round(trend["link_angle"], 4),
                    "velocity_angle": round(trend["velocity_angle"], 4),
                    "velocity_diff": round(velocity_diff, 4),
                    "target_start_count": target["start_count"],
                    "target_length": target["length"],
                    "cost": round(cost, 4),
                }

        if best is not None:
            pairs.append(best)

    pairs.sort(key=lambda row: (row["target_start"], row["cost"]))
    return pairs


def build_label_frame_points(tracklets: dict[int, dict]) -> dict[int, dict[int, list[np.ndarray]]]:
    label_frame_points = defaultdict(lambda: defaultdict(list))

    for segment in tracklets.values():
        label = int(segment["track_id"])

        for obs in segment["observations"]:
            label_frame_points[label][int(obs["frame"])].append(obs["points_xy"])

    return label_frame_points


def label_size_violation(
    label_frame_points: dict[int, dict[int, list[np.ndarray]]],
    label: int,
    target_segment: dict,
    max_label_diameter: float,
    max_label_pca_length: float,
    max_label_pca_width: float,
    min_size_check_points: int,
) -> dict | None:
    if max_label_diameter <= 0 and max_label_pca_length <= 0 and max_label_pca_width <= 0:
        return None

    for obs in target_segment["observations"]:
        frame_id = int(obs["frame"])
        chunks = list(label_frame_points[label].get(frame_id, [])) + [obs["points_xy"]]
        points_xy = np.vstack(chunks)

        if len(points_xy) < min_size_check_points:
            continue

        footprint = point_footprint(points_xy)

        if max_label_diameter > 0 and footprint["diameter"] > max_label_diameter:
            return {"frame": frame_id, "metric": "diameter", "value": footprint["diameter"]}

        if max_label_pca_length > 0 and footprint["pca_length"] > max_label_pca_length:
            return {"frame": frame_id, "metric": "pca_length", "value": footprint["pca_length"]}

        if max_label_pca_width > 0 and footprint["pca_width"] > max_label_pca_width:
            return {"frame": frame_id, "metric": "pca_width", "value": footprint["pca_width"]}

    return None


def label_frame_overlap_violation(
    label_frame_points: dict[int, dict[int, list[np.ndarray]]],
    label: int,
    target_segment: dict,
) -> dict | None:
    for obs in target_segment["observations"]:
        frame_id = int(obs["frame"])

        if label_frame_points[label].get(frame_id):
            return {"frame": frame_id, "metric": "label_frame_overlap", "value": 1.0}

    return None


def max_pairwise_distance(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0

    return float(np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2).max())


def label_chunk_spread_violation(
    label_frame_points: dict[int, dict[int, list[np.ndarray]]],
    label: int,
    target_segment: dict,
    max_label_chunk_distance: float,
) -> dict | None:
    if max_label_chunk_distance <= 0:
        return None

    for obs in target_segment["observations"]:
        frame_id = int(obs["frame"])
        chunks = list(label_frame_points[label].get(frame_id, [])) + [obs["points_xy"]]

        if len(chunks) < 2:
            continue

        centroids = np.asarray([chunk.mean(axis=0) for chunk in chunks], dtype=float)
        spread = max_pairwise_distance(centroids)

        if spread > max_label_chunk_distance:
            return {"frame": frame_id, "metric": "chunk_centroid_distance", "value": spread}

    return None


def build_label_centroid_rows(
    label_frame_points: dict[int, dict[int, list[np.ndarray]]],
    label: int,
    target_segment: dict,
) -> list[tuple[int, np.ndarray]]:
    frame_points = {
        frame_id: list(chunks)
        for frame_id, chunks in label_frame_points[label].items()
    }

    for obs in target_segment["observations"]:
        frame_points.setdefault(int(obs["frame"]), []).append(obs["points_xy"])

    rows = []

    for frame_id, chunks in frame_points.items():
        points_xy = np.vstack(chunks)
        rows.append((int(frame_id), points_xy.mean(axis=0)))

    return sorted(rows, key=lambda row: row[0])


def label_trajectory_violation(
    label_frame_points: dict[int, dict[int, list[np.ndarray]]],
    label: int,
    target_segment: dict,
    max_label_centroid_step: float,
    max_label_motion_residual: float,
    max_label_turn_angle: float,
    trajectory_check_gap: int,
    min_turn_speed: float,
) -> dict | None:
    if max_label_centroid_step <= 0 and max_label_motion_residual <= 0 and max_label_turn_angle <= 0:
        return None

    rows = build_label_centroid_rows(label_frame_points, label, target_segment)

    for (frame_a, xy_a), (frame_b, xy_b) in zip(rows, rows[1:]):
        gap = int(frame_b) - int(frame_a)

        if gap <= 0 or gap > trajectory_check_gap:
            continue

        step = float(np.linalg.norm(xy_b - xy_a)) / max(gap, 1)

        if max_label_centroid_step > 0 and step > max_label_centroid_step:
            return {"frame": frame_b, "metric": "centroid_step", "value": step}

    for (frame_a, xy_a), (frame_b, xy_b), (frame_c, xy_c) in zip(rows, rows[1:], rows[2:]):
        gap_ab = int(frame_b) - int(frame_a)
        gap_bc = int(frame_c) - int(frame_b)

        if gap_ab <= 0 or gap_bc <= 0 or gap_ab > trajectory_check_gap or gap_bc > trajectory_check_gap:
            continue

        velocity_ab = (xy_b - xy_a) / gap_ab
        velocity_bc = (xy_c - xy_b) / gap_bc

        if max_label_motion_residual > 0:
            predicted_c = xy_b + velocity_ab * gap_bc
            residual = float(np.linalg.norm(xy_c - predicted_c))

            if residual > max_label_motion_residual:
                return {"frame": frame_c, "metric": "motion_residual", "value": residual}

        if max_label_turn_angle > 0:
            turn_angle = vector_angle(velocity_ab, velocity_bc, min_turn_speed)

            if turn_angle > max_label_turn_angle:
                return {"frame": frame_c, "metric": "turn_angle", "value": turn_angle}

    return None


def resolve_mapping(
    pairs: list[dict],
    tracklets: dict[int, dict],
    max_label_diameter: float = 0.0,
    max_label_pca_length: float = 0.0,
    max_label_pca_width: float = 0.0,
    min_size_check_points: int = 4,
    max_label_chunk_distance: float = 0.0,
    max_label_centroid_step: float = 0.0,
    max_label_motion_residual: float = 0.0,
    max_label_turn_angle: float = 0.0,
    trajectory_check_gap: int = 5,
    min_turn_speed: float = 0.1,
    allow_label_overlap: bool = False,
) -> dict[int, int]:
    segment_to_label = {}
    used_targets = set()
    label_frame_points = build_label_frame_points(tracklets)

    for pair in pairs:
        target_segment_id = int(pair["target_segment_id"])

        if target_segment_id in used_targets:
            continue

        source_segment_id = int(pair["source_segment_id"])
        source_label = segment_to_label.get(source_segment_id, int(tracklets[source_segment_id]["track_id"]))

        if int(tracklets[target_segment_id]["track_id"]) == source_label:
            continue

        if not allow_label_overlap:
            violation = label_frame_overlap_violation(
                label_frame_points,
                source_label,
                tracklets[target_segment_id],
            )

            if violation is not None:
                continue

        violation = label_size_violation(
            label_frame_points,
            source_label,
            tracklets[target_segment_id],
            max_label_diameter=max_label_diameter,
            max_label_pca_length=max_label_pca_length,
            max_label_pca_width=max_label_pca_width,
            min_size_check_points=min_size_check_points,
        )

        if violation is not None:
            continue

        violation = label_chunk_spread_violation(
            label_frame_points,
            source_label,
            tracklets[target_segment_id],
            max_label_chunk_distance=max_label_chunk_distance,
        )

        if violation is not None:
            continue

        violation = label_trajectory_violation(
            label_frame_points,
            source_label,
            tracklets[target_segment_id],
            max_label_centroid_step=max_label_centroid_step,
            max_label_motion_residual=max_label_motion_residual,
            max_label_turn_angle=max_label_turn_angle,
            trajectory_check_gap=trajectory_check_gap,
            min_turn_speed=min_turn_speed,
        )

        if violation is not None:
            continue

        segment_to_label[target_segment_id] = source_label
        used_targets.add(target_segment_id)

        for obs in tracklets[target_segment_id]["observations"]:
            label_frame_points[source_label][int(obs["frame"])].append(obs["points_xy"])

    return segment_to_label


def apply_mapping(
    masks: dict[int, np.ndarray],
    mapping: dict[int, int],
    tracklets: dict[int, dict],
) -> dict[int, np.ndarray]:
    out = {frame_id: mask.copy() for frame_id, mask in masks.items()}

    for segment_id, new_id in mapping.items():
        segment = tracklets[segment_id]
        old_id = int(segment["track_id"])

        for obs in segment["observations"]:
            frame_id = int(obs["frame"])

            if frame_id in out:
                out[frame_id][masks[frame_id] == old_id] = int(new_id)

    return out


def write_pairs(path: Path, pairs: list[dict]) -> None:
    if not pairs:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline low-point tracklet smoothing for result.csv.")
    parser.add_argument("--input", required=True, help="Input result.csv.")
    parser.add_argument("--output", required=True, help="Smoothed output result.csv.")
    parser.add_argument("--merge-report", default=None)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--max-distance", type=float, default=2.5)
    parser.add_argument("--low-point-threshold", type=int, default=3)
    parser.add_argument("--min-source-length", type=int, default=2)
    parser.add_argument("--split-gap", type=int, default=0)
    parser.add_argument("--split-jump-distance", type=float, default=0.0)
    parser.add_argument("--trend-fit-tail", type=int, default=8, help="Frames used for reconnect regression. 0 uses the whole segment.")
    parser.add_argument("--max-cost", type=float, default=0.0, help="0 disables this filter.")
    parser.add_argument("--max-target-length", type=int, default=0, help="0 disables this filter.")
    parser.add_argument("--max-velocity-diff", type=float, default=0.0, help="0 disables this filter.")
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--reconnect-motion-model", choices=["endpoint", "regression"], default="endpoint")
    parser.add_argument("--max-reconnect-lateral", type=float, default=0.0, help="Trend-corridor lateral error gate. 0 disables.")
    parser.add_argument("--max-reconnect-forward", type=float, default=0.0, help="Trend-corridor forward error gate. 0 disables.")
    parser.add_argument("--max-reconnect-backward", type=float, default=0.0, help="Trend-corridor backward error gate. 0 disables.")
    parser.add_argument("--max-reconnect-link-angle", type=float, default=0.0, help="Degrees between source direction and reconnect link. 0 disables.")
    parser.add_argument("--max-reconnect-velocity-angle", type=float, default=0.0, help="Degrees between source outgoing and target incoming velocity. 0 disables.")
    parser.add_argument("--min-trend-speed", type=float, default=0.05)
    parser.add_argument("--max-label-diameter", type=float, default=0.0, help="Reject merges that make one label too large in one frame. 0 disables.")
    parser.add_argument("--max-label-pca-length", type=float, default=0.0, help="Reject merges by PCA major-axis footprint. 0 disables.")
    parser.add_argument("--max-label-pca-width", type=float, default=0.0, help="Reject merges by PCA minor-axis footprint. 0 disables.")
    parser.add_argument("--min-size-check-points", type=int, default=4)
    parser.add_argument("--max-label-chunk-distance", type=float, default=0.0, help="Reject same-frame multi-chunk labels whose centroids are too far apart. 0 disables.")
    parser.add_argument("--max-label-centroid-step", type=float, default=0.0, help="Reject labels with excessive centroid motion per frame. 0 disables.")
    parser.add_argument("--max-label-motion-residual", type=float, default=0.0, help="Reject locally inconsistent centroid motion. 0 disables.")
    parser.add_argument("--max-label-turn-angle", type=float, default=0.0, help="Degrees. Usually keep disabled for intersections.")
    parser.add_argument("--trajectory-check-gap", type=int, default=5)
    parser.add_argument("--min-turn-speed", type=float, default=0.1)
    parser.add_argument("--allow-label-overlap", action="store_true", help="Allow smoothing to assign one ID to multiple chunks in the same frame.")
    parser.add_argument("--cross-id-only", action="store_true")
    parser.add_argument("--ego-pos", default=None)
    parser.add_argument(
        "--ego-mode",
        choices=["none", "add_xy", "add_x_neg_y", "add_yx", "add_y_neg_x"],
        default="none",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    masks, coords = load_result_csv(args.input)
    ego_positions = read_ego_positions(args.ego_pos)
    tracklets = build_tracklets(
        masks,
        coords,
        split_gap=args.split_gap,
        split_jump_distance=args.split_jump_distance,
        ego_positions=ego_positions,
        ego_mode=args.ego_mode,
        trend_fit_tail=args.trend_fit_tail,
    )
    pairs = find_merge_pairs(
        tracklets,
        max_gap=args.max_gap,
        max_distance=args.max_distance,
        low_point_threshold=args.low_point_threshold,
        min_source_length=args.min_source_length,
        max_cost=args.max_cost,
        max_target_length=args.max_target_length,
        max_velocity_diff=args.max_velocity_diff,
        velocity_weight=args.velocity_weight,
        max_reconnect_lateral=args.max_reconnect_lateral,
        max_reconnect_forward=args.max_reconnect_forward,
        max_reconnect_backward=args.max_reconnect_backward,
        max_reconnect_link_angle=args.max_reconnect_link_angle,
        max_reconnect_velocity_angle=args.max_reconnect_velocity_angle,
        min_trend_speed=args.min_trend_speed,
        reconnect_motion_model=args.reconnect_motion_model,
        cross_id_only=args.cross_id_only,
    )
    mapping = resolve_mapping(
        pairs,
        tracklets,
        max_label_diameter=args.max_label_diameter,
        max_label_pca_length=args.max_label_pca_length,
        max_label_pca_width=args.max_label_pca_width,
        min_size_check_points=args.min_size_check_points,
        max_label_chunk_distance=args.max_label_chunk_distance,
        max_label_centroid_step=args.max_label_centroid_step,
        max_label_motion_residual=args.max_label_motion_residual,
        max_label_turn_angle=args.max_label_turn_angle,
        trajectory_check_gap=args.trajectory_check_gap,
        min_turn_speed=args.min_turn_speed,
        allow_label_overlap=args.allow_label_overlap,
    )
    smoothed = apply_mapping(masks, mapping, tracklets)
    write_result_csv(args.output, smoothed, coords)

    if args.merge_report is not None:
        write_pairs(Path(args.merge_report), pairs)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"tracklets: {len(tracklets)}")
    print(f"candidate_pairs: {len(pairs)}")
    print(f"merged_segments: {len(mapping)}")
    print("[DONE] smoothed result csv")
