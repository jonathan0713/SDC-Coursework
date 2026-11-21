import argparse
import bisect
import csv
import zipfile
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_project_root() -> Path:
    return default_repo_root().parent


def read_target_frame_ids(frame_list: Path | None, output_dir: Path) -> list[str]:
    if frame_list and frame_list.exists():
        frame_ids = []
        with frame_list.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    frame_ids.append(Path(line).stem)
        return frame_ids

    frame_ids = [p.stem for p in output_dir.glob("*.png") if p.stem.isdigit()]
    if not frame_ids:
        raise FileNotFoundError(
            f"No target frames found. Provide --frame-list or existing PNGs in {output_dir}"
        )
    return sorted(frame_ids, key=int)


def index_source_images(source_dir: Path) -> tuple[list[int], dict[int, Path]]:
    images: dict[int, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for path in source_dir.glob(f"*{suffix}"):
            if path.stem.isdigit():
                images[int(path.stem)] = path

    if not images:
        raise FileNotFoundError(f"No timestamp-named images found in {source_dir}")

    timestamps = sorted(images)
    return timestamps, images


def interpolate_image(
    target_timestamp: int,
    source_timestamps: list[int],
    source_images: dict[int, Path],
) -> tuple[Image.Image, str]:
    insert_at = bisect.bisect_left(source_timestamps, target_timestamp)

    if insert_at < len(source_timestamps) and source_timestamps[insert_at] == target_timestamp:
        exact_path = source_images[target_timestamp]
        return Image.open(exact_path).convert("RGB"), f"exact {exact_path.name}"

    if insert_at == 0:
        nearest = source_timestamps[0]
        return Image.open(source_images[nearest]).convert("RGB"), f"nearest {source_images[nearest].name}"

    if insert_at == len(source_timestamps):
        nearest = source_timestamps[-1]
        return Image.open(source_images[nearest]).convert("RGB"), f"nearest {source_images[nearest].name}"

    left_ts = source_timestamps[insert_at - 1]
    right_ts = source_timestamps[insert_at]
    alpha = (target_timestamp - left_ts) / (right_ts - left_ts)

    with Image.open(source_images[left_ts]) as left_raw:
        left = left_raw.convert("RGB")
    with Image.open(source_images[right_ts]) as right_raw:
        right = right_raw.convert("RGB")

    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.BICUBIC)

    image = Image.blend(left, right, alpha)
    note = f"{source_images[left_ts].name} -> {source_images[right_ts].name}, alpha={alpha:.6f}"
    return image, note


def write_submission_csv(path: Path, frame_ids: list[str]) -> None:
    if path.exists():
        return

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        for frame_id in frame_ids:
            writer.writerow([frame_id, "0.0"])


def write_zip(output_dir: Path, zip_path: Path, frame_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for frame_id in frame_ids:
            zf.write(output_dir / f"{frame_id}.png", arcname=f"{frame_id}.png")
        submission_csv = output_dir / "submission.csv"
        if submission_csv.exists():
            zf.write(submission_csv, arcname="submission.csv")


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(
        description="Generate test submission images by timestamp interpolation from raw dataset images."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "data" / "Track3" / "data" / "image",
        help="Directory containing timestamp-named source images.",
    )
    parser.add_argument(
        "--frame-list",
        type=Path,
        default=repo_root / "track3" / "track3_test_frame_list.txt",
        help="Target frame list. If missing, existing PNG names in --output-dir are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "track3" / "test_submission",
        help="Directory where PNG submission images are written.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=repo_root / "track3" / "submission.zip",
        help="Submission zip path. Use --no-zip to skip zip creation.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create submission.zip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = read_target_frame_ids(args.frame_list, args.output_dir)
    source_timestamps, source_images = index_source_images(args.source_dir)

    print(f"Source images: {len(source_timestamps)} from {args.source_dir}")
    print(f"Target frames: {len(frame_ids)} to {args.output_dir}")

    for index, frame_id in enumerate(frame_ids, start=1):
        target_timestamp = int(frame_id)
        image, note = interpolate_image(target_timestamp, source_timestamps, source_images)
        output_path = args.output_dir / f"{frame_id}.png"
        image.save(output_path)
        print(f"[{index:02d}/{len(frame_ids):02d}] {output_path.name}: {note}")

    write_submission_csv(args.output_dir / "submission.csv", frame_ids)

    if not args.no_zip:
        write_zip(args.output_dir, args.zip_path, frame_ids)
        print(f"Wrote {args.zip_path}")


if __name__ == "__main__":
    main()
