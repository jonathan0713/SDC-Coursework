from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an animated GIF to an MP4 video.")
    parser.add_argument("input", help="Input .gif path.")
    parser.add_argument("-o", "--output", default=None, help="Output .mp4 path. Defaults to input path with .mp4 suffix.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality. Lower is better/larger, typical range 18-28.")
    parser.add_argument("--preset", default="medium", help="ffmpeg x264 preset, e.g. ultrafast, medium, slow.")
    parser.add_argument("--background", default="white", help="Background color for transparent GIF frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def even_size(width: int, height: int) -> tuple[int, int]:
    return width + (width % 2), height + (height % 2)


def frame_duration_seconds(frame: Image.Image, fps: float) -> float:
    duration_ms = int(frame.info.get("duration", 0))
    if duration_ms <= 0:
        return 1.0 / fps
    return duration_ms / 1000.0


def iter_rgb_frames(gif_path: Path, fps: float, background: str):
    with Image.open(gif_path) as gif:
        source_width, source_height = gif.size
        output_width, output_height = even_size(source_width, source_height)
        previous = Image.new("RGBA", gif.size, background)

        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")
            disposal = int(getattr(frame, "disposal_method", 0) or 0)

            canvas = previous.copy()
            canvas.alpha_composite(rgba)

            rgb = Image.new("RGB", (output_width, output_height), background)
            rgb.paste(canvas.convert("RGB"), (0, 0))

            repeat = max(1, int(round(frame_duration_seconds(frame, fps) * fps)))
            for _ in range(repeat):
                yield rgb.tobytes()

            if disposal == 2:
                previous = Image.new("RGBA", gif.size, background)
            else:
                previous = canvas


def run_ffmpeg(input_path: Path, output_path: Path, fps: float, crf: int, preset: str, background: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. Install it first, for example: sudo apt install ffmpeg"
        )

    with Image.open(input_path) as gif:
        width, height = even_size(*gif.size)

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert process.stdin is not None

    try:
        for frame_bytes in iter_rgb_frames(input_path, fps, background):
            process.stdin.write(frame_bytes)
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg stopped before all frames were written.") from exc
    finally:
        process.stdin.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".mp4")

    if not input_path.exists():
        print(f"[ERROR] input does not exist: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".gif":
        print(f"[ERROR] input must be a .gif file: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not args.overwrite:
        print(f"[ERROR] output already exists, pass --overwrite: {output_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_ffmpeg(input_path, output_path, args.fps, args.crf, args.preset, args.background)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[DONE] wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
