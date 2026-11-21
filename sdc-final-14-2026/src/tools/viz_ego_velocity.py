"""
Visualise ego velocity derived from ego_global_pos.txt,
comparing raw vs Butterworth low-pass filtered velocity.

Filter parameters are read from src/utils/default_config.yaml (ego_lowpass section).
Override them with --cutoff / --order on the command line.

Usage
─────
  python src/tools/viz_ego_velocity.py --seq seq_3
  python src/tools/viz_ego_velocity.py --seq seq_3 --out data/outputs/ego_vel_seq3.png
  python src/tools/viz_ego_velocity.py --seq seq_3 --cutoff 0.05 --order 2
  python src/tools/viz_ego_velocity.py \
      --ego-pos data/public_dev/seq_3/ego_global_pos.txt

Panels
──────
  1. Trajectory (world XY) — raw arrows vs filtered arrows
  2. vx / vy over time — raw vs filtered
  3. Speed |v| over time — raw vs filtered
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdc_tracking_utils import filter_ego_velocities, load_ego_velocities, read_ego_pos


_PROJECT_ROOT   = Path(__file__).parent.parent.parent
_DEFAULT_DATA_ROOT   = _PROJECT_ROOT / "data" / "public_dev"
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "utils" / "default_config.yaml"


def load_filter_defaults() -> tuple[float, int]:
    raw = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text())
    el = raw.get("ego_lowpass", {})
    return float(el.get("cutoff", 0.1)), int(el.get("order", 4))


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(ego_pos_path: Path):
    positions  = read_ego_pos(ego_pos_path)         # (N, 3): tx, ty, tz
    velocities = load_ego_velocities(ego_pos_path)  # list of (2,) arrays

    if len(positions) == 0 or len(velocities) == 0:
        raise ValueError(f"No data found in {ego_pos_path}")

    n   = min(len(positions), len(velocities))
    pos = positions[:n, :2]         # (N, 2)
    vel = np.array(velocities[:n])  # (N, 2)
    return np.arange(n), pos, vel


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(frames, pos, vel, vel_f, seq_name: str, cutoff: float, order: int,
         out_path: Path | None):
    speed   = np.linalg.norm(vel,   axis=1)
    speed_f = np.linalg.norm(vel_f, axis=1)

    C_RAW  = "steelblue"
    C_FILT = "tomato"
    flabel = f"Butterworth (cutoff={cutoff}, order={order})"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Ego velocity — {seq_name}  |  {flabel}", fontsize=13)

    # ── panel 1: trajectory + arrows ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(pos[:, 0], pos[:, 1], color="grey", lw=1.0, zorder=1, label="trajectory")
    ax.plot(pos[0,  0], pos[0,  1], "go", ms=7, zorder=3, label="start")
    ax.plot(pos[-1, 0], pos[-1, 1], "rs", ms=7, zorder=3, label="end")

    step  = max(1, len(frames) // 30)
    q_kw  = dict(scale=None, scale_units="xy", angles="xy", width=0.003, alpha=0.7)
    ax.quiver(pos[::step, 0], pos[::step, 1],
              vel[::step,   0], vel[::step,   1], color=C_RAW,  label="raw",      zorder=2, **q_kw)
    ax.quiver(pos[::step, 0], pos[::step, 1],
              vel_f[::step, 0], vel_f[::step, 1], color=C_FILT, label="filtered", zorder=4, **q_kw)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory & velocity arrows")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── panel 2: vx / vy over time ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(frames, vel[:,   0], color=C_RAW,            lw=0.8, alpha=0.5, label="vx raw")
    ax.plot(frames, vel[:,   1], color="cornflowerblue", lw=0.8, alpha=0.5, label="vy raw")
    ax.plot(frames, vel_f[:, 0], color=C_FILT,           lw=1.5,            label="vx filtered")
    ax.plot(frames, vel_f[:, 1], color="darkred",        lw=1.5,            label="vy filtered")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("vx / vy — raw vs filtered")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── panel 3: speed over time ──────────────────────────────────────────────
    ax = axes[2]
    ax.plot(frames, speed,   color=C_RAW,  lw=0.8, alpha=0.5, label="raw")
    ax.plot(frames, speed_f, color=C_FILT, lw=1.5,            label="filtered")
    ax.fill_between(frames, speed,   alpha=0.10, color=C_RAW)
    ax.fill_between(frames, speed_f, alpha=0.15, color=C_FILT)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed |v| — raw vs filtered")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[saved] {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    default_cutoff, default_order = load_filter_defaults()

    p = argparse.ArgumentParser(
        description="Visualise ego velocity with Butterworth low-pass filter comparison"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq",     help="Sequence name, e.g. seq_3")
    src.add_argument("--ego-pos", help="Direct path to ego_global_pos.txt")

    p.add_argument("--data-root", default=None,
                   help=f"Dataset root (used with --seq). Defaults to {_DEFAULT_DATA_ROOT}")
    p.add_argument("--out", default=None,
                   help="Output PNG path (omit to display interactively)")
    p.add_argument("--cutoff", type=float, default=default_cutoff,
                   help=f"Normalised cutoff freq in (0,1). Default from config: {default_cutoff}")
    p.add_argument("--order",  type=int,   default=default_order,
                   help=f"Butterworth filter order. Default from config: {default_order}")

    return p.parse_args()


def main():
    args = parse_args()

    if args.ego_pos:
        ego_pos_path = Path(args.ego_pos)
        seq_name = ego_pos_path.parent.name
    else:
        data_root    = Path(args.data_root) if args.data_root else _DEFAULT_DATA_ROOT
        ego_pos_path = data_root / args.seq / "ego_global_pos.txt"
        seq_name     = args.seq

    if not ego_pos_path.exists():
        print(f"[error] File not found: {ego_pos_path}")
        sys.exit(1)

    frames, pos, vel = load_data(ego_pos_path)
    vel_f = np.array(filter_ego_velocities(
        [vel[i] for i in range(len(vel))], cutoff=args.cutoff, order=args.order
    ))

    speed   = np.linalg.norm(vel,   axis=1)
    speed_f = np.linalg.norm(vel_f, axis=1)
    print(f"[info] {seq_name}: {len(frames)} frames  |  "
          f"Butterworth cutoff={args.cutoff} order={args.order}")
    print(f"[info] raw   speed: min={speed.min():.2f}  max={speed.max():.2f}  mean={speed.mean():.2f} m/s")
    print(f"[info] filt  speed: min={speed_f.min():.2f}  max={speed_f.max():.2f}  mean={speed_f.mean():.2f} m/s")

    out_path = Path(args.out) if args.out else None
    plot(frames, pos, vel, vel_f, seq_name, args.cutoff, args.order, out_path)


if __name__ == "__main__":
    main()
