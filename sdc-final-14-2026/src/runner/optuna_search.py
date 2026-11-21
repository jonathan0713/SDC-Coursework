"""
Optuna hyperparameter search for the MOT tracking pipeline.
Results are logged to TensorBoard (HParams + scalar tabs).

Usage
─────
  # search on a single public-dev sequence (GT inferred from utils/paths.py):
  python src/runner/optuna_search.py --seq seq_1

  # search on multiple sequences:
  python src/runner/optuna_search.py --seq seq_1,seq_3

  # override GT paths explicitly:
  python src/runner/optuna_search.py --seq seq_1 --gt data/seq_1/gt_answer_seq1.csv

  # resume a previous study:
      --study-name my_study  (same name → loads existing SQLite DB)

  # view results:
      tensorboard --logdir data/optuna/tb_logs

Directory layout (isolated from the main pipeline)
────────────────────────────────────────────────────
  data/optuna/
  ├── study.db            ← Optuna SQLite storage (persistent)
  ├── tb_logs/            ← TensorBoard event files
  │   ├── scalars/        ← per-trial MOTA / error counts / param values
  │   └── hparams/        ← per-trial HParams tab entries
  ├── best_config.yaml    ← best TrackerConfig after study
  └── trials/             ← temporary result CSVs (cleaned after eval)
      └── trial_<n>/result.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ── make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from tensorboardX import SummaryWriter

from evaluate_tracking import evaluate
from runner.configs import TrackerConfig
from sdc_tracker import Tracker
from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    save_tracking_mask_to_csv,
    separate_static_and_moving,
)
from utils.detection import build_detections, calculate_cluster_centroids
from utils.paths import DATA_ROOT, GT_PATHS, OPTUNA_DIR


# ── search space ──────────────────────────────────────────────────────────────

_SEARCH_SPACE = {
    # lifecycle
    "lifecycle.max_age":  ("int",   2,    12),
    # filter — KF noise parameters
    # r_var_vel is always active: augmented [x,y,ṙ] observation is used for all sequences
    "filter.q_var":       ("log",   0.05, 20.0),
    "filter.r_var":       ("log",   0.05, 20.0),
    "filter.r_var_vel":   ("log",   0.05, 20.0),
    "filter.p_var":       ("log",   1.0, 200.0),
    # gate
    "gate.max_distance":  ("float", 1.0,   8.0),
    "gate.max_euclidean": ("float", 3.0,  30.0),
    # cost
    "cost.range_rate_sigma": ("float", 0.3, 6.0),
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    kind, lo, hi = spec
    if kind == "int":
        return trial.suggest_int(name, int(lo), int(hi))
    if kind == "log":
        return trial.suggest_float(name, lo, hi, log=True)
    return trial.suggest_float(name, lo, hi)


def sample_config(trial: optuna.Trial) -> TrackerConfig:
    flat = {name: _suggest(trial, name, spec) for name, spec in _SEARCH_SPACE.items()}
    return TrackerConfig.from_flat_dict(flat)


# ── single-sequence tracking run ─────────────────────────────────────────────

def run_tracking(
    config: TrackerConfig,
    data_root: Path,
    seq: str,
    output_dir: Path,
) -> Path:
    """
    Run the full tracking pipeline for one sequence and save result.csv.
    Returns the path to the result CSV.
    No visualisation is produced (fully headless).
    """
    seq_dir = data_root / seq
    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"

    result_csv = output_dir / "result.csv"
    if result_csv.exists():
        result_csv.unlink()

    tracker = Tracker.from_config(config)
    frame_names = list_frame_names(mask_dir)

    for frame_idx, mask_name in enumerate(frame_names):
        mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)

        moving_mask, _, moving_radar_pc, _, moving_indices = separate_static_and_moving(
            mask, radar_pc
        )

        cluster_centroids, cluster_velocities, _ = calculate_cluster_centroids(
            moving_mask, moving_radar_pc
        )

        detections = build_detections(cluster_centroids, cluster_velocities)
        confirmed_tracks = tracker.update(detections)

        cluster_track_dict = {
            int(t[3]): int(t[4]) for t in confirmed_tracks
        }

        save_tracking_mask_to_csv(
            csv_path=result_csv,
            frame_id=frame_idx,
            mask=mask,
            cluster_track_dict=cluster_track_dict,
            moving_indices=moving_indices,
            radar_points=radar_pc,
        )

    return result_csv


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(args, scalar_writer: SummaryWriter):
    seqs = [s.strip() for s in args.seq.split(",")]
    if args.gt is None:
        missing = [s for s in seqs if s not in GT_PATHS]
        if missing:
            raise ValueError(
                f"No GT path in utils/paths.py for: {missing}. "
                "Pass --gt explicitly or add entries to GT_PATHS."
            )
        gt_paths = [str(GT_PATHS[s]) for s in seqs]
    else:
        gt_paths = [s.strip() for s in args.gt.split(",")]
        if len(gt_paths) == 1:
            gt_paths = gt_paths * len(seqs)
        if len(gt_paths) != len(seqs):
            raise ValueError("--gt must provide one path per sequence in --seq")

    data_root = Path(args.data_root)
    trials_dir = Path(args.output_dir) / "trials"

    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial)
        n = trial.number

        mota_list, fp_list, fn_list, idsw_list = [], [], [], []

        for seq, gt_csv in zip(seqs, gt_paths):
            trial_dir = trials_dir / f"trial_{n:04d}" / seq
            trial_dir.mkdir(parents=True, exist_ok=True)

            try:
                result_csv = run_tracking(config, data_root, seq, trial_dir)
                metrics = evaluate(gt_csv, result_csv)
            except Exception as exc:
                print(f"[trial {n}] {seq} failed: {exc}")
                return float("-inf")
            finally:
                # clean up CSVs to save disk space
                shutil.rmtree(trial_dir, ignore_errors=True)

            mota_list.append(metrics["MOTA_proxy"])
            fp_list.append(metrics["FP_points"])
            fn_list.append(metrics["FN_points"])
            idsw_list.append(metrics["IDSW_proxy"])

        mota = float(np.mean(mota_list))

        # ── scalar logging ────────────────────────────────────────────────
        scalar_writer.add_scalar("MOTA/trial", mota, n)
        try:
            best_so_far = max(mota, trial.study.best_value)
        except ValueError:
            best_so_far = mota  # no completed trial yet (trial 0)
        scalar_writer.add_scalar("MOTA/best_so_far", best_so_far, n)
        scalar_writer.add_scalars("errors", {
            "FP":   float(np.mean(fp_list)),
            "FN":   float(np.mean(fn_list)),
            "IDSW": float(np.mean(idsw_list)),
        }, n)
        flat = config.to_flat_dict()
        for k, v in flat.items():
            scalar_writer.add_scalar(f"params/{k}", float(v), n)

        # ── HParams tab entry (one sub-run per trial) ────────────────────
        hdir = Path(args.output_dir) / "tb_logs" / "hparams" / f"trial_{n:04d}"
        with SummaryWriter(log_dir=str(hdir)) as hw:
            hw.add_hparams(
                hparam_dict={k: float(v) for k, v in flat.items()},
                metric_dict={"MOTA": mota,
                             "FP": float(np.mean(fp_list)),
                             "FN": float(np.mean(fn_list)),
                             "IDSW": float(np.mean(idsw_list))},
            )

        return mota

    return objective


# ── post-study logging ────────────────────────────────────────────────────────

def log_study_summary(study: optuna.Study, writer: SummaryWriter, output_dir: Path):
    """Log importance scores and best-trial text after the study finishes."""
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, score in importance.items():
            writer.add_scalar(f"importance/{param}", score, 0)
        print("\n[Optuna] Parameter importances:")
        for param, score in importance.items():
            print(f"  {param}: {score:.4f}")
    except Exception:
        pass  # importance computation can fail with few trials

    try:
        best = study.best_trial
        writer.add_text(
            "best_config",
            f"Trial {best.number}  MOTA={best.value:.4f}\n\n"
            + str(TrackerConfig.from_flat_dict(best.params)),
            global_step=0,
        )
    except ValueError:
        print("[optuna] No completed trials — skipping best_config summary.")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the MOT tracker."
    )
    p.add_argument("--data-root",   default=str(DATA_ROOT),
                   help=f"Root of the dataset. Defaults to {DATA_ROOT}")
    p.add_argument("--seq",         required=True,
                   help="Comma-separated sequence names, e.g. seq_1,seq_3")
    p.add_argument("--gt",          default=None,
                   help="Comma-separated GT CSV paths (one per seq, or one shared). "
                        "Defaults to GT_PATHS in utils/paths.py when omitted.")
    p.add_argument("--output-dir",  default=str(OPTUNA_DIR),
                   help=f"Output root (isolated from main pipeline). Defaults to {OPTUNA_DIR}")
    p.add_argument("--n-trials",    type=int, default=3000)
    p.add_argument("--study-name",  default="mot_search",
                   help="Optuna study name; use the same name to resume")
    p.add_argument("--n-jobs",      type=int, default=1,
                   help="Parallel Optuna workers (>1 requires shared DB)")
    p.add_argument("--timeout",     type=float, default=None,
                   help="Stop search after this many seconds (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    tb_scalar_dir = output_dir / "tb_logs" / "scalars"
    tb_scalar_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trials").mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{output_dir / 'study.db'}"

    print(f"[optuna] study : {args.study_name}")
    print(f"[optuna] db    : {output_dir / 'study.db'}")
    print(f"[optuna] tb    : {tb_scalar_dir}")
    print(f"[optuna] seqs  : {args.seq}")
    print(f"[optuna] trials: {args.n_trials}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    scalar_writer = SummaryWriter(log_dir=str(tb_scalar_dir))

    try:
        study.optimize(
            make_objective(args, scalar_writer),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            show_progress_bar=True,
        )
    finally:
        log_study_summary(study, scalar_writer, output_dir)
        scalar_writer.close()

    # ── save best config ──────────────────────────────────────────────────────
    try:
        best_config = TrackerConfig.from_flat_dict(study.best_trial.params)
        best_path = output_dir / "best_config.yaml"
        best_config.save(best_path)
        print(f"\n[optuna] Best MOTA : {study.best_value:.4f}")
        print(f"[optuna] Best config saved → {best_path}")
        print(best_config)
    except ValueError:
        print("\n[optuna] No completed trials — best_config not saved.")

    print(f"\n[tensorboard] tensorboard --logdir {output_dir / 'tb_logs'}")


if __name__ == "__main__":
    main()
