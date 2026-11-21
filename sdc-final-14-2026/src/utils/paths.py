"""
Centralised path configuration for the MOT project.

All code that reads or writes data should import from here rather than
constructing paths ad-hoc.  When the data layout changes, only this file
needs to be updated.

Layout assumed
──────────────
  sdc-final/
  ├── data/
  │   ├── seq_1/          radar/ mask_cluster/ image/ gt_answer_seq1.csv
  │   ├── seq_2/          radar/ mask_cluster/ image/
  │   ├── seq_3/          radar/ mask_cluster/ image/ ego_global_pos.txt gt_answer_seq3.csv
  │   ├── seq_4/          radar/ mask_cluster/ image/ ego_global_pos.txt
  │   ├── outputs/        per-run result folders
  │   ├── optuna/         Optuna SQLite + TensorBoard logs + best_config.yaml
  │   ├── sample_submission_seq2.csv
  │   └── sample_submission_seq4.csv
  └── src/
      └── utils/
          └── default_config.yaml
"""
from __future__ import annotations

from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent   # …/sdc-final/
DATA_ROOT:    Path = PROJECT_ROOT / "data"
OUTPUT_ROOT:  Path = DATA_ROOT / "outputs"
VIZ_ROOT:     Path = DATA_ROOT / "viz"
OPTUNA_DIR:   Path = DATA_ROOT / "optuna"

# ── Tracker config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG: Path = Path(__file__).resolve().parent / "default_config.yaml"
BEST_CONFIG:    Path = OPTUNA_DIR / "best_config.yaml"

# ── Sequences ─────────────────────────────────────────────────────────────────
SEQS: tuple[str, ...] = ("seq_1", "seq_2", "seq_3", "seq_4")

# Sequences that provide ego motion data
EGO_SEQS: frozenset[str] = frozenset({"seq_3", "seq_4"})

# Sequences that have public-dev ground-truth CSVs
GT_SEQS: frozenset[str] = frozenset({"seq_1", "seq_3"})


def seq_dir(seq: str) -> Path:
    """Root directory for a sequence: data/<seq>/"""
    return DATA_ROOT / seq


def mask_dir(seq: str) -> Path:
    return seq_dir(seq) / "mask_cluster"


def radar_dir(seq: str) -> Path:
    return seq_dir(seq) / "radar"


def image_dir(seq: str) -> Path:
    return seq_dir(seq) / "image"


def ego_pos_path(seq: str) -> Path:
    """Path to ego_global_pos.txt for ego-motion sequences."""
    return seq_dir(seq) / "ego_global_pos.txt"


# ── Ground-truth CSVs (public dev only) ───────────────────────────────────────
GT_PATHS: dict[str, Path] = {
    "seq_1": seq_dir("seq_1") / "gt_answer_seq1.csv",
    "seq_3": seq_dir("seq_3") / "gt_answer_seq3.csv",
}

# ── Sample submissions (Kaggle test sequences) ────────────────────────────────
SAMPLE_SUBMISSIONS: dict[str, Path] = {
    "seq_2": DATA_ROOT / "sample_submission_seq2.csv",
    "seq_4": DATA_ROOT / "sample_submission_seq4.csv",
}
