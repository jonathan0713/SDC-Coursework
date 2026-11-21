"""
Hyperparameter dataclasses for the MOT tracking pipeline.

Each dataclass mirrors one pipeline stage so that every tunable knob
is visible, named, and serialisable in one place.

Pipeline mapping
────────────────
  FilterConfig    → [02] 運動預測  (KalmanFilter.__init__)
  GateConfig      → [03a] Mahalanobis 閘門 (gate())
  CostConfig      → [03b] 代價矩陣 (build_cost_matrix())
  LifecycleConfig → [05]  Track 管理 (Tracker lifecycle)
  TrackerConfig   → top-level container
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class FilterConfig:
    """Kalman Filter parameters ([02] 運動預測)."""
    type: str = "kalman"          # "kalman" | "identity"  (identity = disable KF)
    q_var: float = 1.0            # longitudinal (x/vx) process noise variance
    r_var: float = 1.0            # measurement noise variance (雷達量測雜訊)
    p_var: float = 10.0           # initial position covariance diagonal
    dt: float = 1.0               # time step between frames (frame-rate dependent)
    r_var_vel: float = 1.0        # measurement noise for augmented radial-velocity observation
    q_var_lat: Optional[float] = None  # lateral (y/vy) process noise; None = same as q_var


@dataclass
class GateConfig:
    """Distance gating parameters ([03a] 閘門)."""
    max_distance: float = 3.0    # Mahalanobis distance soft threshold
    max_euclidean: float = 10.0  # Euclidean distance hard cap (metres)
    rr_vel_gate: float = 0.0     # |Δrr| threshold for soft rr correction (m/s);
                                 # when exceeded, correct() uses [x,y] only (no velocity update).
                                 # 0.0 = disabled (always use full [x,y,rr] observation).
    rr_spawn_min_pts: int = 0    # min cluster point count to trust rr at spawn;
                                 # if cluster has fewer points AND |rr|>rr_vel_gate, use zero vx/vy init.
                                 # 0 = disabled (always use rr for spawn init).


@dataclass
class CostConfig:
    """Cost matrix parameters ([03b] 代價矩陣)."""
    range_rate_sigma: float = 2.0  # normalisation for range_rate consistency term (m/s)
    y_cost_weight: float = 0.0     # lateral (y) penalty per metre; 0 = disabled


@dataclass
class LifecycleConfig:
    """Track lifecycle parameters ([05] Track 管理)."""
    max_age: int = 5    # frames a track survives without matching before deletion


@dataclass
class EgoLowpassConfig:
    """Butterworth low-pass filter applied to ego velocities before tracking."""
    cutoff: float = 0.1  # normalised cutoff freq in (0, 1) relative to Nyquist
    order: int = 4       # filter order


@dataclass
class CameraGateConfig:
    """Camera-gated age freezing：雷達丟失時以相機偵測結果凍結 track age。"""
    enabled: bool = False                                      # 總開關
    extrinsics: Optional[str] = None                          # extrinsics .npz 路徑；None = 自動尋找
    # YOLO（一級確認：快速、class-aware）
    yolo_model: str = "yolo11n.pt"                            # YOLO 模型名稱或路徑
    yolo_conf: float = 0.35                                   # 信心值閾值
    yolo_classes: list = field(default_factory=lambda: [2, 3, 5, 7])  # COCO: car=2, motorcycle=3, bus=5, truck=7
    proj_margin_px: int = 20                                  # 投影點 (u,v) 命中 box 的容忍像素
    # SAM2（二級 fallback：處理遠距離小物件）
    sam2_ckpt: Optional[str] = None                           # SAM2 checkpoint；None = 自動尋找
    sam2_cfg: str = "sam2_hiera_t"                            # SAM2 config 名稱
    sam2_score_threshold: float = 0.70                        # SAM2 分割信心值閾值
    sam2_mask_min_px: int = 50                                # mask 面積下限（過濾雜訊）
    sam2_mask_max_px: int = 100_000                           # mask 面積上限（過濾背景）
    proj_z_height: float = 0.0                               # 投影時固定的 z 高度（m）；取代雷達雜訊 z
    cam_rescue_gate_px: int = 50                             # 04-② 補救閘門：cluster 投影點到 mask 邊界的最大像素距離
    cam_mask_drift_px: int = 80                              # drift guard：KF 投影點到 mask 邊界的最大容忍像素；超出則移除 mask


@dataclass
class TrackerConfig:
    """
    Top-level container grouping all pipeline stage configs.

    Usage
    -----
    cfg = TrackerConfig()                    # all defaults
    tracker = Tracker.from_config(cfg)       # construct tracker

    cfg.save(path)                           # serialise to JSON
    cfg = TrackerConfig.load(path)           # restore from JSON
    """
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    ego_lowpass: EgoLowpassConfig = field(default_factory=EgoLowpassConfig)
    camera_gate: CameraGateConfig = field(default_factory=CameraGateConfig)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_flat_dict(self) -> dict[str, float | int]:
        """Flatten to a single-level dict for TensorBoard HParams / JSON."""
        return {
            # lifecycle
            "lifecycle.max_age": self.lifecycle.max_age,
            # filter
            "filter.q_var": self.filter.q_var,
            "filter.q_var_lat": self.filter.q_var_lat,
            "filter.r_var": self.filter.r_var,
            "filter.p_var": self.filter.p_var,
            "filter.dt": self.filter.dt,
            "filter.r_var_vel": self.filter.r_var_vel,
            # gate
            "gate.max_distance": self.gate.max_distance,
            "gate.max_euclidean": self.gate.max_euclidean,
            "gate.rr_vel_gate": self.gate.rr_vel_gate,
            "gate.rr_spawn_min_pts": self.gate.rr_spawn_min_pts,
            # cost
            "cost.range_rate_sigma": self.cost.range_rate_sigma,
            "cost.y_cost_weight": self.cost.y_cost_weight,
        }

    @classmethod
    def from_flat_dict(cls, d: dict) -> "TrackerConfig":
        return cls(
            lifecycle=LifecycleConfig(
                max_age=int(d["lifecycle.max_age"]),
            ),
            filter=FilterConfig(
                q_var=float(d["filter.q_var"]),
                r_var=float(d["filter.r_var"]),
                p_var=float(d["filter.p_var"]),
                dt=float(d.get("filter.dt", 1.0)),
                r_var_vel=float(d.get("filter.r_var_vel", 1.0)),
            ),
            gate=GateConfig(
                max_distance=float(d["gate.max_distance"]),
                max_euclidean=float(d["gate.max_euclidean"]),
            ),
            cost=CostConfig(
                range_rate_sigma=float(d["cost.range_rate_sigma"]),
            ),
        )

    def to_nested_dict(self) -> dict:
        """Nested dict matching the YAML structure."""
        return {
            "lifecycle":    asdict(self.lifecycle),
            "filter":       asdict(self.filter),
            "gate":         asdict(self.gate),
            "cost":         asdict(self.cost),
            "ego_lowpass":  asdict(self.ego_lowpass),
            "camera_gate":  asdict(self.camera_gate),
        }

    @classmethod
    def from_nested_dict(cls, d: dict) -> "TrackerConfig":
        lc = d["lifecycle"]
        fl = d["filter"]
        ga = d["gate"]
        co = d["cost"]
        el = d.get("ego_lowpass", {})
        cg = d.get("camera_gate", {})
        return cls(
            lifecycle=LifecycleConfig(
                max_age=int(lc["max_age"]),
            ),
            filter=FilterConfig(
                type=str(fl.get("type", "kalman")),
                q_var=float(fl["q_var"]),
                r_var=float(fl["r_var"]),
                p_var=float(fl["p_var"]),
                dt=float(fl.get("dt", 1.0)),
                r_var_vel=float(fl.get("r_var_vel", 1.0)),
                q_var_lat=float(fl["q_var_lat"]) if fl.get("q_var_lat") is not None else None,
            ),
            gate=GateConfig(
                max_distance=float(ga["max_distance"]),
                max_euclidean=float(ga["max_euclidean"]),
                rr_vel_gate=float(ga.get("rr_vel_gate", 0.0)),
                rr_spawn_min_pts=int(ga.get("rr_spawn_min_pts", 0)),
            ),
            cost=CostConfig(
                range_rate_sigma=float(co["range_rate_sigma"]),
                y_cost_weight=float(co.get("y_cost_weight", 0.0)),
            ),
            ego_lowpass=EgoLowpassConfig(
                cutoff=float(el.get("cutoff", 0.1)),
                order=int(el.get("order", 4)),
            ),
            camera_gate=CameraGateConfig(
                enabled=bool(cg.get("enabled", False)),
                extrinsics=cg.get("extrinsics", None),
                yolo_model=str(cg.get("yolo_model", "yolo11n.pt")),
                yolo_conf=float(cg.get("yolo_conf", 0.35)),
                yolo_classes=list(cg.get("yolo_classes", [2, 3, 5, 7])),
                proj_margin_px=int(cg.get("proj_margin_px", 20)),
                sam2_ckpt=cg.get("sam2_ckpt", None),
                sam2_cfg=str(cg.get("sam2_cfg", "sam2_hiera_t")),
                sam2_score_threshold=float(cg.get("sam2_score_threshold", 0.70)),
                sam2_mask_min_px=int(cg.get("sam2_mask_min_px", 50)),
                sam2_mask_max_px=int(cg.get("sam2_mask_max_px", 100_000)),
                proj_z_height=float(cg.get("proj_z_height", 0.0)),
                cam_rescue_gate_px=int(cg.get("cam_rescue_gate_px", 50)),
                cam_mask_drift_px=int(cg.get("cam_mask_drift_px", 80)),
            ),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path).with_suffix(".yaml")
        path.write_text(yaml.dump(self.to_nested_dict(), default_flow_style=False, sort_keys=False))

    @classmethod
    def load(cls, path: Path | str) -> "TrackerConfig":
        """Load from a YAML file (nested) or legacy flat-dict JSON/YAML."""
        raw = yaml.safe_load(Path(path).read_text())
        if "lifecycle" in raw:
            return cls.from_nested_dict(raw)
        return cls.from_flat_dict(raw)  # backward-compat with old flat JSON

    def __str__(self) -> str:
        lines = ["TrackerConfig:"]
        for k, v in self.to_flat_dict().items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
