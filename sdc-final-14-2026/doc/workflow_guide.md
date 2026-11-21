# 專案操作指南（以 seq_4 為例）

> 本文以 seq_4 為主線，說明從執行 tracker 到分析結果的完整流程。
> 所有指令從專案根目錄 `sdc-final/` 執行。

---

## 專案結構速覽

```
sdc-final/
├── data/
│   ├── seq_1/ ~ seq_4/          ← 雷達點雲、cluster mask、相機影像
│   │   ├── radar/               ← *.bin（每幀雷達點雲，float32 × 5）
│   │   ├── mask_cluster/        ← *.txt（每點的 cluster id，-1=靜止）
│   │   ├── image/               ← *.jpg（相機影像，可選）
│   │   └── ego_global_pos.txt   ← 自車 GPS 紀錄（seq_3/4 才有）
│   ├── outputs/                 ← main.py 每次執行的輸出資料夾
│   ├── viz/                     ← 視覺化 PNG 與 mp4
│   └── optuna/                  ← HPO 搜尋結果
└── src/
    ├── main.py                  ← 主程式入口
    ├── sdc_tracker.py           ← 追蹤器核心
    ├── runner/
    │   ├── configs.py           ← TrackerConfig dataclass
    │   └── optuna_search.py     ← Optuna HPO 搜尋
    ├── utils/
    │   ├── default_config.yaml  ← 預設參數（主要調整這個）
    │   ├── filter.py            ← KalmanFilter / IdentityFilter
    │   ├── association.py       ← gate、cost matrix、Hungarian
    │   └── paths.py             ← 所有資料夾路徑的集中設定
    └── tools/
        ├── viz_mask_clusters.py ← 只看 mask cluster 的視覺化
        ├── viz_stitch.py        ← 並排比較 mask viz 與 tracking viz
        ├── debug_logger.py      ← per-frame 狀態記錄器
        └── inspect_debug_log.py ← debug log 查詢工具
```

---

## 一、執行追蹤

```bash
python src/main.py --seq seq_4
```

每次執行會在 `data/outputs/seq_4_student_<timestamp>/` 產生：

```
seq_4_student_<timestamp>/
├── result.csv       ← Kaggle 提交用
└── tracking.mp4     ← 追蹤視覺化影片（需 --plot-every 1）
```

### 常用選項

```bash
# 每幀都輸出視覺化（較慢）
python src/main.py --seq seq_4 --plot-every 1

# 同時記錄 debug log（分析用）
python src/main.py --seq seq_4 --plot-every 1 --debug-log

# 關閉 Kalman Filter（直接用 centroid 位置，方便比較）
python src/main.py --seq seq_4 --plot-every 1 --no-kf

# 指定自訂 config
python src/main.py --seq seq_4 --config data/optuna/best_config.yaml
```

---

## 二、調整追蹤參數

所有參數集中在 `src/utils/default_config.yaml`：

```yaml
lifecycle:
  max_age: 10      # 幾幀沒配到就刪除 track
  min_hits: 1      # 要幾次配對才算 confirmed（出現在輸出）

filter:
  type: identity   # "kalman" | "identity"（目前關閉 KF）
  q_var: 100       # KF 過程雜訊（identity 模式下無效）
  r_var: 300       # KF 位置量測雜訊
  r_var_vel: 5.0   # KF 速度量測雜訊
  p_var: 100       # KF 初始協方差

gate:
  max_euclidean: 5.5   # Euclidean 硬閘門（公尺），超過直接排除
  max_distance:  4.0   # Mahalanobis 軟閘門（identity 模式下退化為 Euclidean）

cost:
  range_rate_sigma: 1000.0  # range rate 項的正規化（越大越不重視速度差異）
```

**目前狀態**：`filter.type: identity`，range_rate_sigma 很大（1000），
等同於只用 Euclidean 距離做配對。

---

## 三、評估結果

```bash
python src/evaluate_tracking.py \
  --gt   data/seq_3/gt_answer_seq3.csv \
  --pred data/outputs/seq_3_student_<timestamp>/result.csv \
  --out  data/outputs/seq_3_eval.csv
```

（seq_4 無公開 GT，只能靠 seq_3 估計效果）

---

## 四、視覺化工具

### 4-1 只看 Mask Cluster（無追蹤，純雷達點上色）

```bash
python src/tools/viz_mask_clusters.py --seq seq_4
```

輸出到 `data/viz/seq_4_mask_viz/`，mp4 在 `data/viz/seq_4_mask_clusters.mp4`。

這個視覺化用來確認 cluster mask 本身是否合理，與追蹤結果無關。

### 4-2 並排比較（Mask ↔ Tracking）

```bash
python src/tools/viz_stitch.py \
  --left  data/viz/seq_4_mask_viz \
  --right data/outputs/seq_4_student_<timestamp>
```

左邊是純 mask cluster、右邊是追蹤結果，方便直接對照 centroid 是否被正確繼承。
輸出到 `data/viz/seq_4_student_<timestamp>_stitch/`。

---

## 五、Debug Log 分析

執行時加 `--debug-log` 後，`data/outputs/.../debug.jsonl` 會記錄每幀的完整狀態。

### 看各幀概覽

```bash
python src/tools/inspect_debug_log.py summary \
  --log data/outputs/seq_4_student_<timestamp>/debug.jsonl
```

### 看特定幀的 cost matrix 與配對結果

```bash
python src/tools/inspect_debug_log.py frame \
  --log data/outputs/seq_4_student_<timestamp>/debug.jsonl \
  --frame 87
```

輸出範例：
```
Tracks:   track_id, state, hits, age, pred_pos, velocity, range_rate
Dets:     det_idx, x, y, range_rate, cluster_id
Cost matrix: 行=track, 列=detection, * 標出配對結果, ---- 表示被 gate 排除
Assignments: matched / unmatched_tracks / unmatched_dets
```

### 追蹤單一 track 的歷史

```bash
python src/tools/inspect_debug_log.py track \
  --log data/outputs/seq_4_student_<timestamp>/debug.jsonl \
  --track 5 --frames 80-95
```

---

## 六、HPO 超參數搜尋

```bash
python src/runner/optuna_search.py --seq seq_3 --n-trials 200
```

（seq_3 有 GT，seq_4 沒有，所以用 seq_3 搜）

搜尋完後最佳參數存在 `data/optuna/best_config.yaml`，直接：

```bash
python src/main.py --seq seq_4 --config data/optuna/best_config.yaml
```

TensorBoard 查看搜尋過程：

```bash
python -m tensorboard.main --logdir data/optuna/tb_logs/
```

---

## 七、現況與下一步方向

| 項目 | 目前狀態 |
|---|---|
| Filter | `identity`（KF 關閉），centroid 直接繼承 |
| Gate | Euclidean 5.5m 硬閘門 |
| Cost | 純 Euclidean，speed 項幾乎無效（sigma=1000） |
| 對向車道誤搶 | 已知問題，range rate 可分辨但 sigma 太大導致無效 |
| seq_4 GT | 無公開，只能靠 seq_3 MOTA 估計 |

可能的改善方向：
- 調低 `range_rate_sigma`（讓速度差異真正影響配對成本）
- 開啟 KF（`filter.type: kalman`）並用 Optuna 搜最佳 KF 參數
- 分析 debug log 中 cost matrix 找出誤配對的 frame 規律
