# 工具使用說明

> 所有指令從專案根目錄 `sdc-final/` 執行。

---

## 主程式

### `src/main.py` — 執行追蹤器

```bash
python src/main.py --seq seq_4

# 每幀輸出視覺化圖片與追蹤影片
python src/main.py --seq seq_4 --plot-every 1

# 記錄 debug log（可搭配 inspect_debug_log.py 分析）
python src/main.py --seq seq_4 --plot-every 1 --debug-log

# 停用 Kalman Filter（直接用 cluster centroid 位置）
python src/main.py --seq seq_4 --no-kf

# 指定自訂 config
python src/main.py --seq seq_4 --config data/optuna/best_config.yaml
```

**輸出**：`data/outputs/seq_4_student_<timestamp>/result.csv`（提交用）、`tracking.mp4`（視覺化）

**視覺化版面（4 格）**：

```
┌────────────────┬────────────────┐
│  Mask Clusters │    Tracking    │  ← 雷達俯視（上排）
├────────────────┼────────────────┤
│  Camera        │  YOLO + SAM2   │  ← 相機視角（下排）
│  （乾淨影像）  │  （偵測疊加）  │
└────────────────┴────────────────┘
```

下排右側（YOLO + SAM2 面板）說明：
- **綠色框**：YOLO 偵測到的車輛 bounding box（conf > yolo_conf）
- **彩色 mask**：SAM2 對 camera-frozen tracks 的分割結果（對應 track 顏色）
- **鑽石標記 ❄T{id}**：目前被相機 age 凍結的 track（雷達丟失但相機看得到）
- **圓點 T{id}**：雷達確認的 track 投影位置（對齊驗證用）

**相機輔助 age 凍結（Hybrid Camera Gate）**：
在 `src/utils/default_config.yaml` 把 `camera_gate.enabled` 改為 `true` 即可啟用。

| 機制 | 說明 |
|---|---|
| YOLO | 一級確認，每幀一次推論，命中 bounding box 即凍結 |
| SAM2 | 二級 fallback，YOLO 失效時（遠距離小物件）以語義分割確認 |
| `sam2_score_threshold` | SAM2 信心值門檻（預設 0.70） |
| `sam2_mask_min/max_px` | mask 面積範圍，過濾雜訊與背景大區域 |

需求：已有 `data/outputs/<seq>_extrinsics.npz`、已安裝 `ultralytics` 與 `sam2`。

---

## 相機標定工具

### `src/tools/estimate_intrinsics.py` — 自動估算內參

以消失點（VP + Hough）或基礎矩陣法估算焦距 f 與主點 cx, cy。

```bash
python tools/estimate_intrinsics.py --seq seq_4
python tools/estimate_intrinsics.py --seq seq_4 --method vp --lane-width 3.7 --camera-height 1.2
python tools/estimate_intrinsics.py --seq seq_4 --method fm --step 3 --frames 80
```

**輸出**：`data/outputs/seq_4_camera_K.npy`

> **注意**：VP 法容易因車道線條件而得到偏差結果（本專案曾得到 f=374，實際應為 f≈1005）。建議搭配 `adjust_intrinsics.py` 做目視校正。

---

### `src/tools/estimate_intrinsics_sam2.py` — SAM2 標注自動估算內參

外參固定不動，以 SAM2 mask centroid ↔ Cluster 對應點建立線性方程組，最小二乘解 f, cx, cy。
適用於外參已知但需要精修或重新估算內參的情況。

```bash
python tools/estimate_intrinsics_sam2.py \
    --seq seq_4 \
    --extrinsics data/outputs/seq_4_extrinsics.npz \
    --frames 10 20 30 \
    --out data/outputs/seq_4_camera_K.npy
```

**參數說明**：

| 參數 | 說明 | 預設 |
|---|---|---|
| `--seq` | 序列名稱 | 必填 |
| `--extrinsics` | extrinsics .npz 路徑（含 R, t, K） | 必填 |
| `--frames` | 要標注的幀索引 | `10 30 60` |
| `--out` | 輸出 K.npy 路徑 | `data/outputs/<seq>_camera_K.npy` |
| `--no-save-ext` | 不更新 extrinsics.npz 裡的 K | false |

**流程**：
1. 逐幀：點選車輛 → SAM2 mask → centroid（2D）
2. 逐幀：手動配對 cluster（3D）↔ mask centroid
3. 收集全部對應點後，一次求解 f, cx, cy
4. 顯示前後重投影誤差對比（散點圖）及全圖 cluster 驗證

**輸出**：
- `data/outputs/seq_4_camera_K.npy`（新 K）
- `seq_4_extrinsics.npz` 內的 K 同步更新（除非加 `--no-save-ext`）

---

### `src/tools/adjust_intrinsics.py` — 互動式內參調整

滑桿即時調整 f / cx / cy / k1，相機影像上疊加 cluster 投影點做目視比對。外參固定不動。

```bash
python tools/adjust_intrinsics.py \
    --seq seq_4 \
    --extrinsics data/outputs/seq_4_extrinsics.npz
```

**按鈕**：`[Save K]` 同時更新 `seq_4_camera_K.npy` 與 `seq_4_extrinsics.npz` 裡的 K。

**調整診斷**：

| 觀察 | 調整 |
|---|---|
| 所有點整體左移 | cx 往大調 |
| 所有點整體上移 | cy 往小調 |
| 遠處車偏多、近處準 | f 不對 |
| 中心準、四角往外彎 | k1 往負（桶形畸變） |

> k1 僅供預覽，不存入 K.npy（目前為 pinhole 模型）。

---

### `src/tools/estimate_extrinsics.py` — 互動式外參調整

滑桿調整 pitch / yaw / roll / height / forward / lateral，即時顯示 cluster 投影。

```bash
python tools/estimate_extrinsics.py --seq seq_4 \
    --K data/outputs/seq_4_camera_K.npy \
    --pitch 0 --height 1.3

# 直接存檔
python tools/estimate_extrinsics.py --seq seq_4 \
    --K data/outputs/seq_4_camera_K.npy \
    --save-ext data/outputs/seq_4_extrinsics.npz
```

**輸出**：`data/outputs/seq_4_extrinsics.npz`（含 R, t, K）

---

### `src/tools/annotate_extrinsics_sam2.py` — SAM2 標注 + 手動配對 + PnP 估算外參

最精確的外參估算流程。分兩階段：

1. **FrameAnnotator**：在影像上點擊車輛 → SAM2 產生 mask → 取 centroid
2. **ManualMatcher**：在雷達圖點選 cluster 三角形，再點選對應的影像 mask → PnP 求解

```bash
python tools/annotate_extrinsics_sam2.py \
    --seq seq_4 \
    --K data/outputs/seq_4_camera_K.npy \
    --out data/outputs/seq_4_extrinsics.npz \
    --frames 10 20 30
```

**鍵盤快捷鍵（ManualMatcher）**：

| 按鍵 | 動作 |
|---|---|
| 點選雷達三角 | 選取 cluster |
| 點選影像方塊 | 配對 cluster → mask |
| S | 跳過目前選取的 cluster |
| Z | 撤銷上一個配對 |
| Enter | 完成當前幀 |
| Q / Esc | 跳過此幀 |

---

## 視覺化與驗證工具

### `src/tools/viz_yolo_detections.py` — YOLO 偵測視覺化

驗證 YOLO 偵測結果是否正確。左欄顯示 YOLO bounding box（含信心值），右欄為雷達俯視圖；若有 extrinsics 則同時疊加 cluster 投影點以比對對應關係。

```bash
# 基本使用（自動尋找 extrinsics）
python tools/viz_yolo_detections.py --seq seq_4

# 不疊加 cluster 投影
python tools/viz_yolo_detections.py --seq seq_4 --no-proj

# 快速預覽：前 100 幀、每 3 幀取一幀
python tools/viz_yolo_detections.py --seq seq_4 --frames 100 --every 3

# 調整信心值門檻
python tools/viz_yolo_detections.py --seq seq_4 --conf 0.4
```

**輸出**：`data/viz/seq_4_yolo_detections.mp4`

---

### `src/tools/viz_extrinsics.py` — 外參批量驗證影片

將全序列的雷達 cluster 中心投影到相機影像，輸出驗證影片。

```bash
python tools/viz_extrinsics.py \
    --seq seq_4 \
    --extrinsics data/outputs/seq_4_extrinsics.npz

# 每 2 幀取一幀（加速）
python tools/viz_extrinsics.py --seq seq_4 \
    --extrinsics data/outputs/seq_4_extrinsics.npz \
    --every 2
```

**輸出**：`data/viz/seq_4_extrinsics_check.mp4`

---

### `src/tools/viz_mask_clusters.py` — Radar Cluster 視覺化

顯示雷達點雲的 cluster 分類結果（靜態點灰色，移動點彩色，左下角嵌入相機縮圖）。

```bash
python tools/viz_mask_clusters.py --seq seq_4
python tools/viz_mask_clusters.py --seq seq_4 --every 1 --xlim -40 40 --ylim 0 60
```

**輸出**：`data/viz/seq_4_mask_viz/` 資料夾（PNG 幀）+ `seq_4_mask_viz.mp4`

---

### `src/tools/viz_stitch.py` — 並排比較影片

將兩個視覺化資料夾的同名幀左右拼接，方便對比 cluster 視圖與追蹤視圖。

```bash
python tools/viz_stitch.py \
    --left  data/viz/seq_4_mask_viz \
    --right data/outputs/seq_4_student_<timestamp>
```

**輸出**：`data/viz/<right-folder-name>_stitch.mp4`

---

### `src/tools/viz_ego_velocity.py` — 自車速度視覺化

繪製 ego_global_pos.txt 推算的原始速度 vs 低通濾波速度，含軌跡圖、vx/vy 時序圖、速度大小圖。僅適用於 seq_3 / seq_4（有 ego_global_pos.txt 的序列）。

```bash
python tools/viz_ego_velocity.py --seq seq_3
python tools/viz_ego_velocity.py --seq seq_3 --cutoff 0.05 --order 2
```

**輸出**：靜態 PNG 圖片（預設顯示，可加 `--out path.png` 存檔）

---

## Debug 工具

### `src/tools/inspect_debug_log.py` — 查詢 Debug Log

分析 `main.py --debug-log` 產生的 `debug.jsonl`，支援三種查詢模式。

```bash
# 查看某幀的完整狀態（tracks, detections, cost matrix, assignments）
python tools/inspect_debug_log.py frame \
    --log data/outputs/seq_4_student_<ts>/debug.jsonl --frame 42

# 追蹤某個 track 的完整生命週期
python tools/inspect_debug_log.py track \
    --log data/outputs/seq_4_student_<ts>/debug.jsonl --track 5

# 只看特定幀段
python tools/inspect_debug_log.py track \
    --log data/outputs/seq_4_student_<ts>/debug.jsonl --track 5 --frames 40-60

# 全序列摘要（每幀 track 數、detection 數、matched 數）
python tools/inspect_debug_log.py summary \
    --log data/outputs/seq_4_student_<ts>/debug.jsonl
```

---

## 標定產物路徑速查

| 檔案 | 內容 | 產生工具 |
|---|---|---|
| `data/outputs/seq_4_camera_K.npy` | 內參矩陣 K（3×3）| `estimate_intrinsics.py` / `adjust_intrinsics.py` / `estimate_intrinsics_sam2.py` |
| `data/outputs/seq_4_extrinsics.npz` | R（3×3）、t（3,）、K（3×3）| `annotate_extrinsics_sam2.py` / `estimate_extrinsics.py` |

---

## Pipeline 示意圖

完整的內外參估算流程見 [`doc/pipeline_calibration.svg`](pipeline_calibration.svg)。
