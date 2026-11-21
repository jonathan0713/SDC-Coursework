# Kalman Filter：預測與觀測模型設計

> 對應實作：`src/utils/filter.py` → `KalmanFilter`  
> 相關呼叫端：`src/sdc_tracker.py` → `Tracker.update()`

---

## 1. 模型概要

追蹤器採用**等速度（Constant Velocity, CV）Kalman Filter**，  
狀態空間定義在**自車（雷達）座標系**，不需轉換至世界座標系。

| 向量 | 形狀 | 內容 |
|---|---|---|
| 狀態 $\mathbf{x}$ | 4 × 1 | $[x,\ y,\ v_x,\ v_y]^\top$ |
| 基本觀測 $\mathbf{z}$ | 2 × 1 | $[x,\ y]^\top$（僅位置） |
| 增廣觀測 $\mathbf{z}'$ | 3 × 1 | $[x,\ y,\ \dot{r}_j]^\top$（位置 + 原始 range rate） |

增廣觀測採用**未補償的原始 range rate**，KF 速度狀態因此收斂至物體在雷達座標系下的**表觀速度**（apparent velocity）。
此設計的詳細討論見第 4.3 節。

### 關於線性 KF vs EKF

增廣觀測的第三項 $\dot{r}_j = v_x \cos\theta_j + v_y \sin\theta_j$ 以 $\theta_j$ 為參數。

- 若 $\theta_j$ 由 **KF 預測位置** $\hat{\mathbf{x}}$ 計算 → 觀測函數對狀態非線性 → 需要 **EKF**。
- 本實作由**當幀偵測位置** $(x_j, y_j)$ 計算 $\theta_j$ → $\mathbf{H}'$ 在每次 `correct()` 中為已知常數矩陣 → 觀測更新**嚴格線性**，不需要 EKF。

---

## 2. 輸入訊號

### 2.1 雷達偵測（每幀，逐偵測點）

來源：`utils/detection.py` → `build_detections()`

```
detection[j] = [x_j, y_j, z_j, cluster_id_j, ṙ_j]
```

| 訊號 | 說明 |
|---|---|
| $x_j,\ y_j$ | Cluster 重心，自車座標系（前向 = X，側向 = Y），單位 m |
| $\dot{r}_j$ | **Range rate**：物體相對雷達的徑向速度（m/s）。正值表示遠離，負值表示接近。由 4D 雷達都卜勒量測直接提供，是**相對速度**（非絕對速度）。增廣觀測直接使用此值，不作 ego 補償 |

> **Ego 速度**：目前版本中 ego 速度僅保留於管道架構中，不參與 KF 觀測計算。
> 正確的 ego 補償需要自車航向（heading）才能將世界座標系速度轉換至雷達座標系；
> `ego_global_pos.txt` 的姿態欄位全為零（無效），座標系轉換無從實現。
> 待後續取得可靠航向資訊後，再引入 ego 速度補償。

---

## 3. Predict 步驟

### 3.1 方程

$$
\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\, \hat{\mathbf{x}}_{t-1|t-1}
$$

$$
\mathbf{P}_{t|t-1} = \mathbf{F}\, \mathbf{P}_{t-1|t-1}\, \mathbf{F}^\top + \mathbf{Q}
$$

### 3.2 系統矩陣

狀態轉移矩陣（等速模型）：

$$
\mathbf{F} =
\begin{bmatrix}
1 & 0 & dt & 0 \\
0 & 1 & 0  & dt \\
0 & 0 & 1  & 0 \\
0 & 0 & 0  & 1
\end{bmatrix}
$$

`dt`（幀間時間間隔）預設 1.0，可由 `filter.dt` 設定。

過程雜訊（等向性，位置與速度統一）：

$$
\mathbf{Q} = \mathbf{I}_4 \cdot q\_var
$$

`q_var` 控制模型允許的加速度不確定性。值越大，KF 越快跟上突然的速度變化，但也對量測雜訊更敏感。

### 3.3 初始條件

新建 Track 時：

$$
\hat{\mathbf{x}}_0 = [x_\text{det},\ y_\text{det},\ 0,\ 0]^\top
$$

$$
\mathbf{P}_0 = \text{diag}(p\_var,\ p\_var,\ 10 \cdot p\_var,\ 10 \cdot p\_var)
$$

速度分量給予 10 倍初始協方差，反映速度完全未知；位置協方差反映偵測定位的初始不確定性。

---

## 4. Observation（Correct）步驟

### 4.1 方程

$$
\mathbf{S} = \tilde{\mathbf{H}}\, \mathbf{P}\, \tilde{\mathbf{H}}^\top + \tilde{\mathbf{R}}
$$

$$
\mathbf{K} = \mathbf{P}\, \tilde{\mathbf{H}}^\top\, \mathbf{S}^{-1}
$$

$$
\hat{\mathbf{x}} = \hat{\mathbf{x}} + \mathbf{K}\,(\mathbf{z} - \tilde{\mathbf{H}}\,\hat{\mathbf{x}})
$$

$$
\mathbf{P} = (\mathbf{I} - \mathbf{K}\,\tilde{\mathbf{H}})\,\mathbf{P}
$$

$\tilde{\mathbf{H}}$、$\tilde{\mathbf{R}}$ 依觀測向量長度在每次 `correct()` 呼叫時選擇（見下節）。

---

### 4.2 模式一：基本觀測（$|\mathbf{z}| = 2$）

**觸發條件**：range rate 不可信或不存在

$$
\mathbf{z} = \begin{bmatrix} x_j \\ y_j \end{bmatrix},
\qquad
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix},
\qquad
\mathbf{R} = \mathbf{I}_2 \cdot r\_var
$$

速度分量 $[v_x, v_y]$ 不可觀測，完全靠連續幀的位置差異收斂。

---

### 4.3 模式二：增廣觀測（$|\mathbf{z}'| = 3$）

**觸發條件**：4D 雷達提供有效 range rate（一般情況，不依賴 ego 速度）

#### 觀測向量與矩陣

$$
\mathbf{z}' = \begin{bmatrix} x_j \\ y_j \\ \dot{r}_j \end{bmatrix},
\qquad
\mathbf{H}' = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & \cos\theta_j & \sin\theta_j
\end{bmatrix},
\qquad
\mathbf{R}' = \operatorname{diag}(r\_var,\ r\_var,\ r\_var\_vel)
$$

其中 $\theta_j = \operatorname{atan2}(y_j, x_j)$，由當幀偵測位置決定（每次 `correct()` 更新）。

#### 速度狀態的物理意義：表觀速度

原始 range rate 的定義為物體相對雷達的徑向速度：

$$
\dot{r}_j = (\mathbf{v}_\text{obj} - \mathbf{v}_\text{ego}) \cdot \hat{\mathbf{r}}_j
$$

將此直接作為觀測，KF 擬合的是：

$$
v_x \cos\theta_j + v_y \sin\theta_j \;=\; \dot{r}_j \;=\; (v_\text{obj,r} - v_\text{ego,r})
$$

因此速度狀態 $[v_x, v_y]$ 收斂至**表觀速度**（物體相對雷達的速度），而非物體的絕對速度。

| 場景 | $\dot{r}_j$ 含義 | $[v_x, v_y]$ 收斂結果 |
|---|---|---|
| seq 1/2（ego 靜止）| $= \mathbf{v}_\text{obj} \cdot \hat{\mathbf{r}}$ | 物體**絕對**速度的徑向分量 |
| seq 3/4（ego 移動）| $= (\mathbf{v}_\text{obj} - \mathbf{v}_\text{ego}) \cdot \hat{\mathbf{r}}$ | 物體**相對雷達**的表觀速度 |

#### 表觀速度對 Predict 的正確性

KF 的 predict 步驟在雷達座標系下外推位置：

$$
\hat{x}_{t+1} = \hat{x}_t + v_x \cdot dt
$$

對 seq 3/4，即使 ego 以速度 $v_\text{ego,x}$ 前進，前方靜止物體的表觀速度為 $v_x \approx -v_\text{ego,x}$（向後移動）。predict 因此正確預測了下一幀中該靜止物體在雷達座標系的位置——這正是追蹤所需要的。**表觀速度在雷達座標系下的預測是自洽的**，無需 ego 補償。

#### 可觀測性

| 速度分量 | 單幀可觀測性 | 說明 |
|---|---|---|
| 徑向 $\dot{r}_j = v_x\cos\theta + v_y\sin\theta$ | ✓ 直接觀測 | 由當幀 range rate 提供 |
| 切線 $v_t = -v_x\sin\theta + v_y\cos\theta$ | ✗ 不可觀測 | 需依賴位置差異收斂 |

隨物體相對雷達移動，$\theta_j$ 跨幀改變，多幀後 $[v_x, v_y]$ 完全可觀（$\theta^{(1)} \neq \theta^{(2)}$ 時觀測矩陣滿秩）。

---

## 5. Cost Function 中的 Range Rate 一致性

代價矩陣的 range rate 項直接使用原始相對速度，兩側基準一致：

$$
c_{rr}(i,j) = \frac{|\dot{r}_j - \dot{r}_i|}{\sigma_{rr}}
$$

| 偵測側 | Track 側 |
|---|---|
| $\dot{r}_j$（當幀 range rate） | `track.det[4]`（上次配對時記錄的 range rate） |

不作 ego 補償，兩側均為相對速度，差值不受座標系影響。

---

## 6. 參數一覽

所有參數由 `src/utils/default_config.yaml` 統一管理：

```yaml
filter:
  q_var: 1.0        # 過程雜訊方差（控制模型對加速度的容許程度）
  r_var: 1.0        # 位置量測雜訊方差
  p_var: 10.0       # 初始協方差對角元素
  dt: 1.0           # 幀間時間間隔
  r_var_vel: 1.0    # 增廣觀測中 v_r 的量測雜訊方差

ego_lowpass:
  cutoff: 0.1       # Butterworth 正規化截止頻率
  order: 4          # 濾波器階數（filtfilt 雙向套用，有效階數為 2n）
```
