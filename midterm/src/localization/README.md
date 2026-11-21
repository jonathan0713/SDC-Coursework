
# Localization Package

這是一個基於 ROS Noetic 的定位套件，專案名稱為 `localization` 。主要功能包含將 3D 點雲地圖進行降採樣處理，以及使用 GTSAM 結合 NDT、Radar 與 IMU 進行多感測器融合的最佳化定位。

## 資料夾架構說明

```text
├── CMakeLists.txt
├── config
│   ├── localization.rviz
│   └── visualization.rviz
├── data
│   ├── Track1
│   │   ├── imu_data.csv
│   │   ├── init_guess.txt
│   │   ├── map_modified.pcd
│   │   ├── radar_t_list.txt
│   │   └── tf_static.csv
│   └── Track2
│       ├── imu_data.csv
│       ├── init_guess.txt
│       ├── map_modified.pcd
│       ├── radar_t_list.txt
│       └── tf_static.csv
├── include
│   └── localization
├── launch
│   ├── localization.launch
│   ├── map_modified.launch
│   └── visualization.launch
├── localization_report_314605004_111612048.pdf
├── package.xml
├── processing_codes
│   ├── radar_preprocess.py
│   └── txt_to_idx_x_y_yaw.py
├── README.md
├── result
│   ├── result_1.csv
│   ├── result_2.csv
│   ├── result_track1.txt
│   └── result_track2.txt
└── src
    ├── localization.cpp
    ├── map_modified.cpp
    ├── pure_ndt.cpp
    ├── testa.cpp
    ├── testa_track1.cpp
    └── visualization.cpp

```
檔案說明：

/data 里會有我們處理過得map `map_modified.pcd` 以及從rosbag里拉出的`imu_data.csv`, `tf_static.csv`

/result 里會有原始的結果`result_track1.txt`, `result_track2.txt`。以及可上傳得結果`result_1.csv`, `result_2.csv`

/src 里會有主程式 `testa.cpp`, `testa_track1.cpp`, `map_modified.cpp`, `visualization.cpp`。以及已經不使用的`localization.cpp`



## 系統需求與相依套件 (Dependencies)

在編譯此專案之前，請確保您的系統為 **Ubuntu 20.04** 與 **ROS Noetic**，並且已安裝以下相依套件：

* **PCL (Point Cloud Library)**: 用於點雲的濾波（VoxelGrid, PassThrough）與 NDT 配準。

```bash
sudo apt update
sudo apt install libpcl-dev ros-noetic-pcl-ros ros-noetic-pcl-conversions
```
* **GTSAM**: 用於建構因子圖 (Factor Graph) 與 ISAM2 最佳化。
```bash
sudo apt update
sudo apt-get install ros-noetic-gtsam
```




## 編譯指南 (Build Instructions)

請將本套件放置於您的 ROS 工作區 (例如 `~/catkin_ws/src`) 內，然後依照以下步驟進行編譯：

```bash
# 進入您的 ROS 工作區
cd ~/catkin_ws

# 進行編譯 (預設會使用 C++14 與 -O3 優化)
catkin_make

# 載入環境變數
source devel/setup.bash
```

## 執行步驟 (Usage)

本套件包含多個節點，請開啟多個終端機 (Terminal) 依序執行以下指令：

### 1. 啟動 ROS 核心
在第一個終端機執行：
```bash
roscore
```

### 2. 啟動 RViz 視覺化工具
在第二個終端機開啟 RViz 以便觀察點雲與軌跡：
```bash
rviz
```
*(在 RViz 中，請將 Fixed Frame 設為 `map`，並可新增 `PointCloud2` 與 `Path` 顯示以訂閱本套件發布的 Topics)*

### 3. 執行地圖前處理節點
在第三個終端機執行 `map_modified` 節點。此程式會讀取原始的 `map.pcd`，進行邊界濾波並以 1.0m 網格進行降採樣，最後輸出 `map_modified.pcd` 供後續定位使用：
```bash
# 請記得先 source devel/setup.bash
rosrun localization map_modified
```

### 4. 執行主定位融合節點
當地圖處理完成後，在第四個終端機執行 `testa` 節點。此節點會讀取雷達與 IMU 數據，並使用 GTSAM 進行即時定位軌跡計算：
```bash
# For Track1
# 請記得先 source devel/setup.bash
rosrun localization testa_track1
```

```bash
# For Track2
# 請記得先 source devel/setup.bash
rosrun localization testa
```
---

## 視覺化與輸出的 Topics

當 `testa` 節點執行時，會在 ROS 中發布以下 Topics，您可以在 RViz 中加入對應的 Display 來觀察：
* `/visual/map` (`sensor_msgs/PointCloud2`): 全域降採樣地圖。
* `/visual/radar` (`sensor_msgs/PointCloud2`): 當前幀的雷達點雲。
* `/visual/tranformed_radar_pose` (`geometry_msgs/PoseStamped`): 雷達當前的姿態與位置。
* `/visual/localization_path` (`nav_msgs/Path`): GTSAM 最佳化後的平滑定位軌跡。

程式執行完畢後，最終的定位軌跡會自動輸出並儲存於 `data/result` 的 `result.txt` 檔案中。
