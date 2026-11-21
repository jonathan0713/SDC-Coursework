# Self-Driving Cars 課程專案

這個 repo 收錄原本放在 `ROS1/` 工作區的 SDC 作業、期中與期末專案。各子目錄沿用原本的專案名稱及配置。

| 目錄 | 內容 |
| --- | --- |
| [`Lin_Ying/`](Lin_Ying/) | ROS Noetic 練習環境，含 Dockerfile 與 Makefile |
| [`sdc_hw2/`](sdc_hw2/) · [`sdc_hw3/`](sdc_hw3/) · [`sdc_hw4/`](sdc_hw4/) · [`sdc_hw5/`](sdc_hw5/) | 作業程式、報告和原有提交檔 |
| [`midterm/`](midterm/) | ROS Noetic 定位。套件在 [`midterm/src/localization/`](midterm/src/localization/)，已整合原 `ROS1/localization/` 的 README |
| [`sdc-midterm2-5-2026/`](sdc-midterm2-5-2026/) | LiDAR SLAM 與 3D Gaussian Splatting 重建 |
| [`sdc-final-14-2026/`](sdc-final-14-2026/) | 第 14 組期末追蹤專案 |
| [`sdc-final-2026/`](sdc-final-2026/) | 靜止與移動自車的 4D 雷達多目標追蹤，含兩份期末結果 CSV |

## 執行與資料

- 在 `midterm/` 執行 `docker compose up --build` 可啟動 ROS Noetic 容器。定位套件的參數與資料路徑見 [定位 README](midterm/src/localization/README.md)。如需上游 `fast_gicp`，在 repo 根目錄執行 `git submodule update --init --recursive midterm/src/fast_gicp`。
- 期中定位的 `.bag`、`.pcd` 和雷達影格資料保留於原 `ROS1/midterm/src/localization/data/`。需要重跑時，將相應資料放回新 repo 的同名相對路徑。
- 期中二資料在原 `ROS1/sdc-midterm2-5-2026/data/`。期末測試資料依各專案 README 放入 `public_dev/` 和 `test_input/`。虛擬環境、編譯產物、模型重建輸出及除錯影片留在原工作區。

## 來源

整合內容取自原專案的本機工作目錄，包含整合當時尚未提交的改動。原本的 Git 歷史保留在各自的 repo。

| 專案 | 整合時的來源提交 |
| --- | --- |
| 期中定位 | [`68e8752`](https://github.com/jonathan0713/NYCU-Self-Driving-Cars/commit/68e8752be5f18dcefc349ddbf8ed3085f4d42d19) |
| 期中二 | [`8e14105`](https://github.com/BianIO/sdc-midterm2-5-2026/commit/8e14105) |
| 第 14 組期末追蹤 | [`cb4201c`](https://github.com/BianIO/sdc-final-14-2026/commit/cb4201c) |
| 期末 4D 雷達追蹤 | [`bce9a32`](https://github.com/jonathan0713/sdc-final-2026/commit/bce9a32) |
