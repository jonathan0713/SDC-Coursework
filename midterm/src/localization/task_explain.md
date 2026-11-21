根據簡報內容，您的任務「SDC 2026 Midterm01 Competition - Radar Localization」包含了以下幾個主要階段與細節：

### 1. 分組與組隊
* [cite_start]這是一個團隊競賽，每組至多兩人 [cite: 1]。
* [cite_start]您需要在簡報提供的 Google 試算表連結中填寫分組名單 [cite: 1]。
* [cite_start]必須在 Kaggle 上建立以 `group_id` 命名的隊伍，並邀請您的隊友加入 [cite: 1]。

### 2. 演算法開發與實作 (競賽核心)
* [cite_start]**目標**：利用 3D Lidar 地圖和時間序列的雷達（Radar）數據輸入來進行定位（Localization） [cite: 1]。
* **主要實作**：
    * [cite_start]下載資料集與程式碼範本 [cite: 1]。
    * [cite_start]設計並實作您的 scan matching 演算法（例如 ICP、NDT 等）來求得 pose [cite: 1]。
    * [cite_start]程式碼實作主要在 `/src/localization.cpp` 中完成 [cite: 1]。
    * [cite_start]您可以視需求選擇性地修改 `map_modified.cpp`（用於過濾或降採樣地圖以加速）或使用 `radar_preprocess.py` 進行資料預處理 [cite: 1]。
    * [cite_start]您可以修改任何範本程式碼、使用其他套件，或是完全從頭開始撰寫 [cite: 1]。
* [cite_start]**計分條件**：您的演算法表現必須超越 baseline 才能獲得分數 [cite: 1]。

### 3. Kaggle 競賽提交
* [cite_start]您必須將 Track01 與 Track02 兩個賽道的結果上傳至 Kaggle [cite: 1]。
* [cite_start]需使用附帶的 `/process/txt_to_idx_x_y_yaw.py` 將 `result.txt` 轉換成符合 Kaggle 格式的檔案 [cite: 1]。
* [cite_start]提交的結果必須包含每個 Radar scan 的 pose [cite: 1]。
* [cite_start]每日最多可提交 15 次 [cite: 1]。
* [cite_start]**Kaggle 截止時間**：4月16日 09:00 AM [cite: 1]。

### 4. 報告與上台發表 (Report and Presentation)
* [cite_start]**口頭報告**：準備一個 4 到 6 分鐘的簡報，發表日期為 4月16日 [cite: 1]。
    * [cite_start]內容需包含：工作介紹、以影片形式呈現的結果、以及您的貢獻 [cite: 1]。
* **書面報告**：
    * [cite_start]內容需包含：Pipeline (程式如何運作)、貢獻 (與他人的差異)、遇到的問題與解決方案等 [cite: 1]。
    * [cite_start]評分注意：貢獻分數取決於您實作的想法或解決問題的方法；若純粹使用整個開源專案而無自己的想法，貢獻部分將為零分 [cite: 1]。
    * [cite_start]報告內需附上您的簡報影片連結 [cite: 1]。

### 5. E3 最終繳交
* [cite_start]競賽結束後，每組只需上傳一個檔案至 E3 [cite: 1]。
* [cite_start]檔案格式需為 `.zip`，命名規則為 `<student_id1_student_id2>_localization.zip` [cite: 1]。
* 壓縮檔內需包含：
    * [cite_start]完整的 `localization/` 資料夾（不含 `data` 資料夾）及預處理過的雷達資料（若是純用自己的方法則不用附） [cite: 1]。
    * [cite_start]您撰寫的任何程式碼 [cite: 1]。
    * [cite_start]兩個賽道中最佳的提交結果，分別命名為 `result_1.csv` 與 `result_2.csv` [cite: 1]。
    * [cite_start]書面報告 PDF 檔，命名為 `localization_report_<student ID1_student ID2>.pdf` [cite: 1]。
    * [cite_start]一個說明如何執行您程式碼的 `README` 檔案 [cite: 1]。
* [cite_start]**E3 繳交截止時間**：4月23日 23:59 PM [cite: 1]。