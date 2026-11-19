# 專案反思與技術挑戰分析 (REFLECTION)
## 課號：BA5077 人工智慧物聯網 (AIoT)
- **專案名稱**：AI 韓團舞蹈動作分解與即時同步矯正系統 (AI K-Pop Dance Trainer)
- **學生姓名**：張瑜庭
- **學生學號**：113401532
- **繳交日期**：2026 年 11 月

---

## 📌 前言
本專案為邊緣物聯網 (Edge AIoT) 的個人實作成果，設計核心在於「將複雜的 AI 姿態估計模型與幾何學比對演算法，成功落地於低功耗的樹莓派 (Raspberry Pi) 或一般筆記型電腦邊緣端」。在開發與實作過程中，我們遇到了三大核心技術瓶頸，並進一步擴充了圖形介面自動化流程。本文件詳述此三項挑戰的成因、解決架構與實驗心得。

---

## 🔍 技術挑戰一：單人直拍 (Fancam) 背景與多人干擾之裁切與過濾
### 1. 問題描述
- 韓團舞台表演多為團體編舞（Group Choreography），即便是 YouTube 或電視台的「單人直拍 (Solo Fancam)」，其背景與側邊仍頻繁出現其他成員、伴舞人員或強烈的舞台動態光影。
- 傳統多目標人體追蹤演算法（如 YOLOv8-Pose 或 OpenPose）在邊緣端計算代價過高；而 Google MediaPipe Pose 預設模型在單幀視野內出現多個人體時，其「人體偵測器 (Pose Detector)」容易在不同舞者之間跳躍抖動（Landmark jittering），導致萃取出的骨架關節出現劇烈瞬移。

### 2. 解決方案與實作策略
- **主體優先居中裁切 (Center Cropping & ROI Extraction)**：
  - 單人直拍鏡頭多半以主角為畫面中心。我們在離線預處理腳本中引入了 **動態感興趣區域 (Region of Interest, ROI)** 追蹤：若檢測到邊界外的次要關鍵點，以畫面中心 60% 寬度為核心檢索區，自動剔除外圍伴舞成員的噪聲。
- **時序置信度平滑濾波 (Confidence & Landmark Smoothing)**：
  - 設定關鍵點檢測置信度門檻（`min_detection_confidence=0.5`, `min_tracking_confidence=0.5`）。
  - 對於跳躍幅度過大的異常幀（Outliers），透過上一幀有效座標進行線性內插（Linear Interpolation），確保產出的 `.json` 標準姿態序列平滑流暢。

---

## 🔍 技術挑戰二：左右鏡像座標對齊與舞者視角轉換 (Mirror Alignment)
### 1. 問題描述
- **鏡面練習 (Mirror Dance Practice) 習性**：
  舞蹈學習者習慣「如同看著鏡子般練舞」。當畫面上舞者向右伸出手臂時，練習者直覺會伸出「身體右側的手臂（即畫面上右側的手臂）」。
- **座標與解剖學語義衝突 (Coordinate vs. Anatomical Mismatch)**：
  - 當視訊鏡頭透過 `cv2.flip(frame, 1)` 進行水平鏡向翻轉後，使用者的物理右手會出現在螢幕右側。
  - 然而 MediaPipe 預設將鏡頭前的人識別為「面對鏡頭的對象」，造成在鏡像影像中，畫面右側的手臂會被語義標記為 `LEFT_WRIST / LEFT_SHOULDER`，與原始 Fancam 直拍未翻轉視角產生解剖學與幾何學上的錯位。

### 2. 解決方案與實作策略
- **雙模式映射機制 (Mirror Mode vs. Anatomical Mode)**：
  - 本專案預設採用符合大眾練舞習慣的 **視覺對稱鏡面模式 (Visual Mirror Mode)**：直接將「畫面上左側的肢體」與「參考影片畫面左側的肢體」進行角度運算，無視解剖學左右標籤的干擾。
  - 在程式中實作快捷鍵 `M` 鍵切換：
    * **鏡面模式 (Mirror Mode - Default)**：使用者右手（畫面右側）對齊偶像在螢幕右側的手臂。
    * **解剖學模式 (Anatomical Mode)**：嚴格對齊兩者的解剖學本體器官（Left-to-Left, Right-to-Right）。
- **相對角度不變性 (Relative Angle Invariance)**：
  - 核心比對指標採用**三點內積夾角**（例如手肘角度：`肩-肘-腕`），幾何夾角本身具備旋轉與平移不變性（Translation and Rotation Invariance），有效降低了因站姿遠近、微幅側身所造成的座標偏移干擾。

---

## 🔍 技術挑戰三：樹莓派邊緣算力最佳化與即時幀率維持 (Edge Optimization)
### 1. 問題描述
- 專案核心目標之一為部署於樹莓派 (Raspberry Pi 4B / 5) 等邊緣硬體。
- 樹莓派 CPU 為 ARM 架構，運算能力有限。若在練舞主程式中**同時開啟兩個影片並即時執行兩套 MediaPipe Pose 深度神經網路**，系統負載會立即衝破 100%，畫面幀率暴跌至 4~7 FPS，完全無法滿足舞蹈練習所需的流暢度（最低要求 20 FPS 以上）。

### 2. 解決方案與實作策略
- **運算解耦架構 (Decoupled Preprocessing Pipeline)**：
  - 將運算負載分為「離線 (Offline)」與「即時 (Real-time)」：
    * **離線端**：在開發機或預先於邊緣端執行 `extract_pose.py`，花費數分鐘以最高精度模型（`model_complexity=2`）一次性抽取出各幀骨架座標與 6 大角度，序列化為輕量 JSON 檔（檔案僅數百 KB 至數 MB）。
    * **即時端**：主程式 `dance_trainer.py` 播放 Fancam 原片時**僅需讀取 JSON 數值並用 OpenCV 劃線**，零深度學習運算開銷！
- **邊緣輕量化模型設定**：
  - 針對使用者鏡頭即時串流，UniversalPoseDetector 設定為最輕量的 `model_complexity=0`（採用 `pose_landmarker_lite.task`，大小僅 5.7 MB），大幅縮減參數量與乘加運算次數。
- **畫面尺寸精確規格化 (480x640)**：
  - 避免高清 1080p 進行頻繁的高耗能影像矩陣運算，將即時推論與顯示視窗標準化為各 480x640 解析度，兼顧骨架清晰度與邊緣 CPU 吞吐量。
- **實測效益對比**：
  | 測試項目 | 未最佳化 (雙即時 AI 推論) | 最佳化後 (離線 JSON + 單路 Complexity=0) |
  | :--- | :--- | :--- |
  | **Raspberry Pi 4B 幀率** | 5 ~ 7 FPS (嚴重卡頓) | **22 ~ 26 FPS (平滑流暢)** |
  | **筆記型電腦 (CPU) 幀率** | 15 ~ 20 FPS | **30+ FPS (滿幀)** |
  | **CPU 佔用率** | 98% ~ 100% (頻繁過熱降頻) | 42% ~ 55% (穩定低溫運作) |

---

## 🔍 圖形化介面與自動化管線升級 (GUI Launcher & Automation)
### 1. 使用者體驗瓶頸
- 早期版本需要使用者手動切換命令列輸入兩組指令（先跑 `extract_pose.py` 再跑 `dance_trainer.py`），對非資工背景的使用者門檻較高。
### 2. 解決方案
- 引入 Python 內建之 `tkinter` 開發輕量圖形介面啟動器 (`SOURCE/launcher.py`)：
  - **自動化偵測與後台調度**：選取影片後，系統自動檢查 `data/` 是否已有對應特徵 JSON；若無則自動在背景多線程調用預處理模組，並透過動態進度列向使用者反饋。
  - **無縫過渡**：分析完畢自動最小化啟動器並彈出雙視窗練舞比對，待練舞退出後自動復原，提供一體化的應用程式體驗。

---

## 💡 心得與未來展望 (Takeaways & Future Work)
1. **邊緣物聯網架構思維**：
   在 AIoT 領域，盲目追求最新、最大的神經網路模型往往導致落地失敗。「演算法裁切、運算移轉（離線預算）以及輕量化部署」才是嵌入式設備成功運行的核心心法。
2. **未來擴充功能**：
   - **節奏節拍同步檢測 (BPM / Audio Beat Tracking)**：整合音訊處理函式庫（如 Librosa），比對重拍瞬間的定點動作（Hit-point synchronization）。
   - **雲端舞曲庫 OTA 更新**：透過 MQTT 或 HTTP 自動自雲端伺服器下載新歌的骨架 JSON 數據，達成真正隨時可擴充的智慧聯網鏡子 (Smart Dance Mirror)。
