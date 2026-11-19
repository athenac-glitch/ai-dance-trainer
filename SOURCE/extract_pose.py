"""
extract_pose.py
---------------
韓團單人直拍 (Solo Fancam) 離線骨架姿態與關節夾角預處理腳本
課程：BA5077 人工智慧物聯網 (AIoT)
功能：
1. 讀取輸入之韓團單人直拍 .mp4 影片
2. 使用 Google MediaPipe Pose (高精度模型 complexity=2) 提取每幀 33 關鍵點
3. 幾何運算 6 大核心關節夾角 (雙手肘、雙肩膀抬起、雙膝蓋彎曲)
4. 匯出輕量化 JSON 結構檔，作為樹莓派/筆電即時比對之低負載標準數據
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# 加入當前目錄以匯入 pose_utils
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from pose_utils import (
    UniversalPoseDetector,
    extract_6_joint_angles,
    draw_skeleton,
    resize_and_pad,
)


def find_default_video(videos_dir: Path) -> Path:
    """若未指定影片，自動搜尋 videos/ 目錄下的 .mp4 檔案"""
    default_fancam = videos_dir / "fancam.mp4"
    if default_fancam.exists():
        return default_fancam

    mp4_files = list(videos_dir.glob("*.mp4"))
    if mp4_files:
        return mp4_files[0]
    return default_fancam


def process_fancam_video(
    video_path: str,
    output_json_path: str = None,
    show_preview: bool = False,
    model_complexity: int = 2,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
):
    video_file = Path(video_path).resolve()
    if not video_file.exists():
        print(f"[錯誤] 找不到輸入影片檔案：{video_file}")
        print("💡 請將韓團直拍影片放置於 videos/ 資料夾中，或以 --video 指定正確路徑。")
        sys.exit(1)

    # 決定輸出 JSON 路徑
    if output_json_path is None:
        data_dir = CURRENT_DIR.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_json_file = data_dir / f"{video_file.stem}_pose.json"
    else:
        output_json_file = Path(output_json_path).resolve()
        output_json_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🎬 AI K-Pop Dance Trainer - 離線姿態提取與特徵工程")
    print(f"🔹 輸入影片：{video_file.name} ({video_file})")
    print(f"🔹 輸出路徑：{output_json_file}")
    print(f"🔹 離線模型複雜度：model_complexity={model_complexity} (最高精確度模式)")
    print("=" * 65)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"[錯誤] 無法以 OpenCV 開啟影片：{video_file}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 影片規格：解析度 {width}x{height} | 總影格數 {total_frames} | 影格率 {fps:.2f} FPS")

    # 初始化跨平台通用姿態偵測器 (UniversalPoseDetector)
    pose_detector = UniversalPoseDetector(
        complexity=model_complexity,
        static_image_mode=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    extracted_frames = []
    frame_idx = 0
    detected_count = 0
    start_time = time.time()
    last_valid_angles = {
        "left_elbow": 160.0,
        "right_elbow": 160.0,
        "left_shoulder": 30.0,
        "right_shoulder": 30.0,
        "left_knee": 175.0,
        "right_knee": 175.0,
    }
    last_valid_landmarks = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用姿態偵測器進行關鍵點預測
            landmarks = pose_detector.process(frame)

            has_pose = False
            landmarks_data = []
            angles = {}

            if landmarks:
                has_pose = True
                detected_count += 1
                landmarks_data = [
                    [
                        round(pt[0], 5),
                        round(pt[1], 5),
                        round(pt[2], 5),
                        round(pt[3], 3),
                    ]
                    for pt in landmarks
                ]
                # 計算 6 大核心關節角度
                angles = extract_6_joint_angles(landmarks_data)
                last_valid_angles = angles
                last_valid_landmarks = landmarks_data
            else:
                # 姿態短暫遺失時，使用上一幀有效姿態避免骨架斷裂跳動
                landmarks_data = last_valid_landmarks if last_valid_landmarks else []
                angles = last_valid_angles

            timestamp_sec = round(frame_idx / fps, 3)
            extracted_frames.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "has_pose": has_pose,
                "angles": angles,
                "landmarks": landmarks_data,
            })

            # 進度列與終端機回報
            frame_idx += 1
            if frame_idx % 15 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_idx / (elapsed + 1e-5)
                percent = (frame_idx / max(1, total_frames)) * 100.0
                sys.stdout.write(
                    f"\r⚡ 預處理進度: {frame_idx}/{total_frames} 幀 ({percent:.1f}%) | "
                    f"速度: {curr_fps:.1f} FPS | 成功偵測: {detected_count} 幀"
                )
                sys.stdout.flush()

            # 可視化預覽模式 (若環境支援 GUI)
            if show_preview:
                preview_frame, _, _, _ = resize_and_pad(frame, 480, 640)
                if landmarks_data:
                    draw_skeleton(preview_frame, landmarks_data, color=(0, 255, 255), thickness=2)
                cv2.putText(
                    preview_frame,
                    f"Frame: {frame_idx}/{total_frames}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                try:
                    cv2.imshow("Fancam Pose Extraction Preview (Press 'q' to stop preview)", preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[提示] 使用者中斷預覽，繼續在後台完成提取...")
                        show_preview = False
                        cv2.destroyAllWindows()
                except cv2.error:
                    show_preview = False

    finally:
        cap.release()
        pose_detector.close()
        if show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n")
    elapsed_total = time.time() - start_time
    print(f"✅ 姿態特徵提取完成！耗時：{elapsed_total:.2f} 秒 (平均 {(frame_idx/elapsed_total):.1f} FPS)")
    print(f"📈 姿態捕捉覆蓋率：{detected_count}/{frame_idx} 幀 ({(detected_count/max(1, frame_idx)*100):.1f}%)")

    # 封裝 JSON 結構數據
    dataset = {
        "metadata": {
            "project": "AI K-Pop Dance Trainer (Edge AIoT BA5077)",
            "video_source": video_file.name,
            "original_width": width,
            "original_height": height,
            "fps": round(fps, 2),
            "total_frames": frame_idx,
            "joint_definitions": [
                "left_elbow",
                "right_elbow",
                "left_shoulder",
                "right_shoulder",
                "left_knee",
                "right_knee",
            ],
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "frames": extracted_frames,
    }

    # 輸出輕量 JSON 檔案
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, separators=(",", ":"))

    file_size_kb = output_json_file.stat().st_size / 1024.0
    print(f"💾 JSON 資料檔已成功生成：{output_json_file} ({file_size_kb:.1f} KB)")
    print("🚀 樹莓派端即可直接載入此 JSON，即時練舞時零 AI 重複運算！")


def main():
    parser = argparse.ArgumentParser(
        description="AI K-Pop Dance Trainer - 單人直拍姿態特徵離線萃取器 (BA5077)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="輸入韓團直拍 .mp4 檔案路徑 (預設自動搜尋 videos/fancam.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="指定輸出 JSON 路徑 (預設為 data/<video_name>_pose.json)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="是否於處理時開啟 OpenCV 畫面即時預覽骨架提取效果",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe Pose 模型複雜度 (0:最輕量, 1:中等, 2:最高精度，離線推薦 2)",
    )

    args = parser.parse_args()

    project_root = CURRENT_DIR.parent
    videos_dir = project_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        target_video = args.video
    else:
        target_video = str(find_default_video(videos_dir))

    process_fancam_video(
        video_path=target_video,
        output_json_path=args.output,
        show_preview=args.preview,
        model_complexity=args.complexity,
    )


if __name__ == "__main__":
    main()
