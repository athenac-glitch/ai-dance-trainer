"""
dance_trainer.py
----------------
AI 韓團舞蹈動作分解與即時同步矯正系統 - 雙視窗主程式
課程：BA5077 人工智慧物聯網 (AIoT)
功能特色：
1. 左右雙分割視窗 (np.hstack, 各 480x640，總視窗 960x640)
2. 左側視窗：播放 Fancam 直拍，疊加 preprocessed JSON 骨架 (零 AI 重複推論，保護邊緣算力)
3. 右側視窗：USB 鏡頭輸入水平鏡向翻轉 (cv2.flip(1))，UniversalPoseDetector complexity=0 輕量推論
4. 即時姿態比對：6 大幾何角度逐幀比對，容許誤差 ±20 度
5. 動態視覺反饋：標準亮綠色、誤差亮紅色，即時 HUD 儀表板顯示總體 Sync %
6. 互動快捷鍵：[Space] 暫停定格、[S] 0.5x 慢動作分解、[M] 鏡像/解剖模式、[R] 重播、[Q] 退出
"""

import os
import sys
import time
import json
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
    compare_pose_angles,
    resize_and_pad,
    draw_skeleton,
    draw_hud,
)


def load_pose_dataset(pose_json_path: Path) -> dict:
    """載入離線預先提取的骨架姿態 JSON 數據"""
    if not pose_json_path.exists():
        print(f"[錯誤] 找不到對應的骨架數據檔：{pose_json_path}")
        print("💡 請先針對此影片執行預處理腳本：")
        print(f"   python SOURCE/extract_pose.py --video <您的影片路徑>")
        sys.exit(1)

    print(f"📂 正在載入標準骨架數據：{pose_json_path.name}...")
    with open(pose_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    frame_dict = {f["frame_idx"]: f for f in frames}
    print(f"✅ 成功載入 {len(frame_dict)} 幀標準姿態特徵！")
    return frame_dict


def open_camera(cam_index: int = 0):
    """
    開啟視訊鏡頭（Windows 優先採用 DirectShow，相容性與啟動速度最佳）
    """
    print(f"📷 正在啟動 USB/內建鏡頭 (Device ID: {cam_index})...")
    cap = None
    if sys.platform.startswith("win"):
        # Windows DirectShow 優先
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
    else:
        # Linux / macOS / Raspberry Pi
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[錯誤] 無法開啟編號 {cam_index} 的攝影機！請檢查連線或確認 --cam 參數。")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 鏡頭暖機嘗試讀取第 1 幀
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✅ 鏡頭串流初始化成功！")
            return cap
        time.sleep(0.05)

    print("⚠️ 鏡頭讀取逾時，但已完成連線。")
    return cap


def run_dance_trainer(
    video_path: str,
    pose_path: str = None,
    cam_index: int = 0,
    tolerance: float = 20.0,
    mirror_mode: bool = True,
):
    project_root = CURRENT_DIR.parent
    video_file = Path(video_path).resolve()

    # 自動推導 pose_json 路徑
    if pose_path:
        pose_file = Path(pose_path).resolve()
    else:
        pose_file = project_root / "data" / f"{video_file.stem}_pose.json"

    if not pose_file.exists():
        data_dir = project_root / "data"
        existing_jsons = list(data_dir.glob("*_pose.json"))
        if existing_jsons:
            pose_file = existing_jsons[0]
            print(f"[提示] 自動套用現存姿態檔：{pose_file.name}")
        else:
            print(f"[錯誤] 找不到姿態 JSON 檔案：{pose_file}")
            print(f"請先執行: python SOURCE/extract_pose.py --video {video_file}")
            sys.exit(1)

    # 載入離線標準骨架特徵
    ref_pose_map = load_pose_dataset(pose_file)

    # 開啟 Fancam 直拍影片
    cap_video = cv2.VideoCapture(str(video_file))
    if not cap_video.isOpened():
        print(f"[錯誤] 無法開啟參考影片：{video_file}")
        sys.exit(1)

    total_video_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap_video.get(cv2.CAP_PROP_FPS) or 30.0

    # 開啟使用者鏡頭
    cap_cam = open_camera(cam_index)

    # 初始化輕量級 UniversalPoseDetector (complexity=0 邊緣最佳化)
    pose_detector = UniversalPoseDetector(
        complexity=0,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("=" * 65)
    print("🚀 AI K-Pop Dance Trainer 啟動成功！")
    print("🎮 快捷鍵操作指引：")
    print("   [空白鍵 Space] : 暫停 / 定格 (可停格維持動作精細矯正)")
    print("   [S 鍵]         : 切換 0.5x 慢動作分解模式")
    print("   [M 鍵]         : 切換 鏡面練習模式 / 解剖學模式")
    print("   [R 鍵]         : 重頭開始 (Restart)")
    print("   [Q 鍵 / ESC]   : 退出系統")
    print("=" * 65)

    window_name = "AI K-Pop Dance Trainer (BA5077 Edge AIoT)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    is_paused = False
    is_slow_motion = False
    current_mirror_mode = mirror_mode

    video_frame_idx = 0
    slow_counter = 0

    fps_timer = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    last_video_frame = None
    cam_fail_count = 0

    try:
        while True:
            # ---------------------------------------------------------
            # 1. 處理左半邊：參考影片 (Fancam) 播放與骨架疊加
            # ---------------------------------------------------------
            advance_video = not is_paused
            if is_slow_motion and not is_paused:
                slow_counter += 1
                if slow_counter % 2 != 0:
                    advance_video = False

            if advance_video or last_video_frame is None:
                ret_v, frame_v = cap_video.read()
                if not ret_v:
                    # 影片播畢 -> 自動無縫循環重播 (Looping)
                    cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_frame_idx = 0
                    ret_v, frame_v = cap_video.read()
                    if not ret_v:
                        print("[警告] 影片重新讀取失敗！")
                        break

                last_video_frame = frame_v
                video_frame_idx = int(cap_video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                frame_v = last_video_frame

            # 標準化左側畫面為 480x640 置中填充
            left_canvas, _, _, _ = resize_and_pad(frame_v, 480, 640)

            # 直接自 JSON 檢索該幀標準數據 (零 AI 重複運算！)
            ref_data = ref_pose_map.get(video_frame_idx, None)
            ref_landmarks = ref_data.get("landmarks", []) if ref_data else []
            ref_angles = ref_data.get("angles", {}) if ref_data else {}

            # 在左視窗繪製標準金黃色骨架
            if ref_landmarks:
                draw_skeleton(left_canvas, ref_landmarks, color=(0, 215, 255), thickness=2)

            # 左側視窗 HUD
            v_status = "PAUSED" if is_paused else ("SLOW 0.5x" if is_slow_motion else "PLAYING")
            draw_hud(
                left_canvas,
                title="[1] FANCAM REFERENCE",
                sync_percent=None,
                fps=video_fps if not is_slow_motion else (video_fps * 0.5),
                status_tag=v_status,
                mirror_mode=current_mirror_mode,
                frame_info=f"Frame: {video_frame_idx}/{total_video_frames} | Song: {video_file.stem}",
            )

            # ---------------------------------------------------------
            # 2. 處理右半邊：USB 鏡頭捕捉、水平鏡像與輕量即時推論
            # ---------------------------------------------------------
            ret_c, frame_c = cap_cam.read()
            if not ret_c or frame_c is None:
                cam_fail_count += 1
                if cam_fail_count > 30:
                    print("[錯誤] USB 鏡頭訊號中斷！")
                    break
                time.sleep(0.03)
                continue
            cam_fail_count = 0

            # 水平鏡像翻轉
            frame_c = cv2.flip(frame_c, 1)

            # 標準化右側畫面為 480x640 置中填充
            right_canvas, _, _, _ = resize_and_pad(frame_c, 480, 640)

            # 姿態推論
            user_landmarks = pose_detector.process(right_canvas)

            sync_percent = 0.0
            comparison_results = None

            if user_landmarks:
                # 即時計算使用者 6 大關節幾何夾角
                user_angles = extract_6_joint_angles(user_landmarks)

                # 與參考骨架進行幾何容許值比對 (容許誤差 ±20 度)
                if ref_angles:
                    comparison_results = compare_pose_angles(
                        user_angles=user_angles,
                        ref_angles=ref_angles,
                        tolerance=tolerance,
                        mirror_mode=current_mirror_mode,
                    )
                    sync_percent = comparison_results["sync_percent"]

                # 繪製使用者反饋骨架 (動態切換色彩：綠色合格，紅色警示)
                draw_skeleton(
                    right_canvas,
                    user_landmarks,
                    color=(0, 255, 0),
                    joint_results=comparison_results,
                    thickness=3,
                )
            else:
                cv2.putText(
                    right_canvas,
                    "Step back to full body view",
                    (60, 320),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 右側視窗 HUD
            draw_hud(
                right_canvas,
                title="[2] YOUR DANCE (REAL-TIME)",
                sync_percent=sync_percent if user_landmarks else 0.0,
                fps=current_fps,
                status_tag="ACTIVE",
                mirror_mode=current_mirror_mode,
                joint_results=comparison_results,
            )

            # ---------------------------------------------------------
            # 3. 雙視窗左右拼接 (np.hstack: 480x640 + 480x640 = 960x640)
            # ---------------------------------------------------------
            combined_display = np.hstack([left_canvas, right_canvas])
            cv2.line(combined_display, (480, 0), (480, 640), (60, 60, 60), 2)

            cv2.imshow(window_name, combined_display)

            # ---------------------------------------------------------
            # 4. FPS 計算與鍵盤互動事件監聽
            # ---------------------------------------------------------
            fps_frame_count += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_timer)
                fps_frame_count = 0
                fps_timer = time.time()

            # 設定每幀 1ms 監聽
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                is_paused = not is_paused
                print(f"[狀態] 暫停定格: {'已暫停' if is_paused else '繼續播放'}")

            elif key == ord('s') or key == ord('S'):
                is_slow_motion = not is_slow_motion
                print(f"[狀態] 慢動作 0.5x 分解: {'開啟' if is_slow_motion else '關閉'}")

            elif key == ord('m') or key == ord('M'):
                current_mirror_mode = not current_mirror_mode
                mode_name = "鏡面練習模式 (視覺對齊)" if current_mirror_mode else "解剖學模式 (嚴格左右本體)"
                print(f"[狀態] 切換模式為: {mode_name}")

            elif key == ord('r') or key == ord('R'):
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_frame_idx = 0
                print("[狀態] 影片重置，從第 0 幀重新開始。")

            elif key == ord('q') or key == ord('Q') or key == 27:
                print("\n👋 練舞系統安全結束，感謝使用！")
                break

    finally:
        cap_video.release()
        cap_cam.release()
        pose_detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="AI K-Pop Dance Trainer - 韓團舞蹈動作即時比對系統 (BA5077)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="輸入韓團直拍 .mp4 檔案 (預設自動搜尋 videos/fancam.mp4)",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="預處理骨架 .json 檔案 (預設自動對應 data/<video_stem>_pose.json)",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="外接 USB/內建攝影機索引編號 (預設為 0)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="關節角度容許誤差度數 (預設 ±20.0 度)",
    )
    parser.add_argument(
        "--anatomical",
        action="store_true",
        help="啟用嚴格解剖學模式 (預設為鏡像練習模式)",
    )

    args = parser.parse_args()

    project_root = CURRENT_DIR.parent
    videos_dir = project_root / "videos"

    if args.video:
        target_video = args.video
    else:
        default_video = videos_dir / "fancam.mp4"
        if default_video.exists():
            target_video = str(default_video)
        else:
            mp4_candidates = list(videos_dir.glob("*.mp4"))
            if mp4_candidates:
                target_video = str(mp4_candidates[0])
            else:
                target_video = str(default_video)

    run_dance_trainer(
        video_path=target_video,
        pose_path=args.pose,
        cam_index=args.cam,
        tolerance=args.tolerance,
        mirror_mode=not args.anatomical,
    )


if __name__ == "__main__":
    main()
