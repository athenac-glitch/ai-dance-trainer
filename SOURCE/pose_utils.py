"""
pose_utils.py
-------------
幾何學角度計算、骨架可視化與 HUD 繪製模組
課程：BA5077 人工智慧物聯網 (AIoT)
"""

import os
import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Pose 33 關鍵點索引對照
class PoseLandmarkIndex:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# 骨架連線定義 (肢體分群)
LIMB_CONNECTIONS = {
    "torso": [
        (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.RIGHT_SHOULDER),
        (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.LEFT_HIP),
        (PoseLandmarkIndex.RIGHT_SHOULDER, PoseLandmarkIndex.RIGHT_HIP),
        (PoseLandmarkIndex.LEFT_HIP, PoseLandmarkIndex.RIGHT_HIP),
    ],
    "head": [
        (PoseLandmarkIndex.NOSE, PoseLandmarkIndex.LEFT_EYE),
        (PoseLandmarkIndex.NOSE, PoseLandmarkIndex.RIGHT_EYE),
        (PoseLandmarkIndex.LEFT_EYE, PoseLandmarkIndex.LEFT_EAR),
        (PoseLandmarkIndex.RIGHT_EYE, PoseLandmarkIndex.RIGHT_EAR),
    ],
    "left_arm": [
        (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.LEFT_ELBOW),
        (PoseLandmarkIndex.LEFT_ELBOW, PoseLandmarkIndex.LEFT_WRIST),
    ],
    "right_arm": [
        (PoseLandmarkIndex.RIGHT_SHOULDER, PoseLandmarkIndex.RIGHT_ELBOW),
        (PoseLandmarkIndex.RIGHT_ELBOW, PoseLandmarkIndex.RIGHT_WRIST),
    ],
    "left_leg": [
        (PoseLandmarkIndex.LEFT_HIP, PoseLandmarkIndex.LEFT_KNEE),
        (PoseLandmarkIndex.LEFT_KNEE, PoseLandmarkIndex.LEFT_ANKLE),
    ],
    "right_leg": [
        (PoseLandmarkIndex.RIGHT_HIP, PoseLandmarkIndex.RIGHT_KNEE),
        (PoseLandmarkIndex.RIGHT_KNEE, PoseLandmarkIndex.RIGHT_ANKLE),
    ],
}


class UniversalPoseDetector:
    """
    通用 MediaPipe Pose 偵測器封裝。
    自動適配：
    1. Legacy mp.solutions.pose (MediaPipe <= 0.10.x)
    2. Modern mp.tasks.python.vision.PoseLandmarker (MediaPipe 0.10+ / 1.x)
    支援 Raspberry Pi、Linux、Windows、macOS 邊緣跨平台運作。
    """

    MODEL_URLS = {
        0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    }

    def __init__(
        self,
        complexity: int = 2,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.complexity = complexity
        self.mode = None
        self.detector = None

        # 1. 優先檢測是否有 legacy solutions 介面
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mode = "solutions"
            self.detector = mp.solutions.pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # 2. 自動採用 Google 現代 MediaPipe Tasks API
            self.mode = "tasks"
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            model_filenames = {
                0: "pose_landmarker_lite.task",
                1: "pose_landmarker_full.task",
                2: "pose_landmarker_heavy.task",
            }
            target_model_name = model_filenames.get(complexity, "pose_landmarker_heavy.task")
            models_dir = Path(__file__).resolve().parent.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / target_model_name

            # 若模型檔案不存在，自動下載或降級使用現存模型
            if not model_path.exists() or model_path.stat().st_size < 1000:
                lite_path = models_dir / "pose_landmarker_lite.task"
                if lite_path.exists() and lite_path.stat().st_size > 1000:
                    model_path = lite_path
                else:
                    url = self.MODEL_URLS.get(complexity, self.MODEL_URLS[0])
                    print(f"📥 正在下載 MediaPipe 姿態模型至 {model_path.name}...")
                    try:
                        urllib.request.urlretrieve(url, str(model_path))
                    except Exception as e:
                        print(f"[警告] 模型下載失敗，嘗試使用 lite 輕量模型: {e}")
                        model_path = lite_path
                        urllib.request.urlretrieve(self.MODEL_URLS[0], str(model_path))

            options = vision.PoseLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                num_poses=1,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray):
        """
        輸入 BGR 影像，回傳標準化 33 個關鍵點清單 [[x, y, z, visibility], ...]
        若未偵測到人體，回傳 None。
        """
        if self.mode == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([
                        float(lm.x),
                        float(lm.y),
                        float(lm.z),
                        float(getattr(lm, "visibility", 1.0)),
                    ])
                return landmarks
            return None
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.detector.detect(mp_img)
            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                first_pose = res.pose_landmarks[0]
                landmarks = []
                for lm in first_pose:
                    landmarks.append([
                        float(lm.x),
                        float(lm.y),
                        float(lm.z),
                        float(getattr(lm, "visibility", 1.0)),
                    ])
                return landmarks
            return None

    def close(self):
        if hasattr(self, "detector") and hasattr(self.detector, "close"):
            self.detector.close()


def calculate_angle_2d(p_a, p_b, p_c) -> float:
    """
    計算以關節點 p_b 為頂點，由向量 BA 與向量 BC 所夾之幾何內角 (0 ~ 180 度)。
    p_a, p_b, p_c 可為 (x, y) 座標或 [x, y, ...] 序列。
    """
    a = np.array([p_a[0], p_a[1]], dtype=np.float32)
    b = np.array([p_b[0], p_b[1]], dtype=np.float32)
    c = np.array([p_c[0], p_c[1]], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 避免除以零異常
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # 限制在 [-1.0, 1.0] 避免浮點數溢位造成 arccos 錯誤
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)


def extract_6_joint_angles(landmarks) -> dict:
    """
    自 33 個關鍵點座標清單計算 6 大核心關節幾何夾角：
    1. left_elbow: 左手肘 (左肩 - 左肘 - 左腕)
    2. right_elbow: 右手肘 (右肩 - 右肘 - 右腕)
    3. left_shoulder: 左肩抬起夾角 (左髖 - 左肩 - 左肘)
    4. right_shoulder: 右肩抬起夾角 (右髖 - 右肩 - 右肘)
    5. left_knee: 左膝蓋彎曲度 (左髖 - 左膝 - 左踝)
    6. right_knee: 右膝蓋彎曲度 (右髖 - 右膝 - 右踝)
    """
    pts = []
    for lm in landmarks:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            pts.append((lm.x, lm.y))
        elif isinstance(lm, (list, tuple)):
            pts.append((lm[0], lm[1]))
        elif isinstance(lm, dict):
            pts.append((lm.get("x", 0.0), lm.get("y", 0.0)))
        else:
            pts.append((0.0, 0.0))

    if len(pts) < 33:
        return {
            "left_elbow": 0.0,
            "right_elbow": 0.0,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
        }

    idx = PoseLandmarkIndex
    angles = {
        # 手肘角度：肩 - 肘 - 腕
        "left_elbow": calculate_angle_2d(pts[idx.LEFT_SHOULDER], pts[idx.LEFT_ELBOW], pts[idx.LEFT_WRIST]),
        "right_elbow": calculate_angle_2d(pts[idx.RIGHT_SHOULDER], pts[idx.RIGHT_ELBOW], pts[idx.RIGHT_WRIST]),
        # 肩膀抬起角度：髖 - 肩 - 肘
        "left_shoulder": calculate_angle_2d(pts[idx.LEFT_HIP], pts[idx.LEFT_SHOULDER], pts[idx.LEFT_ELBOW]),
        "right_shoulder": calculate_angle_2d(pts[idx.RIGHT_HIP], pts[idx.RIGHT_SHOULDER], pts[idx.RIGHT_ELBOW]),
        # 膝蓋彎曲角度：髖 - 膝 - 踝
        "left_knee": calculate_angle_2d(pts[idx.LEFT_HIP], pts[idx.LEFT_KNEE], pts[idx.LEFT_ANKLE]),
        "right_knee": calculate_angle_2d(pts[idx.RIGHT_HIP], pts[idx.RIGHT_KNEE], pts[idx.RIGHT_ANKLE]),
    }

    # 四捨五入至小數點後 1 位
    return {k: round(v, 1) for k, v in angles.items()}


def compare_pose_angles(user_angles: dict, ref_angles: dict, tolerance: float = 20.0, mirror_mode: bool = True) -> dict:
    """
    逐項比對使用者關節角度與參考影片標準角度。
    """
    joint_keys = [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_knee",
        "right_knee",
    ]

    mapping = {k: k for k in joint_keys}
    if mirror_mode:
        # 鏡面練習模式：左臂對右臂、左肩對右肩、左膝對右膝
        mapping = {
            "left_elbow": "right_elbow",
            "right_elbow": "left_elbow",
            "left_shoulder": "right_shoulder",
            "right_shoulder": "left_shoulder",
            "left_knee": "right_knee",
            "right_knee": "left_knee",
        }

    details = {}
    passed_count = 0
    scores = []

    for u_key in joint_keys:
        r_key = mapping[u_key]
        u_val = user_angles.get(u_key, 0.0)
        r_val = ref_angles.get(r_key, 0.0)

        diff = abs(u_val - r_val)
        is_pass = diff <= tolerance

        if is_pass:
            passed_count += 1
            score = max(85.0, 100.0 - (diff / tolerance) * 15.0)
        else:
            excess = diff - tolerance
            score = max(0.0, 85.0 - excess * 2.5)

        scores.append(score)
        details[u_key] = {
            "user_val": u_val,
            "ref_val": r_val,
            "diff": round(diff, 1),
            "is_pass": is_pass,
            "score": round(score, 1),
        }

    sync_percent = round(float(np.mean(scores)), 1) if scores else 0.0

    return {
        "sync_percent": sync_percent,
        "passed_count": passed_count,
        "total_joints": len(joint_keys),
        "details": details,
    }


def resize_and_pad(image: np.ndarray, target_w: int = 480, target_h: int = 640) -> tuple:
    """
    等比例縮放圖片並置中填充黑邊至 (target_w, target_h)，避免舞者身型扭曲失真。
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def draw_skeleton(
    image: np.ndarray,
    landmarks,
    color=(0, 255, 255),
    joint_results: dict = None,
    thickness: int = 2,
    radius: int = 4,
):
    """
    在影像上繪製骨架連線與關節點。
    """
    h, w = image.shape[:2]

    pts_px = {}
    for idx, lm in enumerate(landmarks):
        if hasattr(lm, "x") and hasattr(lm, "y"):
            lx, ly = lm.x, lm.y
            vis = getattr(lm, "visibility", 1.0)
        elif isinstance(lm, (list, tuple)):
            lx, ly = lm[0], lm[1]
            vis = lm[3] if len(lm) > 3 else 1.0
        elif isinstance(lm, dict):
            lx, ly = lm.get("x", 0.0), lm.get("y", 0.0)
            vis = lm.get("visibility", 1.0)
        else:
            continue

        if vis > 0.3:
            px = int(np.clip(lx * w, 0, w - 1))
            py = int(np.clip(ly * h, 0, h - 1))
            pts_px[idx] = (px, py)

    COLOR_PASS = (0, 255, 0)
    COLOR_WARN = (0, 0, 255)
    COLOR_NEUTRAL = (240, 240, 240)

    limb_colors = {
        "torso": COLOR_NEUTRAL,
        "head": COLOR_NEUTRAL,
        "left_arm": color,
        "right_arm": color,
        "left_leg": color,
        "right_leg": color,
    }

    if joint_results and "details" in joint_results:
        det = joint_results["details"]
        l_arm_pass = det.get("left_elbow", {}).get("is_pass", False) and det.get("left_shoulder", {}).get("is_pass", False)
        limb_colors["left_arm"] = COLOR_PASS if l_arm_pass else COLOR_WARN

        r_arm_pass = det.get("right_elbow", {}).get("is_pass", False) and det.get("right_shoulder", {}).get("is_pass", False)
        limb_colors["right_arm"] = COLOR_PASS if r_arm_pass else COLOR_WARN

        l_leg_pass = det.get("left_knee", {}).get("is_pass", False)
        limb_colors["left_leg"] = COLOR_PASS if l_leg_pass else COLOR_WARN

        r_leg_pass = det.get("right_knee", {}).get("is_pass", False)
        limb_colors["right_leg"] = COLOR_PASS if r_leg_pass else COLOR_WARN

    for group_name, connections in LIMB_CONNECTIONS.items():
        c = limb_colors.get(group_name, color)
        for p1_idx, p2_idx in connections:
            if p1_idx in pts_px and p2_idx in pts_px:
                cv2.line(image, pts_px[p1_idx], pts_px[p2_idx], c, thickness, cv2.LINE_AA)

    for idx, pt in pts_px.items():
        cv2.circle(image, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, pt, radius + 1, (40, 40, 40), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    title: str,
    sync_percent: float = None,
    fps: float = 0.0,
    status_tag: str = "PLAYING",
    mirror_mode: bool = True,
    joint_results: dict = None,
    frame_info: str = "",
):
    """
    在畫面頂部與底部繪製高質感半透明 HUD 資訊板。
    """
    h, w = frame.shape[:2]

    # 頂部半透明橫幅
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    status_color = (0, 230, 100)
    if "PAUSE" in status_tag:
        status_color = (0, 200, 255)
    elif "SLOW" in status_tag:
        status_color = (255, 180, 0)

    cv2.putText(
        frame,
        f"[{status_tag}]",
        (w - 170, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        status_color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (w - 78, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if sync_percent is not None:
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - 90), (w, h), (18, 18, 22), -1)
        cv2.addWeighted(overlay_bottom, 0.75, frame, 0.25, 0, frame)

        if sync_percent >= 80.0:
            meter_color = (0, 255, 0)
            rating_text = "PERFECT!"
        elif sync_percent >= 60.0:
            meter_color = (0, 215, 255)
            rating_text = "GOOD"
        else:
            meter_color = (0, 80, 255)
            rating_text = "CORRECTING"

        sync_str = f"SYNC: {sync_percent:.1f}%"
        cv2.putText(
            frame,
            sync_str,
            (14, h - 48),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            meter_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            rating_text,
            (210, h - 48),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            meter_color,
            1,
            cv2.LINE_AA,
        )

        mode_str = "MIRROR (ON)" if mirror_mode else "ANATOMICAL"
        guide_str = f"Mode: {mode_str} | [Space] Pause  [S] 0.5x Slow  [M] Mode"
        cv2.putText(
            frame,
            guide_str,
            (14, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if joint_results and "details" in joint_results:
            det = joint_results["details"]
            y_offset = 75
            for k, label in [
                ("left_elbow", "L-Elbow"),
                ("right_elbow", "R-Elbow"),
                ("left_shoulder", "L-Shldr"),
                ("right_shoulder", "R-Shldr"),
                ("left_knee", "L-Knee"),
                ("right_knee", "R-Knee"),
            ]:
                item = det.get(k, {})
                diff = item.get("diff", 0.0)
                is_p = item.get("is_pass", False)
                dot_color = (0, 255, 0) if is_p else (0, 0, 255)

                cv2.circle(frame, (w - 110, y_offset), 4, dot_color, -1)
                cv2.putText(
                    frame,
                    f"{label}: {diff:.0f} deg",
                    (w - 98, y_offset + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 18

    elif frame_info:
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - 40), (w, h), (18, 18, 22), -1)
        cv2.addWeighted(overlay_bottom, 0.70, frame, 0.30, 0, frame)
        cv2.putText(
            frame,
            frame_info,
            (14, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
