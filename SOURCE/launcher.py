"""
launcher.py
-----------
AI 韓團舞蹈動作分解與即時同步矯正系統 - 圖形介面一鍵啟動器
課程：BA5077 人工智慧物聯網 (AIoT)
學生：張瑜庭 (學號：113401532)

功能特色：
1. 基於 Python 內建 tkinter 打造簡潔美觀的暗色系 GUI (零額外安裝套件)
2. 「一鍵選擇影片並自動比對」：支援本機任一 .mp4 影片匯入
3. 自動檢查 data/ 骨架 JSON：
   - 若尚未提取骨架：自動於背景線程執行 extract_pose.py，並顯示進度條與狀態
   - 若已有骨架特徵：自動隱藏啟動器，無縫啟動 dance_trainer.py 進入雙視窗即時練舞
4. 支援快速載入預設影片 (videos/fancam.mp4)、攝影機編號設定與鏡像練習切換
"""

import os
import sys
import shutil
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 專案路徑常數
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# 確保目錄存在
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class DanceTrainerLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("AI K-Pop Dance Trainer (BA5077 Edge AIoT)")
        self.geometry("640x520")
        self.minsize(580, 480)
        self.configure(bg="#181824")

        # 狀態變數
        self.selected_video_path = None
        self.is_processing = False

        # 置中視窗
        self.center_window(640, 520)

        # 建立 UI 介面
        self.setup_styles()
        self.create_widgets()

        # 自動檢查預設影片
        self.check_default_fancam()

    def center_window(self, width, height):
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # 設定色彩主題
        self.style.configure(".", background="#181824", foreground="#ffffff")
        self.style.configure(
            "TLabel",
            background="#181824",
            foreground="#ffffff",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Header.TLabel",
            background="#181824",
            foreground="#38bdf8",
            font=("Segoe UI", 16, "bold"),
        )
        self.style.configure(
            "SubHeader.TLabel",
            background="#181824",
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        self.style.configure(
            "Card.TFrame",
            background="#232334",
            relief="flat",
        )
        self.style.configure(
            "Card.TLabel",
            background="#232334",
            foreground="#e2e8f0",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Status.TLabel",
            background="#232334",
            foreground="#38bdf8",
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#1e1e2e",
            background="#38bdf8",
            lightcolor="#38bdf8",
            darkcolor="#0284c7",
            bordercolor="#1e1e2e",
        )

    def create_widgets(self):
        # 頂部 Header
        header_frame = tk.Frame(self, bg="#181824", pady=16)
        header_frame.pack(fill="x", padx=24)

        title_lbl = ttk.Label(
            header_frame,
            text="🎵 AI 韓團舞蹈動作分解與即時同步矯正系統",
            style="Header.TLabel",
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ttk.Label(
            header_frame,
            text="AI K-Pop Dance Trainer | BA5077 人工智慧物聯網 (AIoT) | 學生：張瑜庭 (113401532)",
            style="SubHeader.TLabel",
        )
        sub_lbl.pack(anchor="w", pady=(4, 0))

        # 中間資訊卡片容器
        card_frame = ttk.Frame(self, style="Card.TFrame", padding=18)
        card_frame.pack(fill="both", expand=True, padx=24, pady=8)

        # 影片選取狀態顯示
        self.video_info_lbl = ttk.Label(
            card_frame,
            text="📁 當前舞曲：尚未選取影片",
            style="Card.TLabel",
            font=("Segoe UI", 11, "bold"),
        )
        self.video_info_lbl.pack(anchor="w", pady=(0, 6))

        self.pose_status_lbl = ttk.Label(
            card_frame,
            text="📊 骨架狀態：等待選取影片",
            style="Card.TLabel",
            foreground="#94a3b8",
        )
        self.pose_status_lbl.pack(anchor="w", pady=(0, 16))

        # 核心功能按鈕區
        btn_frame = tk.Frame(card_frame, bg="#232334")
        btn_frame.pack(fill="x", pady=6)

        # 醒目的「選擇舞蹈影片並開始」主按鈕
        self.btn_select_video = tk.Button(
            btn_frame,
            text="📂 選擇舞蹈影片 (.mp4) 並開始練舞",
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=10,
            command=self.on_select_video_clicked,
        )
        self.btn_select_video.pack(fill="x", pady=(0, 10))

        # 快速載入預設影片按鈕
        self.btn_quick_start = tk.Button(
            btn_frame,
            text="⚡ 快速載入預設直拍 (videos/fancam.mp4)",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=7,
            command=self.on_quick_start_clicked,
        )
        self.btn_quick_start.pack(fill="x")

        # 參數微調設定區 (攝影機編號、鏡像模式、容許角度)
        settings_frame = tk.Frame(card_frame, bg="#232334")
        settings_frame.pack(fill="x", pady=(18, 8))

        # 攝影機選擇
        cam_lbl = tk.Label(settings_frame, text="攝影機 ID:", bg="#232334", fg="#cbd5e1", font=("Segoe UI", 9))
        cam_lbl.pack(side="left", padx=(0, 4))
        self.cam_var = tk.StringVar(value="0")
        cam_spin = ttk.Spinbox(settings_frame, from_=0, to=4, textvariable=self.cam_var, width=3)
        cam_spin.pack(side="left", padx=(0, 16))

        # 鏡像練習模式勾選 (預設 True)
        self.mirror_var = tk.BooleanVar(value=True)
        mirror_chk = tk.Checkbutton(
            settings_frame,
            text="鏡面練習模式 (如照鏡子對齊)",
            variable=self.mirror_var,
            bg="#232334",
            fg="#cbd5e1",
            activebackground="#232334",
            activeforeground="#38bdf8",
            selectcolor="#181824",
            font=("Segoe UI", 9),
        )
        mirror_chk.pack(side="left", padx=(0, 16))

        # 容許誤差角度
        tol_lbl = tk.Label(settings_frame, text="容許誤差:", bg="#232334", fg="#cbd5e1", font=("Segoe UI", 9))
        tol_lbl.pack(side="left", padx=(0, 4))
        self.tol_var = tk.StringVar(value="20")
        tol_combo = ttk.Combobox(settings_frame, values=["15", "20", "25", "30"], textvariable=self.tol_var, width=4, state="readonly")
        tol_combo.pack(side="left")
        deg_lbl = tk.Label(settings_frame, text="度", bg="#232334", fg="#cbd5e1", font=("Segoe UI", 9))
        deg_lbl.pack(side="left")

        # 底部進度條與動態狀態列
        status_box = tk.Frame(self, bg="#181824", pady=10)
        status_box.pack(fill="x", padx=24, side="bottom")

        self.progress_bar = ttk.Progressbar(status_box, style="Horizontal.TProgressbar", mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(0, 6))

        self.status_lbl = ttk.Label(
            status_box,
            text="💡 請點擊上方按鈕選擇 .mp4 舞曲，系統將自動啟動分析與雙視窗比對！",
            style="TLabel",
            foreground="#94a3b8",
        )
        self.status_lbl.pack(anchor="w")

    def check_default_fancam(self):
        """檢查專案預設的 videos/fancam.mp4 是否存在"""
        default_video = VIDEOS_DIR / "fancam.mp4"
        if default_video.exists():
            self.selected_video_path = default_video
            pose_json = DATA_DIR / "fancam_pose.json"
            status_str = "✅ 骨架特徵已就緒" if pose_json.exists() else "⚠️ 尚未提取骨架特徵 (啟動時自動分析)"
            self.video_info_lbl.config(text=f"📁 當前舞曲：{default_video.name}")
            self.pose_status_lbl.config(
                text=f"📊 骨架狀態：{status_str}",
                foreground="#34d399" if pose_json.exists() else "#fbbf24",
            )

    def on_quick_start_clicked(self):
        """點擊快速載入預設影片"""
        default_video = VIDEOS_DIR / "fancam.mp4"
        if not default_video.exists():
            messagebox.showwarning(
                "找不到預設影片",
                "在 videos/ 目錄下未找到 fancam.mp4！\n請點選「選擇舞蹈影片」載入您的自訂舞曲。",
            )
            return
        self.process_and_launch(default_video)

    def on_select_video_clicked(self):
        """開啟檔案選擇對話框選取本地 .mp4 影片"""
        file_path = filedialog.askopenfilename(
            title="選擇韓團直拍舞蹈影片 (.mp4)",
            initialdir=str(VIDEOS_DIR),
            filetypes=[("MP4 舞蹈影片", "*.mp4"), ("所有檔案", "*.*")],
        )
        if not file_path:
            return  # 使用者取消選擇

        video_path = Path(file_path).resolve()

        # 若使用者選取的檔案不在 videos 目錄內，自動複製一份至 videos 目錄保持專案結構完整
        if video_path.parent != VIDEOS_DIR:
            dest_video = VIDEOS_DIR / video_path.name
            try:
                shutil.copy2(str(video_path), str(dest_video))
                video_path = dest_video
            except Exception as e:
                print(f"[提示] 無法複製影片至 videos/，直接使用原路徑：{e}")

        self.process_and_launch(video_path)

    def process_and_launch(self, video_path: Path):
        """核心自動化流程：檢查 JSON -> 必要時背景執行提取 -> 無縫啟動練舞系統"""
        if self.is_processing:
            return

        self.selected_video_path = video_path
        video_stem = video_path.stem
        pose_json_path = DATA_DIR / f"{video_stem}_pose.json"

        self.video_info_lbl.config(text=f"📁 當前舞曲：{video_path.name}")

        # 檢查是否已有對應的骨架數據
        if pose_json_path.exists() and pose_json_path.stat().st_size > 1000:
            self.pose_status_lbl.config(text="📊 骨架狀態：✅ 特徵檔已就緒，即將啟動比對系統...", foreground="#34d399")
            self.status_lbl.config(text="🚀 正在啟動雙視窗練舞矯正系統...", foreground="#38bdf8")
            self.after(300, lambda: self.launch_dance_trainer(video_path, pose_json_path))
        else:
            # 尚未提取骨架：啟動背景線程執行 extract_pose.py
            self.is_processing = True
            self.btn_select_video.config(state="disabled", bg="#475569")
            self.btn_quick_start.config(state="disabled", bg="#475569")
            self.pose_status_lbl.config(text="📊 骨架狀態：⏳ AI 正在分析舞步骨架，請稍候...", foreground="#fbbf24")
            self.status_lbl.config(
                text="⏳ 正在執行 MediaPipe 高精度骨架特徵與 6 大關節角度運算...",
                foreground="#fbbf24",
            )
            self.progress_bar.start(12)

            thread = threading.Thread(
                target=self.run_pose_extraction_worker,
                args=(video_path, pose_json_path),
                daemon=True,
            )
            thread.start()

    def run_pose_extraction_worker(self, video_path: Path, output_json: Path):
        """背景線程：執行離線預處理腳本"""
        python_exe = sys.executable
        extract_script = CURRENT_DIR / "extract_pose.py"

        cmd = [
            python_exe,
            str(extract_script),
            "--video",
            str(video_path),
            "--output",
            str(output_json),
        ]

        try:
            # 執行預處理
            process = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if process.returncode == 0 and output_json.exists():
                self.after(0, lambda: self.on_extraction_completed(video_path, output_json))
            else:
                err_msg = process.stderr or process.stdout or "未知錯誤"
                self.after(0, lambda: self.on_extraction_failed(err_msg))

        except Exception as e:
            self.after(0, lambda: self.on_extraction_failed(str(e)))

    def on_extraction_completed(self, video_path: Path, output_json: Path):
        """骨架提取完成回調"""
        self.progress_bar.stop()
        self.is_processing = False
        self.btn_select_video.config(state="normal", bg="#2563eb")
        self.btn_quick_start.config(state="normal", bg="#059669")

        self.pose_status_lbl.config(text="📊 骨架狀態：✅ 特徵分析完成！", foreground="#34d399")
        self.status_lbl.config(text="🚀 正在啟動即時雙視窗練舞比對系統...", foreground="#38bdf8")

        # 延遲 500ms 後開啟練舞主程式
        self.after(500, lambda: self.launch_dance_trainer(video_path, output_json))

    def on_extraction_failed(self, error_message: str):
        """骨架提取失敗處理"""
        self.progress_bar.stop()
        self.is_processing = False
        self.btn_select_video.config(state="normal", bg="#2563eb")
        self.btn_quick_start.config(state="normal", bg="#059669")

        self.pose_status_lbl.config(text="📊 骨架狀態：❌ 分析失敗", foreground="#f87171")
        self.status_lbl.config(text="❌ 骨架特徵分析發生錯誤，請檢查影片檔案。", foreground="#f87171")

        messagebox.showerror(
            "姿態分析失敗",
            f"無法自影片萃取骨架特徵：\n{error_message[:400]}",
        )

    def launch_dance_trainer(self, video_path: Path, pose_json: Path):
        """無縫啟動 dance_trainer.py 進入雙視窗即時練舞"""
        # 暫時隱藏啟動器視窗，讓使用者的視覺焦點集中在練舞雙視窗
        self.withdraw()

        python_exe = sys.executable
        trainer_script = CURRENT_DIR / "dance_trainer.py"

        cam_idx = self.cam_var.get().strip() or "0"
        tol_val = self.tol_var.get().strip() or "20"

        cmd = [
            python_exe,
            str(trainer_script),
            "--video",
            str(video_path),
            "--pose",
            str(pose_json),
            "--cam",
            cam_idx,
            "--tolerance",
            tol_val,
        ]

        if not self.mirror_var.get():
            cmd.append("--anatomical")

        # 在新執行緒中監控主程式執行，當練舞結束 (按 Q/ESC) 後重新顯示啟動器
        def monitor_trainer():
            try:
                subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            finally:
                # 主程式結束後恢復啟動器視窗
                self.after(0, self.on_trainer_exited)

        thread = threading.Thread(target=monitor_trainer, daemon=True)
        thread.start()

    def on_trainer_exited(self):
        """練舞主程式關閉後恢復啟動器介面"""
        self.deiconify()
        self.status_lbl.config(
            text="✨ 練舞結束！您可以再次選擇不同舞曲或使用預設直拍繼續練習。",
            foreground="#38bdf8",
        )


def main():
    app = DanceTrainerLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
