"""
实时兼容性测试：PyQt6 UI + OpenCV 摄像头 + MediaPipe FaceMesh 持续推理
验证三者在同一进程/同一 Qt 事件循环下能否稳定共存 2 秒。

用法：
    conda activate gaze-env
    python Look2Act_Tracker_Project/tools/compat_test_realtime_cam_qt_mp.py
"""
import sys
import time
import traceback
import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, PYQT_VERSION_STR, Qt, QThread, pyqtSignal, pyqtSlot


# ============ 配置参数 ============
DURATION_SEC = 2.0          # 每种模式运行时长（秒）
TIMER_INTERVAL_MS = 33      # QTimer 间隔（约 30 FPS）
INFER_EVERY_N = 1           # 每 N 帧做一次 FaceMesh 推理
CAMERA_INDICES = [0, 1, 2]  # 尝试打开的摄像头索引


@dataclass
class RunStats:
    """单次运行的统计数据。"""
    mode: str = ""
    total_frames: int = 0
    infer_count: int = 0
    infer_times_ms: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def ui_fps(self) -> float:
        return self.total_frames / self.elapsed if self.elapsed > 0 else 0

    @property
    def infer_fps(self) -> float:
        return self.infer_count / self.elapsed if self.elapsed > 0 else 0

    @property
    def avg_infer_ms(self) -> float:
        return statistics.mean(self.infer_times_ms) if self.infer_times_ms else 0

    @property
    def median_infer_ms(self) -> float:
        return statistics.median(self.infer_times_ms) if self.infer_times_ms else 0

    def summary(self) -> str:
        lines = [
            f"--- [{self.mode}] 运行统计 ---",
            f"  运行时长:       {self.elapsed:.2f} 秒",
            f"  总帧数:         {self.total_frames}",
            f"  推理次数:       {self.infer_count}",
            f"  UI 帧率:        {self.ui_fps:.1f} FPS",
            f"  推理帧率:       {self.infer_fps:.1f} FPS",
            f"  平均推理耗时:   {self.avg_infer_ms:.2f} ms",
            f"  中位推理耗时:   {self.median_infer_ms:.2f} ms",
        ]
        return "\n".join(lines)


def open_camera() -> cv2.VideoCapture:
    """尝试打开摄像头，返回 VideoCapture 对象。"""
    for idx in CAMERA_INDICES:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[OK] 摄像头 index={idx} 已打开，分辨率: {w}x{h}")
            return cap
        cap.release()
    return None


def create_face_mesh():
    """创建 MediaPipe FaceMesh 实例。"""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def do_infer(face_mesh, frame_bgr) -> float:
    """执行一次 FaceMesh 推理，返回耗时（毫秒）。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    face_mesh.process(rgb)
    return (time.perf_counter() - t0) * 1000.0


def frame_to_qpixmap(frame_bgr) -> QPixmap:
    """将 OpenCV BGR 帧转换为 QPixmap。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ============ QThread Worker（线程模式用） ============
class InferWorker(QThread):
    """在独立线程中执行 FaceMesh 推理。"""
    result_ready = pyqtSignal(float)  # 推理耗时 ms

    def __init__(self):
        super().__init__()
        self._frame = None
        self._running = True
        self._has_frame = False
        self.face_mesh = create_face_mesh()

    def submit_frame(self, frame_bgr):
        self._frame = frame_bgr.copy()
        self._has_frame = True

    def stop(self):
        self._running = False
        self.face_mesh.close()

    def run(self):
        while self._running:
            if self._has_frame and self._frame is not None:
                self._has_frame = False
                elapsed_ms = do_infer(self.face_mesh, self._frame)
                self.result_ready.emit(elapsed_ms)
            else:
                self.msleep(5)


# ============ 主窗口 ============
class TestWindow(QMainWindow):
    """测试窗口：显示摄像头画面 + 执行 FaceMesh 推理。"""

    def __init__(self, cap: cv2.VideoCapture, use_thread: bool):
        super().__init__()
        self.cap = cap
        self.use_thread = use_thread
        self.stats = RunStats(mode="QThread" if use_thread else "主线程")

        # UI
        self.setWindowTitle(f"兼容性测试 - {self.stats.mode}")
        self.label = QLabel("等待摄像头...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.label)
        self.setCentralWidget(container)
        self.resize(640, 480)

        # FaceMesh（主线程模式）
        self.face_mesh = None
        if not use_thread:
            self.face_mesh = create_face_mesh()

        # Worker（线程模式）
        self.worker = None
        if use_thread:
            self.worker = InferWorker()
            self.worker.result_ready.connect(self._on_infer_done)
            self.worker.start()

        # 进度输出节流
        self._last_progress = 0.0

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(TIMER_INTERVAL_MS)
        self.stats.start_time = time.perf_counter()

    @pyqtSlot(float)
    def _on_infer_done(self, elapsed_ms: float):
        """线程模式：收到推理结果。"""
        self.stats.infer_count += 1
        self.stats.infer_times_ms.append(elapsed_ms)

    def _on_timer(self):
        """QTimer 回调：读帧、显示、推理。"""
        try:
            now = time.perf_counter()
            elapsed = now - self.stats.start_time

            # 到时间则停止
            if elapsed >= DURATION_SEC:
                self._finish()
                return

            # 读帧
            ret, frame = self.cap.read()
            if not ret:
                return

            self.stats.total_frames += 1

            # 显示到 QLabel
            pixmap = frame_to_qpixmap(frame)
            self.label.setPixmap(pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

            # 推理
            if self.stats.total_frames % INFER_EVERY_N == 0:
                if self.use_thread and self.worker:
                    self.worker.submit_frame(frame)
                elif self.face_mesh:
                    ms = do_infer(self.face_mesh, frame)
                    self.stats.infer_count += 1
                    self.stats.infer_times_ms.append(ms)

            # 每 0.5 秒输出一次进度
            if now - self._last_progress >= 0.5:
                self._last_progress = now
                print(f"  [{self.stats.mode}] {elapsed:.1f}s | "
                      f"帧={self.stats.total_frames} 推理={self.stats.infer_count}")

        except Exception:
            print(f"[FAIL] 定时器回调异常:")
            traceback.print_exc()
            self._finish(exit_code=1)

    def _finish(self, exit_code=0):
        """停止并输出统计。"""
        self.timer.stop()
        self.stats.end_time = time.perf_counter()

        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        if self.face_mesh:
            self.face_mesh.close()

        print(self.stats.summary())
        QApplication.exit(exit_code)


# ============ 入口 ============
def run_test(cap: cv2.VideoCapture, use_thread: bool, app: QApplication) -> int:
    """运行一次测试，返回退出码。"""
    win = TestWindow(cap, use_thread=use_thread)
    win.show()
    return app.exec()


def main():
    print("=" * 60)
    print("实时兼容性测试：PyQt6 + OpenCV 摄像头 + MediaPipe FaceMesh")
    print("=" * 60)
    print(f"Python:     {sys.executable}")
    print(f"PyQt6:      {PYQT_VERSION_STR}")
    print(f"mediapipe:  {mp.__version__}")
    print(f"OpenCV:     {cv2.__version__}")
    print(f"测试时长:   {DURATION_SEC} 秒/模式")
    print()

    # 打开摄像头
    cap = open_camera()
    if cap is None:
        print("[FAIL] 无法打开摄像头。请检查：")
        print("  1. 摄像头是否被其他程序占用")
        print("  2. Windows 隐私设置 > 摄像头 是否允许桌面应用访问")
        print("  3. 设备管理器中摄像头驱动是否正常")
        return 1

    all_ok = True

    # --- 模式 1：主线程推理 ---
    print("\n>>> 模式 1：主线程推理")
    app1 = QApplication(sys.argv)
    code1 = run_test(cap, use_thread=False, app=app1)
    if code1 != 0:
        print(f"[FAIL] 主线程模式退出码: {code1}")
        all_ok = False
    else:
        print("[OK] 主线程模式完成")
    del app1

    # 重新打开摄像头（QApplication 销毁后可能需要）
    cap.release()
    cap = open_camera()
    if cap is None:
        print("[WARN] 第二次打开摄像头失败，跳过线程模式测试")
        return 0 if all_ok else 1

    # --- 模式 2：QThread 推理 ---
    print("\n>>> 模式 2：QThread 推理")
    app2 = QApplication(sys.argv)
    code2 = run_test(cap, use_thread=True, app=app2)
    if code2 != 0:
        print(f"[FAIL] QThread 模式退出码: {code2}")
        all_ok = False
    else:
        print("[OK] QThread 模式完成")
    del app2

    # 清理
    cap.release()

    print("\n" + "=" * 60)
    if all_ok:
        print("结论：PyQt6 + MediaPipe FaceMesh 实时共存测试全部通过")
    else:
        print("结论：存在兼容性问题，请查看上方错误信息")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("[FATAL] 未捕获异常:")
        traceback.print_exc()
        sys.exit(1)
