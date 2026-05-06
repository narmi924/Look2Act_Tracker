"""摄像头流模块。

基于 OpenCV 和 Qt 线程，在后台持续读取摄像头帧并通过信号发送到主线程。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.ui.i18n import tx


@dataclass(frozen=True)
class Resolution:
    """分辨率数据结构。"""
    w: int
    h: int

    def label(self) -> str:
        """返回可读的分辨率标签，例如 "1920x1080"。"""
        return f"{self.w}x{self.h}"


class _CaptureWorker(QObject):
    """摄像头读取工作线程（内部类）。
    
    在后台线程中持续读取摄像头帧，通过 Qt 信号发送到主线程。
    
    Signals:
        frame_received: 发送 BGR 图像（numpy array，uint8）。
        error: 发送错误消息（str）。
    """
    frame_received = pyqtSignal(object)  # np.ndarray(BGR)
    error = pyqtSignal(str)

    def __init__(self, camera_index: int, resolution: Resolution, fps_limit: int = 30):
        super().__init__()
        self._camera_index = camera_index
        self._resolution = resolution
        self._fps_limit = max(1, int(fps_limit))
        self._running = False

    def stop(self) -> None:
        """停止读取。"""
        self._running = False

    def run(self) -> None:
        """工作线程主循环：持续读取帧并发送信号。"""
        self._running = True

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            msg = tx(f"摄像头打开失败：index={self._camera_index}", f"Failed to open camera: index={self._camera_index}")
            print(f"[CAMERA] {msg}")
            try:
                self.error.emit(msg)
            except RuntimeError as e:
                print(f"[CAMERA] error signal emit failed: {e!r}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._resolution.w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._resolution.h))
        cap.set(cv2.CAP_PROP_FPS, float(self._fps_limit))

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                err = tx("读取帧失败", "Failed to read frame")
                print(f"[CAMERA] {err}")
                try:
                    self.error.emit(err)
                except RuntimeError as e:
                    print(f"[CAMERA] error signal emit failed: {e!r}")
                break
            
            try:
                self.frame_received.emit(frame)
            except RuntimeError as e:
                print(f"[CAMERA] frame_received emit failed: {e!r}")
                break

            # 帧率限制
            cv2.waitKey(int(1000 / self._fps_limit))

        cap.release()


class CameraStream(QObject):
    """摄像头流管理器。
    
    封装摄像头读取逻辑，提供 start/stop 接口。
    使用后台线程持续读取帧，通过 Qt 信号发送到主线程。
    
    Signals:
        frame_received: 发送 BGR 图像（numpy array，uint8）。
        error: 发送错误消息（str）。
    """
    frame_received = pyqtSignal(object)  # np.ndarray(BGR)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._thread: Optional[QThread] = None
        self._worker: Optional[_CaptureWorker] = None

    def is_running(self) -> bool:
        """检查摄像头流是否正在运行。"""
        return self._thread is not None and self._thread.isRunning()

    def start(self, camera_index: int, resolution: Resolution, fps_limit: int = 30) -> None:
        """启动摄像头流。
        
        Args:
            camera_index: 摄像头索引（通常 0 为默认摄像头）。
            resolution: 目标分辨率。
            fps_limit: 帧率限制（fps），默认 30。
        """
        self.stop()

        thread = QThread()
        worker = _CaptureWorker(camera_index=camera_index, resolution=resolution, fps_limit=fps_limit)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)  # type: ignore[arg-type]
        worker.frame_received.connect(self.frame_received)  # type: ignore[arg-type]
        worker.error.connect(self.error)  # type: ignore[arg-type]
        worker.error.connect(lambda _msg: self.stop())

        self._thread = thread
        self._worker = worker

        thread.start()

    def stop(self) -> None:
        """停止摄像头流。"""
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(200):
                print("[CAMERA] thread did not stop in 200ms, calling terminate()")
                self._thread.terminate()
                self._thread.wait()
        self._worker = None
        self._thread = None
