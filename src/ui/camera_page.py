"""摄像头预览页面。

显示实时摄像头画面，叠加人脸检测框和眼部关键点可视化。
提供启动/停止预览的控制按钮，显示当前 FPS 信息。

主要功能：
1. 实时显示摄像头画面
2. 在画面上绘制人脸边界框
3. 在画面上绘制 68 个关键点
4. 显示 FPS 信息
5. 提供启动/停止控制按钮
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, CardWidget, PrimaryPushButton, PushButton

from src.ui.camera_stream import CameraStream, Resolution
from src.ui.fluent_theme import PALETTE
from src.ui.i18n import tx, tx_button
from src.vision.face_detector import FaceDetector


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """将 BGR numpy 图像转换为 QImage（用于 Qt 显示）。"""
    rgb = bgr[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


class CameraPage(QWidget):
    """摄像头预览页面。
    
    显示实时摄像头画面并叠加人脸检测框和眼部关键点。
    
    页面布局：
    - 顶部：标题和控制按钮
    - 中间：摄像头预览窗口（显示检测结果）
    - 底部：状态信息（FPS、检测状态）
    
    功能：
    - 启动/停止摄像头预览
    - 实时人脸检测和关键点可视化
    - FPS 显示
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # 摄像头流和人脸检测器
        self._stream = CameraStream()
        self._stream.frame_received.connect(self._on_frame)  # type: ignore[arg-type]
        self._stream.error.connect(self._on_error)  # type: ignore[arg-type]
        
        self._detector: Optional[FaceDetector] = None
        
        # FPS 计算
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        # 默认摄像头配置
        self._camera_index = 0
        self._resolution = Resolution(w=1280, h=720)
        
        self._init_ui()

    def _init_ui(self) -> None:
        """初始化 UI 布局。"""
        # 标题
        title = QLabel(tx("摄像头预览", "Camera Preview"))
        title.setStyleSheet("font-size: 32px; font-weight: 900;")
        
        # 控制按钮
        self.start_btn = PrimaryPushButton(tx_button("启动预览", "Start Preview"))
        self.start_btn.setFixedSize(160, 60)
        self.start_btn.clicked.connect(self._handle_start)  # type: ignore[arg-type]
        
        self.stop_btn = PushButton(tx_button("停止预览", "Stop Preview"))
        self.stop_btn.setFixedSize(160, 60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._handle_stop)  # type: ignore[arg-type]
        
        # 顶部栏
        top_bar = QHBoxLayout()
        top_bar.addWidget(title)
        top_bar.addStretch(1)
        top_bar.addWidget(self.start_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.stop_btn)
        
        # 预览窗口
        self.preview_label = QLabel(tx_button("点击启动预览开始", "Click Start Preview to begin"))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(960, 540)
        self.preview_label.setStyleSheet("background: #000; border-radius: 12px; color: #888; font-size: 18px;")
        
        preview_card = CardWidget()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.addWidget(self.preview_label)
        
        # 状态信息
        status_card = CardWidget()
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(20)
        
        # FPS 显示
        fps_label = BodyLabel("FPS:")
        fps_label.setStyleSheet("font-weight: 600;")
        self.fps_value = BodyLabel("0.0")
        self.fps_value.setStyleSheet(f"font-weight: 900; color: {PALETTE['accent']};")
        
        # 检测状态
        detect_label = BodyLabel(tx("人脸检测：", "Face Detection:"))
        detect_label.setStyleSheet("font-weight: 600;")
        self.detect_value = BodyLabel(tx("未启动", "Not Started"))
        
        # 错误信息
        self.error_label = BodyLabel("")
        self.error_label.setStyleSheet("color: #D32F2F; font-weight: 600;")
        
        status_layout.addWidget(fps_label)
        status_layout.addWidget(self.fps_value)
        status_layout.addSpacing(30)
        status_layout.addWidget(detect_label)
        status_layout.addWidget(self.detect_value)
        status_layout.addStretch(1)
        status_layout.addWidget(self.error_label)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(16)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(preview_card, stretch=1)
        main_layout.addWidget(status_card)

    def _handle_start(self) -> None:
        """启动摄像头预览。"""
        try:
            self.error_label.setText("")
            self.preview_label.setText(tx_button("正在启动摄像头...", "Starting camera..."))
            
            # 初始化人脸检测器
            if self._detector is None:
                self._detector = FaceDetector(eye_crop_size=128)
            
            # 重置 FPS 计数
            self._frame_count = 0
            self._last_fps_time = time.time()
            self._current_fps = 0.0
            
            # 启动摄像头流
            self._stream.start(
                camera_index=self._camera_index,
                resolution=self._resolution,
                fps_limit=30
            )
            
            # 更新按钮状态
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            print(f"[CAMERA_PAGE] 摄像头预览已启动: camera_index={self._camera_index}, resolution={self._resolution.label()}")
            
        except Exception as e:
            error_msg = tx(f"启动失败：{str(e)}", f"Start failed: {str(e)}")
            self.error_label.setText(error_msg)
            self.preview_label.setText(tx_button("启动失败", "Start Failed"))
            print(f"[CAMERA_PAGE] 启动错误: {e!r}")

    def _handle_stop(self) -> None:
        """停止摄像头预览。"""
        self._stream.stop()
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 重置显示
        self.preview_label.setText(tx_button("预览已停止", "Preview Stopped"))
        self.fps_value.setText("0.0")
        self.detect_value.setText(tx("未启动", "Not Started"))
        
        print("[CAMERA_PAGE] 摄像头预览已停止")

    def _on_error(self, msg: str) -> None:
        """处理摄像头错误。"""
        self.error_label.setText(tx(f"错误：{msg}", f"Error: {msg}"))
        self.detect_value.setText(tx("错误", "Error"))
        print(f"[CAMERA_PAGE] 摄像头错误: {msg}")

    def _on_frame(self, frame_bgr: object) -> None:
        """处理摄像头帧。
        
        Args:
            frame_bgr: BGR 格式的图像帧（numpy array）。
        """
        if not isinstance(frame_bgr, np.ndarray):
            return
        
        if self._detector is None:
            return
        
        # 计算 FPS
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        if elapsed >= 1.0:  # 每秒更新一次 FPS
            self._current_fps = self._frame_count / elapsed
            self.fps_value.setText(f"{self._current_fps:.1f}")
            self._frame_count = 0
            self._last_fps_time = current_time
        
        # 人脸检测（每 2 帧处理一次，减少计算负担）
        if self._frame_count % 2 == 0:
            try:
                result = self._detector.detect(frame_bgr)
                
                if result.detected:
                    self.detect_value.setText(tx("检测到", "Detected"))
                    
                    # 绘制人脸边界框
                    if result.face_bbox is not None:
                        x, y, w, h = result.face_bbox
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 绘制 68 个关键点
                    if result.landmarks_68 is not None:
                        for i, (px, py) in enumerate(result.landmarks_68):
                            # 不同区域使用不同颜色
                            if i < 17:  # 脸部轮廓
                                color = (255, 255, 0)  # 青色
                            elif i < 27:  # 眉毛
                                color = (0, 255, 255)  # 黄色
                            elif i < 36:  # 鼻子
                                color = (255, 0, 255)  # 品红
                            elif i < 48:  # 眼睛
                                color = (0, 0, 255)  # 红色
                            else:  # 嘴巴
                                color = (255, 0, 0)  # 蓝色
                            
                            cv2.circle(frame_bgr, (int(px), int(py)), 2, color, -1)
                else:
                    self.detect_value.setText(tx("未检测到", "Not Detected"))
                    
            except Exception as e:
                print(f"[CAMERA_PAGE] 人脸检测错误: {e!r}")
                self.detect_value.setText(tx("检测错误", "Detection Error"))
        
        # 水平镜像翻转（用户习惯）
        frame_bgr_mirrored = cv2.flip(frame_bgr, 1)
        
        # 转换为 QPixmap 并显示
        qimg = _bgr_to_qimage(frame_bgr_mirrored)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def closeEvent(self, event) -> None:
        """窗口关闭时停止摄像头。"""
        self._stream.stop()
        if self._detector is not None:
            self._detector.close()
        super().closeEvent(event)
