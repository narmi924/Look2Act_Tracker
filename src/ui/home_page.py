"""主页。

参照 Eye_Touch 首页布局：标题区域 + 3 张水平排列的步骤卡片。
卡片为摄像头预览、视线校准、实时追踪三个核心功能。

主要功能：
1. 显示系统标题和欢迎信息
2. 显示 3 张水平排列的功能卡片（步骤 1/2/3）
3. 提供快速导航按钮
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
)

from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    PrimaryPushButton,
    TitleLabel,
)

from src.ui.fluent_theme import PALETTE


class HomePage(QWidget):
    """主页。
    
    显示系统欢迎信息和功能导航。
    参照 Eye_Touch 首页：标题区域 + 3 张水平等分的步骤卡片。
    
    信号：
    - navigate_to_camera: 导航到摄像头预览页面
    - navigate_to_calibration: 导航到校准页面
    - navigate_to_tracking: 导航到实时追踪页面
    - navigate_to_settings: 导航到设置页面
    """
    
    navigate_to_camera = pyqtSignal()
    navigate_to_calibration = pyqtSignal()
    navigate_to_tracking = pyqtSignal()
    navigate_to_settings = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化 UI 布局（参照 Eye_Touch 首页结构）。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 15, 25, 15)
        layout.setSpacing(12)
        
        # 标题区域
        self._create_title_area(layout)
        
        # 导航卡片区域（3 张水平排列）
        self._create_navigation_cards(layout)
    
    def _create_title_area(self, parent_layout: QVBoxLayout) -> None:
        """创建标题区域（参照 Eye_Touch 的 create_title_area）。"""
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")
        title_frame.setStyleSheet(f"""
            #titleFrame {{
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(69, 40, 41, 0.08), stop:1 rgba(69, 40, 41, 0.03));
                border: 1px solid rgba(69, 40, 41, 0.15);
                border-radius: 14px;
                margin: 10px 0px;
            }}
        """)
        
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(40, 30, 40, 30)
        title_layout.setSpacing(15)
        
        # 主标题
        main_title = TitleLabel("Look2Act Tracker")
        main_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_title.setStyleSheet(f"""
            QLabel {{
                font-size: 28px;
                font-weight: 800;
                color: {PALETTE['accent']};
                margin: 0px;
            }}
        """)
        title_layout.addWidget(main_title)

        # 副标题
        sub_title = CaptionLabel("视线驱动交互系统 · Gaze-Driven Interaction System")
        sub_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_title.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 14px;
                font-weight: 500;
                letter-spacing: 1px;
                margin: 0px;
            }
        """)
        title_layout.addWidget(sub_title)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("""
            QFrame {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 #BDC3C7, stop:1 transparent);
                border: none;
                height: 1px;
                margin: 8px 40px;
            }
        """)
        title_layout.addWidget(separator)
        
        # 说明文字
        desc = CaptionLabel("请按照以下步骤完成视线追踪的准备和使用")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 13px;
                font-style: italic;
                margin: 8px 0px 0px 0px;
            }
        """)
        title_layout.addWidget(desc)
        
        parent_layout.addWidget(title_frame)
    
    def _create_navigation_cards(self, parent_layout: QVBoxLayout) -> None:
        """创建 3 张水平排列的导航卡片（参照 Eye_Touch 的 create_navigation_cards）。"""
        frame = QFrame()
        cards_layout = QHBoxLayout(frame)
        cards_layout.setSpacing(25)
        cards_layout.setContentsMargins(10, 0, 10, 0)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 卡片通用样式（使用 PALETTE 配色）
        card_style = f"""
            CardWidget {{
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(69, 40, 41, 0.06), stop:1 rgba(69, 40, 41, 0.02));
                border: 1px solid rgba(69, 40, 41, 0.15);
                border-radius: 16px;
                margin: 8px;
                padding: 20px;
            }}
        """
        
        btn_style = f"""
            PrimaryPushButton {{
                background: {PALETTE['accent']};
                color: {PALETTE['bg']};
                border: none;
                border-radius: 12px;
                padding: 10px 18px;
                font-weight: 600;
            }}
            PrimaryPushButton:hover {{
                background: #3A2223;
            }}
            PrimaryPushButton:pressed {{
                background: #2D191A;
            }}
        """
        
        # --- 步骤 1：摄像头预览 ---
        cam_card = CardWidget()
        cam_card.setStyleSheet(card_style)
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.addWidget(TitleLabel("步骤 1：摄像头预览"))
        cam_layout.addWidget(CaptionLabel("检查摄像头画面和人脸检测"))
        cam_layout.addStretch()
        cam_btn = PrimaryPushButton("开始预览 / Preview")
        cam_btn.setStyleSheet(btn_style)
        cam_btn.clicked.connect(self.navigate_to_camera.emit)
        cam_layout.addWidget(cam_btn)
        
        # --- 步骤 2：视线校准 ---
        calib_card = CardWidget()
        calib_card.setStyleSheet(card_style)
        calib_layout = QVBoxLayout(calib_card)
        calib_layout.addWidget(TitleLabel("步骤 2：视线校准"))
        calib_layout.addWidget(CaptionLabel("注视校准点优化映射精度"))
        calib_layout.addStretch()
        calib_btn = PrimaryPushButton("开始校准 / Calibrate")
        calib_btn.setStyleSheet(btn_style)
        calib_btn.clicked.connect(self.navigate_to_calibration.emit)
        calib_layout.addWidget(calib_btn)
        
        # --- 步骤 3：实时追踪 ---
        track_card = CardWidget()
        track_card.setStyleSheet(card_style)
        track_layout = QVBoxLayout(track_card)
        track_layout.addWidget(TitleLabel("步骤 3：验证与交互"))
        track_layout.addWidget(CaptionLabel("进入全屏验证和独立交互窗口"))
        track_layout.addStretch()
        track_btn = PrimaryPushButton("开始追踪 / Track")
        track_btn.setStyleSheet(btn_style)
        track_btn.clicked.connect(self.navigate_to_tracking.emit)
        track_layout.addWidget(track_btn)
        
        # 水平等分排列
        cards_layout.addWidget(cam_card, 1)
        cards_layout.addWidget(calib_card, 1)
        cards_layout.addWidget(track_card, 1)
        
        parent_layout.addWidget(frame, 1)
