"""主页。

显示系统欢迎信息、功能介绍和快速导航。

主要功能：
1. 显示系统标题和欢迎信息
2. 显示功能卡片（摄像头预览、校准、实时追踪、设置）
3. 提供快速导航按钮
4. 显示系统状态信息
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
)

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    PrimaryPushButton,
    TitleLabel,
    SmoothScrollArea,
)


class FeatureCard(CardWidget):
    """功能卡片。
    
    显示单个功能的图标、标题、描述和操作按钮。
    """
    
    clicked = pyqtSignal()
    
    def __init__(
        self,
        title: str,
        description: str,
        icon: str = "📷",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        # 布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # 图标
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 标题
        title_label = TitleLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 描述
        desc_label = BodyLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #888; font-size: 13px;")
        
        # 按钮
        self.action_btn = PrimaryPushButton("进入 / Enter")
        self.action_btn.setFixedHeight(40)
        self.action_btn.clicked.connect(self.clicked.emit)
        
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch(1)
        layout.addWidget(self.action_btn)
        
        # 样式
        self.setFixedHeight(280)


class HomePage(QWidget):
    """主页。
    
    显示系统欢迎信息和功能导航。
    
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
        """初始化 UI 布局。"""
        # 标题
        title = QLabel("Look2Act Tracker")
        title.setStyleSheet("font-size: 48px; font-weight: 900; color: #0078D4;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 副标题
        subtitle = BodyLabel("视线驱动交互系统 / Gaze-Driven Interaction System")
        subtitle.setStyleSheet("font-size: 18px; color: #666;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 欢迎信息
        welcome = BodyLabel(
            "欢迎使用 Look2Act Tracker！\n"
            "本系统基于三维视线方向回归技术，实现实时视线追踪与交互。\n\n"
            "Welcome to Look2Act Tracker!\n"
            "Real-time gaze tracking system based on 3D gaze direction regression."
        )
        welcome.setWordWrap(True)
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet("font-size: 14px; color: #888; line-height: 1.6;")
        
        # 功能卡片网格
        cards_layout = QGridLayout()
        cards_layout.setSpacing(20)
        
        # 摄像头预览卡片
        camera_card = FeatureCard(
            title="摄像头预览 / Camera Preview",
            description="实时显示摄像头画面，叠加人脸检测框和关键点可视化",
            icon="📷"
        )
        camera_card.clicked.connect(self.navigate_to_camera.emit)
        
        # 校准卡片
        calibration_card = FeatureCard(
            title="视线校准 / Calibration",
            description="通过注视校准点优化视线映射精度，提高追踪准确性",
            icon="🎯"
        )
        calibration_card.clicked.connect(self.navigate_to_calibration.emit)
        
        # 实时追踪卡片
        tracking_card = FeatureCard(
            title="实时追踪 / Real-time Tracking",
            description="启动视线追踪，在屏幕上显示注视点光标和性能监控",
            icon="👁️"
        )
        tracking_card.clicked.connect(self.navigate_to_tracking.emit)
        
        # 设置卡片
        settings_card = FeatureCard(
            title="系统设置 / Settings",
            description="配置摄像头、模型、几何参数和平滑系数等系统参数",
            icon="⚙️"
        )
        settings_card.clicked.connect(self.navigate_to_settings.emit)
        
        # 添加到网格（2 行 × 2 列）
        cards_layout.addWidget(camera_card, 0, 0)
        cards_layout.addWidget(calibration_card, 0, 1)
        cards_layout.addWidget(tracking_card, 1, 0)
        cards_layout.addWidget(settings_card, 1, 1)
        
        # 系统信息卡片
        info_card = CardWidget()
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(20, 16, 20, 16)
        info_layout.setSpacing(8)
        
        info_title = BodyLabel("系统信息 / System Information")
        info_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        
        info_content = BodyLabel(
            "• 技术路线：三维视线方向回归 + 头部姿态估计 + 屏幕几何建模\n"
            "• 运行环境：Windows + Intel CPU/IPEX\n"
            "• 目标性能：≥15 FPS 端到端延迟 <66ms\n"
            "• 校准方式：9 点仿射变换校准"
        )
        info_content.setStyleSheet("font-size: 12px; color: #666; line-height: 1.8;")
        
        info_layout.addWidget(info_title)
        info_layout.addWidget(info_content)
        
        # 创建可滚动区域
        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("QWidget { background: transparent; }")
        
        scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(24)
        scroll_layout.addWidget(title)
        scroll_layout.addWidget(subtitle)
        scroll_layout.addSpacing(10)
        scroll_layout.addWidget(welcome)
        scroll_layout.addSpacing(20)
        scroll_layout.addLayout(cards_layout)
        scroll_layout.addSpacing(10)
        scroll_layout.addWidget(info_card)
        scroll_layout.addStretch(1)
        
        self.scroll_area.setWidget(self.scroll_widget)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(60, 40, 60, 40)
        main_layout.addWidget(self.scroll_area)

