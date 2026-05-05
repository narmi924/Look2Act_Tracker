"""设置页面。

提供系统配置界面：
- 摄像头设置（索引、分辨率、帧率）
- 模型设置（权重路径、IPEX 优化、ONNX Runtime）
- 几何设置（屏幕物理尺寸）
- 平滑设置（平滑系数 alpha）
- 追踪设置（目标帧率）

设置变更实时生效，保存到 configs/system_config.yaml。

主要功能：
1. 加载当前配置
2. 提供各项设置的编辑控件
3. 保存配置到 YAML 文件
4. 恢复默认设置
5. 设置变更实时应用到 TrackerPipeline（如果正在运行）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QMessageBox,
)

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    DoubleSpinBox,
    ComboBox,
    LineEdit,
    SwitchButton,
    Slider,
    RadioButton,
    SmoothScrollArea,
)

from src.tracker.pipeline import SystemConfig
from src.ui.fluent_theme import PALETTE


class SettingsPage(QWidget):
    """设置页面。
    
    提供系统配置界面，支持：
    - 摄像头配置
    - 模型配置
    - 几何配置
    - 平滑配置
    - 追踪配置
    
    设置变更保存到 configs/system_config.yaml。
    
    信号：
    - config_changed: 配置变更信号，携带新的 SystemConfig 对象
    """
    
    config_changed = pyqtSignal(object)  # SystemConfig
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # 当前配置
        self.config: SystemConfig = SystemConfig()
        self.config_path = Path("configs/system_config.yaml")
        
        # 控件引用
        self.camera_index_spin: Optional[SpinBox] = None
        self.camera_width_combo: Optional[ComboBox] = None
        self.camera_height_combo: Optional[ComboBox] = None
        self.camera_backend_combo: Optional[ComboBox] = None
        
        self.model_path_edit: Optional[LineEdit] = None
        self.use_ipex_switch: Optional[SwitchButton] = None
        self.use_onnx_switch: Optional[SwitchButton] = None
        self.onnx_path_edit: Optional[LineEdit] = None
        self.deep_gaze_space_combo: Optional[ComboBox] = None
        self.deep_pose_input_combo: Optional[ComboBox] = None
        
        self.screen_w_mm_spin: Optional[DoubleSpinBox] = None
        self.screen_h_mm_spin: Optional[DoubleSpinBox] = None
        
        self.alpha_slider: Optional[Slider] = None
        self.alpha_value_label: Optional[BodyLabel] = None
        self.smoother_type_combo: Optional[ComboBox] = None
        
        self.target_fps_spin: Optional[SpinBox] = None
        self.backend_combo: Optional[ComboBox] = None
        self.advanced_visible = False
        self.model_card: Optional[CardWidget] = None
        self.geometry_card: Optional[CardWidget] = None
        
        self._init_ui()
        self._load_config()
    
    def _init_ui(self) -> None:
        """初始化 UI 布局。"""
        # 标题
        title = QLabel("系统设置 / System Settings")
        title.setStyleSheet("font-size: 32px; font-weight: 900;")
        
        # 控制按钮
        self.save_btn = PrimaryPushButton("保存设置\nSave Settings")
        self.save_btn.setFixedSize(160, 60)
        self.save_btn.clicked.connect(self._handle_save)
        
        self.reset_btn = PushButton("恢复默认\nReset Defaults")
        self.reset_btn.setFixedSize(160, 60)
        self.reset_btn.clicked.connect(self._handle_reset)

        self.advanced_btn = PushButton("高级设置\nAdvanced")
        self.advanced_btn.setFixedSize(140, 60)
        self.advanced_btn.clicked.connect(self._toggle_advanced_settings)
        
        # 顶部栏
        top_bar = QHBoxLayout()
        top_bar.addWidget(title)
        top_bar.addStretch(1)
        top_bar.addWidget(self.reset_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.advanced_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.save_btn)
        
        # 各种设置卡片
        window_card = self._create_window_settings_card()
        camera_card = self._create_camera_settings_card()
        self.model_card = self._create_model_settings_card()
        self.geometry_card = self._create_geometry_settings_card()
        smoother_card = self._create_smoother_settings_card()
        tracker_card = self._create_tracker_settings_card()
        self.model_card.hide()
        self.geometry_card.hide()
        
        # 状态信息
        self.status_label = BodyLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: 600; margin-top: 10px;")
        self.status_label.setWordWrap(True)
        
        # 创建可滚动区域
        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("QWidget { background: transparent; }")
        
        scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_layout.setContentsMargins(10, 0, 10, 0)
        scroll_layout.setSpacing(16)
        
        scroll_layout.addWidget(window_card)
        scroll_layout.addWidget(camera_card)
        scroll_layout.addWidget(self.model_card)
        scroll_layout.addWidget(self.geometry_card)
        scroll_layout.addWidget(smoother_card)
        scroll_layout.addWidget(tracker_card)
        scroll_layout.addWidget(self.status_label)
        scroll_layout.addStretch(1)
        
        self.scroll_area.setWidget(self.scroll_widget)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(16)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.scroll_area)
    
    def _create_window_settings_card(self) -> CardWidget:
        """创建窗口显示设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("窗口显示设置 / Window Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 提示信息
        hint = BodyLabel("💡 自适应模式：保留系统任务栏可见\n💡 全屏模式：覆盖任务栏（相当于独立应用形态）\n注意：设置将在下次启动时生效。")
        hint.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 8px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # 模式单选框
        self.adaptive_radio = RadioButton("自适应模式（保留任务栏） / Adaptive Mode")
        self.fullscreen_radio = RadioButton("全屏模式（覆盖任务栏） / Fullscreen Mode")
        
        layout.addWidget(self.adaptive_radio)
        layout.addWidget(self.fullscreen_radio)
        
        return card
    
    def _create_camera_settings_card(self) -> CardWidget:
        """创建摄像头设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("摄像头设置 / Camera Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 摄像头索引
        index_row = QHBoxLayout()
        index_label = BodyLabel("摄像头索引 / Camera Index:")
        index_label.setFixedWidth(200)
        self.camera_index_spin = SpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)
        self.camera_index_spin.setFixedWidth(100)
        index_row.addWidget(index_label)
        index_row.addWidget(self.camera_index_spin)
        index_row.addStretch(1)
        layout.addLayout(index_row)
        
        # 分辨率（宽度）
        width_row = QHBoxLayout()
        width_label = BodyLabel("分辨率宽度 / Width (px):")
        width_label.setFixedWidth(200)
        self.camera_width_combo = ComboBox()
        self.camera_width_combo.addItems(["320", "640", "1280", "1920"])
        self.camera_width_combo.setCurrentText("640")
        self.camera_width_combo.setFixedWidth(150)
        width_row.addWidget(width_label)
        width_row.addWidget(self.camera_width_combo)
        width_row.addStretch(1)
        layout.addLayout(width_row)
        
        # 分辨率（高度）
        height_row = QHBoxLayout()
        height_label = BodyLabel("分辨率高度 / Height (px):")
        height_label.setFixedWidth(200)
        self.camera_height_combo = ComboBox()
        self.camera_height_combo.addItems(["240", "480", "720", "1080"])
        self.camera_height_combo.setCurrentText("480")
        self.camera_height_combo.setFixedWidth(150)
        height_row.addWidget(height_label)
        height_row.addWidget(self.camera_height_combo)
        height_row.addStretch(1)
        layout.addLayout(height_row)
        
        # 后端
        backend_row = QHBoxLayout()
        backend_label = BodyLabel("摄像头后端 / Backend:")
        backend_label.setFixedWidth(200)
        self.camera_backend_combo = ComboBox()
        self.camera_backend_combo.addItems(["dshow", "auto"])
        self.camera_backend_combo.setCurrentText("dshow")
        self.camera_backend_combo.setFixedWidth(150)
        backend_row.addWidget(backend_label)
        backend_row.addWidget(self.camera_backend_combo)
        backend_row.addStretch(1)
        layout.addLayout(backend_row)
        
        return card
    
    def _create_model_settings_card(self) -> CardWidget:
        """创建模型设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("模型设置 / Model Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 模型权重路径
        model_path_row = QHBoxLayout()
        model_path_label = BodyLabel("模型权重路径 / Model Path:")
        model_path_label.setFixedWidth(200)
        self.model_path_edit = LineEdit()
        self.model_path_edit.setPlaceholderText("checkpoints/best_model.pth")
        self.model_path_edit.setText("checkpoints/best_model.pth")
        browse_model_btn = PushButton("浏览 / Browse")
        browse_model_btn.setFixedWidth(120)
        browse_model_btn.clicked.connect(self._browse_model_path)
        model_path_row.addWidget(model_path_label)
        model_path_row.addWidget(self.model_path_edit, stretch=1)
        model_path_row.addWidget(browse_model_btn)
        layout.addLayout(model_path_row)
        
        # IPEX 优化开关
        ipex_row = QHBoxLayout()
        ipex_label = BodyLabel("IPEX 优化 / IPEX Optimization:")
        ipex_label.setFixedWidth(200)
        self.use_ipex_switch = SwitchButton()
        self.use_ipex_switch.setChecked(False)
        ipex_hint = BodyLabel("(Intel Extension for PyTorch)")
        ipex_hint.setStyleSheet("color: #888; font-size: 12px;")
        ipex_row.addWidget(ipex_label)
        ipex_row.addWidget(self.use_ipex_switch)
        ipex_row.addWidget(ipex_hint)
        ipex_row.addStretch(1)
        layout.addLayout(ipex_row)
        
        # ONNX Runtime 开关
        onnx_row = QHBoxLayout()
        onnx_label = BodyLabel("ONNX Runtime:")
        onnx_label.setFixedWidth(200)
        self.use_onnx_switch = SwitchButton()
        self.use_onnx_switch.setChecked(True)
        onnx_hint = BodyLabel("(推荐 / Recommended)")
        onnx_hint.setStyleSheet("color: #888; font-size: 12px;")
        onnx_row.addWidget(onnx_label)
        onnx_row.addWidget(self.use_onnx_switch)
        onnx_row.addWidget(onnx_hint)
        onnx_row.addStretch(1)
        layout.addLayout(onnx_row)
        
        # ONNX 模型路径
        onnx_path_row = QHBoxLayout()
        onnx_path_label = BodyLabel("ONNX 模型路径 / ONNX Path:")
        onnx_path_label.setFixedWidth(200)
        self.onnx_path_edit = LineEdit()
        self.onnx_path_edit.setPlaceholderText("checkpoints/gaze_net.onnx")
        self.onnx_path_edit.setText("checkpoints/gaze_net.onnx")
        browse_onnx_btn = PushButton("浏览 / Browse")
        browse_onnx_btn.setFixedWidth(120)
        browse_onnx_btn.clicked.connect(self._browse_onnx_path)
        onnx_path_row.addWidget(onnx_path_label)
        onnx_path_row.addWidget(self.onnx_path_edit, stretch=1)
        onnx_path_row.addWidget(browse_onnx_btn)
        layout.addLayout(onnx_path_row)

        # Deep gaze coordinate space
        deep_space_row = QHBoxLayout()
        deep_space_label = BodyLabel("Deep 坐标空间 / Deep Space:")
        deep_space_label.setFixedWidth(200)
        self.deep_gaze_space_combo = ComboBox()
        self.deep_gaze_space_combo.addItems(["head", "camera"])
        self.deep_gaze_space_combo.setCurrentText("head")
        self.deep_gaze_space_combo.setFixedWidth(150)
        deep_space_hint = BodyLabel("head 为原链路，camera 用于跳过 PnP 旋转实验")
        deep_space_hint.setStyleSheet("color: #888; font-size: 12px;")
        deep_space_row.addWidget(deep_space_label)
        deep_space_row.addWidget(self.deep_gaze_space_combo)
        deep_space_row.addWidget(deep_space_hint)
        deep_space_row.addStretch(1)
        layout.addLayout(deep_space_row)

        deep_pose_row = QHBoxLayout()
        deep_pose_label = BodyLabel("Deep 姿态输入 / Pose Input:")
        deep_pose_label.setFixedWidth(200)
        self.deep_pose_input_combo = ComboBox()
        self.deep_pose_input_combo.addItems(["live", "zero"])
        self.deep_pose_input_combo.setCurrentText("live")
        self.deep_pose_input_combo.setFixedWidth(150)
        deep_pose_hint = BodyLabel("zero 用于排查坏 head-pose 特征污染")
        deep_pose_hint.setStyleSheet("color: #888; font-size: 12px;")
        deep_pose_row.addWidget(deep_pose_label)
        deep_pose_row.addWidget(self.deep_pose_input_combo)
        deep_pose_row.addWidget(deep_pose_hint)
        deep_pose_row.addStretch(1)
        layout.addLayout(deep_pose_row)
        
        return card
    
    def _create_geometry_settings_card(self) -> CardWidget:
        """创建几何设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("几何设置 / Geometry Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 提示信息
        hint = BodyLabel("请使用尺子测量屏幕的实际物理尺寸（不含边框）\nPlease measure the actual physical size of the screen (excluding bezels)")
        hint.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 8px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # 屏幕宽度
        width_row = QHBoxLayout()
        width_label = BodyLabel("屏幕宽度 / Screen Width (mm):")
        width_label.setFixedWidth(220)
        self.screen_w_mm_spin = DoubleSpinBox()
        self.screen_w_mm_spin.setRange(100.0, 1000.0)
        self.screen_w_mm_spin.setValue(344.0)
        self.screen_w_mm_spin.setDecimals(1)
        self.screen_w_mm_spin.setSingleStep(1.0)
        self.screen_w_mm_spin.setFixedWidth(120)
        width_row.addWidget(width_label)
        width_row.addWidget(self.screen_w_mm_spin)
        width_row.addStretch(1)
        layout.addLayout(width_row)
        
        # 屏幕高度
        height_row = QHBoxLayout()
        height_label = BodyLabel("屏幕高度 / Screen Height (mm):")
        height_label.setFixedWidth(220)
        self.screen_h_mm_spin = DoubleSpinBox()
        self.screen_h_mm_spin.setRange(100.0, 1000.0)
        self.screen_h_mm_spin.setValue(194.0)
        self.screen_h_mm_spin.setDecimals(1)
        self.screen_h_mm_spin.setSingleStep(1.0)
        self.screen_h_mm_spin.setFixedWidth(120)
        height_row.addWidget(height_label)
        height_row.addWidget(self.screen_h_mm_spin)
        height_row.addStretch(1)
        layout.addLayout(height_row)
        
        return card
    
    def _create_smoother_settings_card(self) -> CardWidget:
        """创建平滑设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("平滑设置 / Smoothing Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 提示信息
        hint = BodyLabel("平滑系数越小，注视点越平滑但响应越慢\nSmaller alpha = smoother but slower response")
        hint.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 8px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 平滑方式
        type_row = QHBoxLayout()
        type_label = BodyLabel("平滑方式 / Smoother:")
        type_label.setFixedWidth(200)
        self.smoother_type_combo = ComboBox()
        self.smoother_type_combo.addItems(["kalman", "ema", "none"])
        self.smoother_type_combo.setCurrentText("kalman")
        self.smoother_type_combo.setFixedWidth(150)
        type_hint = BodyLabel("classic 默认 Kalman；EMA/none 用于对照")
        type_hint.setStyleSheet("color: #888; font-size: 12px;")
        type_row.addWidget(type_label)
        type_row.addWidget(self.smoother_type_combo)
        type_row.addWidget(type_hint)
        type_row.addStretch(1)
        layout.addLayout(type_row)
        
        # 平滑系数 alpha
        alpha_row = QHBoxLayout()
        alpha_label = BodyLabel("平滑系数 / Alpha:")
        alpha_label.setFixedWidth(200)
        self.alpha_slider = Slider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(1, 100)  # 0.01 ~ 1.00
        self.alpha_slider.setValue(30)  # 默认 0.30
        self.alpha_slider.setFixedWidth(300)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.alpha_value_label = BodyLabel("0.30")
        self.alpha_value_label.setStyleSheet(f"font-weight: 600; color: {PALETTE['accent']}; font-size: 16px;")
        self.alpha_value_label.setFixedWidth(60)
        alpha_row.addWidget(alpha_label)
        alpha_row.addWidget(self.alpha_slider)
        alpha_row.addWidget(self.alpha_value_label)
        alpha_row.addStretch(1)
        layout.addLayout(alpha_row)
        
        return card
    
    def _create_tracker_settings_card(self) -> CardWidget:
        """创建追踪设置卡片。"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # 标题
        title = BodyLabel("追踪设置 / Tracking Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        
        # 目标帧率
        backend_row = QHBoxLayout()
        backend_label = BodyLabel("追踪模式 / Backend:")
        backend_label.setFixedWidth(200)
        self.backend_combo = ComboBox()
        self.backend_combo.addItems(["classic", "deep"])
        self.backend_combo.setCurrentText("classic")
        self.backend_combo.setFixedWidth(150)
        backend_hint = BodyLabel("classic 用于流畅体验，deep 用于研究模型")
        backend_hint.setStyleSheet("color: #888; font-size: 12px;")
        backend_row.addWidget(backend_label)
        backend_row.addWidget(self.backend_combo)
        backend_row.addWidget(backend_hint)
        backend_row.addStretch(1)
        layout.addLayout(backend_row)

        # 目标帧率
        fps_row = QHBoxLayout()
        fps_label = BodyLabel("目标帧率 / Target FPS:")
        fps_label.setFixedWidth(200)
        self.target_fps_spin = SpinBox()
        self.target_fps_spin.setRange(10, 60)
        self.target_fps_spin.setValue(30)
        self.target_fps_spin.setFixedWidth(100)
        fps_row.addWidget(fps_label)
        fps_row.addWidget(self.target_fps_spin)
        fps_row.addStretch(1)
        layout.addLayout(fps_row)
        
        return card

    def _toggle_advanced_settings(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.model_card is not None:
            self.model_card.setVisible(self.advanced_visible)
        if self.geometry_card is not None:
            self.geometry_card.setVisible(self.advanced_visible)
        self.advanced_btn.setText("隐藏高级\nAdvanced" if self.advanced_visible else "高级设置\nAdvanced")
    
    def _on_alpha_changed(self, value: int) -> None:
        """平滑系数滑块变化回调。"""
        alpha = value / 100.0
        if self.alpha_value_label is not None:
            self.alpha_value_label.setText(f"{alpha:.2f}")
    
    def _browse_model_path(self) -> None:
        """浏览模型权重文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型权重文件 / Select Model File",
            "checkpoints",
            "PyTorch 模型 (*.pth *.pt);;所有文件 (*.*)"
        )
        
        if file_path and self.model_path_edit is not None:
            self.model_path_edit.setText(file_path)
    
    def _browse_onnx_path(self) -> None:
        """浏览 ONNX 模型文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型文件 / Select ONNX File",
            "checkpoints",
            "ONNX 模型 (*.onnx);;所有文件 (*.*)"
        )
        
        if file_path and self.onnx_path_edit is not None:
            self.onnx_path_edit.setText(file_path)
    
    def _load_config(self) -> None:
        """从 YAML 文件加载配置。"""
        if not self.config_path.exists():
            print(f"[SETTINGS_PAGE] 配置文件不存在，使用默认配置: {self.config_path}")
            self._update_ui_from_config()
            return
        
        try:
            self.config = SystemConfig.from_yaml(str(self.config_path))
            self._update_ui_from_config()
            print(f"[SETTINGS_PAGE] 配置已加载: {self.config_path}")
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "加载失败",
                f"加载配置文件失败：{e}\n\n将使用默认配置。"
            )
            print(f"[SETTINGS_PAGE] 加载配置失败: {e}")
    
    def _update_ui_from_config(self) -> None:
        """根据配置对象更新 UI 控件。"""
        # 窗口设置
        if hasattr(self, 'fullscreen_radio'):
            if self.config.window_mode == 'fullscreen':
                self.fullscreen_radio.setChecked(True)
            else:
                self.adaptive_radio.setChecked(True)
                
        # 摄像头设置
        if self.camera_index_spin is not None:
            self.camera_index_spin.setValue(self.config.camera_index)
        if self.camera_width_combo is not None:
            self.camera_width_combo.setCurrentText(str(self.config.camera_width))
        if self.camera_height_combo is not None:
            self.camera_height_combo.setCurrentText(str(self.config.camera_height))
        if self.camera_backend_combo is not None:
            self.camera_backend_combo.setCurrentText(self.config.camera_backend)
        
        # 模型设置
        if self.model_path_edit is not None:
            self.model_path_edit.setText(self.config.checkpoint_path)
        if self.use_ipex_switch is not None:
            self.use_ipex_switch.setChecked(self.config.use_ipex)
        if self.use_onnx_switch is not None:
            self.use_onnx_switch.setChecked(self.config.use_onnx)
        if self.onnx_path_edit is not None:
            self.onnx_path_edit.setText(self.config.onnx_path)
        if self.deep_gaze_space_combo is not None:
            self.deep_gaze_space_combo.setCurrentText(self.config.deep_gaze_space)
        if self.deep_pose_input_combo is not None:
            self.deep_pose_input_combo.setCurrentText(self.config.normalized_deep_pose_input)
        if self.backend_combo is not None:
            self.backend_combo.setCurrentText(self.config.normalized_backend)
        
        # 几何设置
        if self.screen_w_mm_spin is not None:
            self.screen_w_mm_spin.setValue(self.config.screen_w_mm)
        if self.screen_h_mm_spin is not None:
            self.screen_h_mm_spin.setValue(self.config.screen_h_mm)
        
        # 平滑设置
        if self.alpha_slider is not None:
            alpha_int = int(self.config.smoother_alpha * 100)
            self.alpha_slider.setValue(alpha_int)
            if self.alpha_value_label is not None:
                self.alpha_value_label.setText(f"{self.config.smoother_alpha:.2f}")
        if self.smoother_type_combo is not None:
            self.smoother_type_combo.setCurrentText(self.config.normalized_smoother_type)
        
        # 追踪设置
        if self.target_fps_spin is not None:
            self.target_fps_spin.setValue(self.config.target_fps)
    
    def _update_config_from_ui(self) -> None:
        """根据 UI 控件更新配置对象。"""
        # 窗口设置
        if hasattr(self, 'fullscreen_radio'):
            self.config.window_mode = 'fullscreen' if self.fullscreen_radio.isChecked() else 'adaptive'
            
        # 摄像头设置
        if self.camera_index_spin is not None:
            self.config.camera_index = self.camera_index_spin.value()
        if self.camera_width_combo is not None:
            self.config.camera_width = int(self.camera_width_combo.currentText())
        if self.camera_height_combo is not None:
            self.config.camera_height = int(self.camera_height_combo.currentText())
        if self.camera_backend_combo is not None:
            self.config.camera_backend = self.camera_backend_combo.currentText()
        if self.backend_combo is not None:
            self.config.tracker_backend = self.backend_combo.currentText()
        
        # 模型设置
        if self.model_path_edit is not None:
            self.config.checkpoint_path = self.model_path_edit.text()
        if self.use_ipex_switch is not None:
            self.config.use_ipex = self.use_ipex_switch.isChecked()
        if self.use_onnx_switch is not None:
            self.config.use_onnx = self.use_onnx_switch.isChecked()
        if self.onnx_path_edit is not None:
            self.config.onnx_path = self.onnx_path_edit.text()
        if self.deep_gaze_space_combo is not None:
            value = self.deep_gaze_space_combo.currentText()
            self.config.deep_gaze_space = value if value in {"head", "camera"} else "head"
        if self.deep_pose_input_combo is not None:
            value = self.deep_pose_input_combo.currentText()
            self.config.deep_pose_input = value if value in {"live", "zero"} else "live"
        
        # 几何设置
        if self.screen_w_mm_spin is not None:
            self.config.screen_w_mm = self.screen_w_mm_spin.value()
        if self.screen_h_mm_spin is not None:
            self.config.screen_h_mm = self.screen_h_mm_spin.value()
        
        # 平滑设置
        if self.alpha_slider is not None:
            self.config.smoother_alpha = self.alpha_slider.value() / 100.0
        if self.smoother_type_combo is not None:
            self.config.smoother_type = self.smoother_type_combo.currentText()
        
        # 追踪设置
        if self.target_fps_spin is not None:
            self.config.target_fps = self.target_fps_spin.value()
    
    def _handle_save(self) -> None:
        """保存设置到 YAML 文件。"""
        try:
            # 更新配置对象
            self._update_config_from_ui()
            
            # 确保配置目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建 YAML 数据
            config_data = {
                'ui': {
                    'window_mode': self.config.window_mode,
                },
                'camera': {
                    'index': self.config.camera_index,
                    'width': self.config.camera_width,
                    'height': self.config.camera_height,
                    'backend': self.config.camera_backend,
                },
                'face_detection': {
                    'eye_crop_size': self.config.eye_crop_size,
                    'min_detection_confidence': self.config.min_detection_confidence,
                    'min_tracking_confidence': self.config.min_tracking_confidence,
                },
                'model': {
                    'checkpoint_path': self.config.checkpoint_path,
                    'use_ipex': self.config.use_ipex,
                    'use_onnx': self.config.use_onnx,
                    'onnx_path': self.config.onnx_path,
                    'deep_gaze_space': self.config.deep_gaze_space,
                    'deep_pose_input': self.config.deep_pose_input,
                },
                'geometry': {
                    'screen_w_mm': self.config.screen_w_mm,
                    'screen_h_mm': self.config.screen_h_mm,
                    'screen_distance_mm': self.config.screen_distance_mm,
                    'cam_above_screen_mm': self.config.cam_above_screen_mm,
                },
                'calibration': {
                    'num_points': 25 if self.config.normalized_backend == 'classic' else 9,
                    'save_path': self.config.calibration_path,
                    'max_residual_px': 300.0,
                },
                'smoother': {
                    'type': self.config.smoother_type,
                    'alpha': self.config.smoother_alpha,
                },
                'tracker': {
                    'backend': self.config.normalized_backend,
                    'target_fps': self.config.target_fps,
                    'timer_interval_ms': int(1000 / self.config.target_fps),
                },
            }
            
            # 保存到 YAML 文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            # 更新状态
            self.status_label.setText(f"✓ 设置已保存到 {self.config_path}")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: 600;")
            
            # 发出配置变更信号
            self.config_changed.emit(self.config)
            
            print(f"[SETTINGS_PAGE] 设置已保存: {self.config_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存设置失败：{e}")
            self.status_label.setText(f"✗ 保存失败: {e}")
            self.status_label.setStyleSheet("color: #D32F2F; font-weight: 600;")
            print(f"[SETTINGS_PAGE] 保存失败: {e}")
    
    def _handle_reset(self) -> None:
        """恢复默认设置。"""
        reply = QMessageBox.question(
            self,
            "确认恢复默认",
            "确定要恢复所有设置为默认值吗？\n\nAre you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 重置为默认配置
            self.config = SystemConfig()
            self._update_ui_from_config()
            
            self.status_label.setText("✓ 已恢复默认设置（未保存）")
            self.status_label.setStyleSheet("color: #FF9800; font-weight: 600;")
            
            print("[SETTINGS_PAGE] 已恢复默认设置")
    
    def get_config(self) -> SystemConfig:
        """获取当前配置对象。
        
        返回:
            SystemConfig 对象
        """
        self._update_config_from_ui()
        return self.config
