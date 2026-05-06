"""主窗口。

管理整个应用的页面导航和状态，使用 QStackedWidget 实现页面切换。

页面流程：
主页 ↔ 摄像头预览 ↔ 校准 ↔ 实时追踪 ↔ 设置

状态管理：
- 管理 TrackerPipeline 单例（在需要时创建，应用退出时释放）
- 管理页面间的导航
- 处理应用生命周期（启动、退出、资源释放）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox

from qfluentwidgets import NavigationInterface, NavigationItemPosition, FluentIcon as FIF, FluentWindow

from src.ui.home_page import HomePage
from src.ui.camera_page import CameraPage
from src.ui.calibration_page import CalibrationPage
from src.ui.tracking_page import TrackingPage
from src.ui.settings_page import SettingsPage
from src.ui.i18n import tx
from src.tracker.pipeline import TrackerPipeline, SystemConfig


class MainWindow(FluentWindow):
    """主窗口类。
    
    负责管理所有页面和页面间的导航。
    使用 FluentWindow 实现左侧滑动测边导航。
    
    页面导航流程：
    - 主页 ↔ 各功能页面
    - 按 Esc 键返回主页（或退出）
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Look2Act Tracker - {tx('视线驱动交互系统', 'Gaze-Driven Interaction System')}")
        
        # TrackerPipeline 单例（延迟初始化）
        self.tracker: Optional[TrackerPipeline] = None
        self.tracker_config: Optional[SystemConfig] = None
        
        # 创建所有页面
        self.page_home = HomePage()
        self.page_camera = CameraPage()
        self.page_calibration = CalibrationPage()
        self.page_tracking = TrackingPage()
        self.page_settings = SettingsPage()

        # 设置 objectName 供 FluentWindow 路由标识
        self.page_home.setObjectName("HomePage")
        self.page_camera.setObjectName("CameraPage")
        self.page_calibration.setObjectName("CalibrationPage")
        self.page_tracking.setObjectName("TrackingPage")
        self.page_settings.setObjectName("SettingsPage")
        
        # 添加页面到侧边导航栏
        self.addSubInterface(self.page_home, FIF.HOME, tx('主页', 'Home'))
        self.addSubInterface(self.page_camera, FIF.PHOTO, tx('预览', 'Preview'))
        self.addSubInterface(self.page_calibration, FIF.EDIT, tx('校准', 'Calibrate'))
        self.addSubInterface(self.page_tracking, FIF.VIEW, tx('追踪', 'Track'))
        self.addSubInterface(self.page_settings, FIF.SETTING, tx('设置', 'Settings'))
        
        # 连接主页导航信号
        self.page_home.navigate_to_camera.connect(self.go_camera)
        self.page_home.navigate_to_calibration.connect(self.go_calibration)
        self.page_home.navigate_to_tracking.connect(self.go_tracking)
        self.page_home.navigate_to_settings.connect(self.go_settings)
        
        # 连接设置页面配置变更信号
        self.page_settings.config_changed.connect(self._on_config_changed)
        self.page_calibration.calibration_ready.connect(self._on_calibration_ready)
        self.page_calibration.return_home_requested.connect(self.go_home)
        
        # 获取屏幕尺寸以便像 Eye_Touch 一样进行自适应全屏布局
        from PyQt6.QtWidgets import QApplication
        import yaml
        
        screen = QApplication.primaryScreen()
        
        # 读取窗口模式配置（参照 Eye_Touch 的 setup_fullscreen_window）
        window_mode = 'adaptive'  # 默认自适应模式
        try:
            with open("configs/system_config.yaml", 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                window_mode = data.get('ui', {}).get('window_mode', 'adaptive')
        except Exception:
            pass
            
        if window_mode == 'fullscreen':
            # 全屏模式（覆盖任务栏）
            screen_geometry = screen.geometry()
            self.setFixedSize(screen_geometry.size())
            self.move(0, 0)
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        else:
            # 自适应模式（保留任务栏）
            available_geometry = screen.availableGeometry()
            self.setFixedSize(available_geometry.size())
            self.move(available_geometry.x(), available_geometry.y())
        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.CustomizeWindowHint | 
            Qt.WindowType.WindowTitleHint | 
            Qt.WindowType.WindowSystemMenuHint | 
            Qt.WindowType.WindowMinimizeButtonHint | 
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # 禁用 QFluentWidgets 自定义标题栏的最大化/还原功能
        if hasattr(self, 'titleBar'):
            if hasattr(self.titleBar, 'maxBtn'):
                self.titleBar.maxBtn.hide()
                self.titleBar.maxBtn.setDisabled(True)
            if hasattr(self.titleBar, 'setDoubleClickEnabled'):
                self.titleBar.setDoubleClickEnabled(False)
        
        # 导航栏行为设定 (参考 Eye_Touch 的紧凑折叠实现)
        self.navigationInterface.setExpandWidth(130)
        try:
            if hasattr(self.navigationInterface, 'setCollapsible'):
                self.navigationInterface.setCollapsible(True)
            if hasattr(self.navigationInterface, 'setMenuButtonMinimumWidth'):
                self.navigationInterface.setMenuButtonMinimumWidth(80)
            if hasattr(self.navigationInterface, 'setCollapseWidth'):
                self.navigationInterface.setCollapseWidth(32)
        except Exception:
            pass
        
        # 默认选中第一项
        self.navigationInterface.setCurrentItem(self.page_home.objectName())
        
        print("[MAIN_WINDOW] 主窗口已初始化，窗口大小已自适应屏幕")
    
    def _center_window(self) -> None:
        """将窗口居中显示。"""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_geometry = screen.geometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
    
    def _ensure_tracker_initialized(self) -> bool:
        """确保 TrackerPipeline 已初始化。
        
        返回:
            初始化是否成功
        """
        if self.tracker is not None:
            return True
        
        # 加载系统配置
        config_path = Path("configs/system_config.yaml")
        if config_path.exists():
            try:
                self.tracker_config = SystemConfig.from_yaml(str(config_path))
                print(f"[MAIN_WINDOW] 系统配置已加载：{config_path}")
            except Exception as e:
                print(f"[MAIN_WINDOW] 加载配置失败：{e}，使用默认配置")
                self.tracker_config = SystemConfig()
        else:
            print("[MAIN_WINDOW] 配置文件不存在，使用默认配置")
            self.tracker_config = SystemConfig()
        
        # 创建 TrackerPipeline
        try:
            self.tracker = TrackerPipeline(
                model_path=self.tracker_config.checkpoint_path,
                config=self.tracker_config,
                error_callback=self._on_tracker_error
            )
            print("[MAIN_WINDOW] TrackerPipeline 已创建")
            return True
            
        except Exception as e:
            print(f"[MAIN_WINDOW] TrackerPipeline 创建失败：{e}")
            QMessageBox.critical(
                self,
                tx("初始化失败", "Initialization Failed"),
                tx(
                    f"TrackerPipeline 初始化失败：{e}\n\n请检查模型文件和配置。",
                    f"TrackerPipeline initialization failed: {e}\n\nPlease check model files and settings.",
                )
            )
            return False
    
    def _on_tracker_error(self, message: str) -> None:
        """TrackerPipeline 错误回调。
        
        参数:
            message: 错误消息
        """
        print(f"[MAIN_WINDOW] TrackerPipeline 错误: {message}")
    
    def _on_config_changed(self, config: SystemConfig) -> None:
        """设置页面配置变更回调。
        
        参数:
            config: 新的系统配置
        """
        print("[MAIN_WINDOW] 系统配置已变更")
        self.tracker_config = config
        
        # 如果 TrackerPipeline 正在运行，提示用户重启
        if self.tracker is not None and self.tracker.is_running():
            QMessageBox.information(
                self,
                tx("配置已变更", "Settings Changed"),
                tx(
                    "系统配置已更新。\n\n部分配置需要重启追踪才能生效。",
                    "System settings have been updated.\n\nSome changes require restarting tracking.",
                )
            )

    def _on_calibration_ready(self) -> None:
        """用户保存校准后进入独立验证阶段。"""
        self.go_tracking()
        self.page_tracking.start_verification_flow()
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """按键事件处理：按 Esc 键返回主页或退出。"""
        if event.key() == Qt.Key.Key_Escape:
            # 如果在主页，则退出应用
            if self.stackedWidget.currentWidget() == self.page_home:
                self.close()
            else:
                # 否则返回主页
                self.go_home()
            return
        super().keyPressEvent(event)
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """窗口关闭事件：释放所有资源。"""
        print("[MAIN_WINDOW] 正在关闭应用...")
        
        # 停止 TrackerPipeline
        if self.tracker is not None:
            print("[MAIN_WINDOW] 正在停止 TrackerPipeline...")
            self.tracker.stop()
            self.tracker = None
        
        # 关闭所有页面（释放摄像头等资源）
        self.page_camera.close()
        self.page_tracking.close()
        self.page_calibration.close()
        
        print("[MAIN_WINDOW] 应用已关闭")
        event.accept()
    
    # 页面导航方法
    
    def go_home(self) -> None:
        """导航到主页。"""
        self.switchTo(self.page_home)
        print("[MAIN_WINDOW] 导航到主页")
    
    def go_camera(self) -> None:
        """导航到摄像头预览页面。"""
        self.switchTo(self.page_camera)
        print("[MAIN_WINDOW] 导航到摄像头预览页面")
    
    def go_calibration(self) -> None:
        """导航到校准页面。"""
        # 确保 TrackerPipeline 已初始化
        if not self._ensure_tracker_initialized():
            return
        
        # 如果 TrackerPipeline 未运行，启动它
        if not self.tracker.is_running():
            print("[MAIN_WINDOW] 启动 TrackerPipeline 用于校准...")
            success = self.tracker.start()
            if not success:
                QMessageBox.critical(
                    self,
                    tx("启动失败", "Start Failed"),
                    tx(
                        "TrackerPipeline 启动失败，无法进行校准。\n\n请检查摄像头和模型文件。",
                        "TrackerPipeline failed to start, so calibration cannot begin.\n\nPlease check the camera and model files.",
                    )
                )
                return
        
        # 将 TrackerPipeline 传递给校准页面
        self.page_calibration.set_tracker(self.tracker)
        
        self.switchTo(self.page_calibration)
        print("[MAIN_WINDOW] 导航到校准页面")
    
    def go_tracking(self) -> None:
        """导航到实时追踪页面。"""
        # 确保 TrackerPipeline 已初始化
        if not self._ensure_tracker_initialized():
            return
        
        # 将 TrackerPipeline 传递给追踪页面
        if self.page_tracking.tracker is None:
            self.page_tracking.tracker = self.tracker
            self.page_tracking.tracker_config = self.tracker_config
        else:
            self.page_tracking.tracker = self.tracker
            self.page_tracking.tracker_config = self.tracker_config

        if self.page_calibration.calibrator.is_calibrated:
            self.page_tracking.set_calibrator(self.page_calibration.calibrator)
        
        self.switchTo(self.page_tracking)
        print("[MAIN_WINDOW] 导航到实时追踪页面")
    
    def go_settings(self) -> None:
        """导航到设置页面。"""
        self.switchTo(self.page_settings)
        print("[MAIN_WINDOW] 导航到设置页面")

