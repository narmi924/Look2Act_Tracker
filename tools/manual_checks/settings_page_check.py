"""测试设置页面的基本功能。

运行方式：
conda run -n gaze-env python test_settings_page.py
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ui.settings_page import SettingsPage


def main():
    """主函数。"""
    app = QApplication(sys.argv)
    
    # 创建设置页面
    settings_page = SettingsPage()
    
    # 连接配置变更信号
    def on_config_changed(config):
        print(f"\n[TEST] 配置已变更:")
        print(f"  摄像头索引: {config.camera_index}")
        print(f"  摄像头分辨率: {config.camera_width}x{config.camera_height}")
        print(f"  模型路径: {config.checkpoint_path}")
        print(f"  IPEX 优化: {config.use_ipex}")
        print(f"  ONNX Runtime: {config.use_onnx}")
        print(f"  屏幕尺寸: {config.screen_w_mm}x{config.screen_h_mm} mm")
        print(f"  平滑系数: {config.smoother_alpha}")
        print(f"  目标帧率: {config.target_fps}")
    
    settings_page.config_changed.connect(on_config_changed)
    
    # 显示窗口
    settings_page.setWindowTitle("Look2Act Tracker - 设置页面测试")
    settings_page.resize(1000, 800)
    settings_page.show()
    
    print("[TEST] 设置页面已启动")
    print("[TEST] 请在 UI 中修改设置并点击保存按钮")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
