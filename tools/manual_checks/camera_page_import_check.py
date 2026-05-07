"""测试摄像头预览页面导入。

验证模块可以正确导入，不启动 GUI。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.ui.camera_page import CameraPage
    from src.ui.camera_stream import CameraStream, Resolution
    print("✓ 模块导入成功")
    print(f"✓ CameraPage 类: {CameraPage}")
    print(f"✓ CameraStream 类: {CameraStream}")
    print(f"✓ Resolution 类: {Resolution}")
    print("\n所有导入测试通过！")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
