"""
最小共存测试：仅 import
验证 PyQt6 + PyQt6-Fluent-Widgets + mediapipe 能否在同一进程中导入。
"""
import sys
import traceback


def main():
    print(f"Python: {sys.executable}")
    errors = []

    # 1. PyQt6 核心
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import PYQT_VERSION_STR
        print(f"[OK] PyQt6 imported, version={PYQT_VERSION_STR}")
    except Exception as e:
        errors.append(("PyQt6", e))
        print(f"[FAIL] PyQt6: {e}")
        traceback.print_exc()

    # 2. PyQt6-Fluent-Widgets
    try:
        import qfluentwidgets
        print(f"[OK] qfluentwidgets imported, version={getattr(qfluentwidgets, '__version__', 'unknown')}")
    except Exception as e:
        errors.append(("qfluentwidgets", e))
        print(f"[FAIL] qfluentwidgets: {e}")
        traceback.print_exc()

    # 3. mediapipe
    try:
        import mediapipe as mp
        print(f"[OK] mediapipe imported, version={mp.__version__}")
    except Exception as e:
        errors.append(("mediapipe", e))
        print(f"[FAIL] mediapipe: {e}")
        traceback.print_exc()

    # 总结
    if not errors:
        print("\n=== 结论：所有 import 成功，无冲突 ===")
        return 0
    else:
        print(f"\n=== 结论：{len(errors)} 个模块导入失败 ===")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
