"""
最小共存测试：Qt event loop + mediapipe 初始化
验证在 Qt 事件循环运行时，mediapipe FaceMesh 能否正常工作。
"""
import sys
import traceback
import numpy as np


def run_mediapipe_test():
    """在 Qt 事件循环内执行 mediapipe FaceMesh 推理。"""
    try:
        import mediapipe as mp

        # 初始化 FaceMesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # 生成随机 BGR 图像（480x640x3）
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 转 RGB 并推理
        rgb = fake_frame[:, :, ::-1]
        result = face_mesh.process(rgb)

        face_mesh.close()

        # 随机图像大概率检测不到人脸，但不应崩溃
        detected = result.multi_face_landmarks is not None
        print(f"[OK] mediapipe FaceMesh 推理完成，检测到人脸: {detected}")
        print("OK: Qt loop running + mediapipe process success")

        from PyQt6.QtWidgets import QApplication
        QApplication.quit()

    except Exception as e:
        print(f"[FAIL] mediapipe 推理失败: {e}")
        traceback.print_exc()
        from PyQt6.QtWidgets import QApplication
        QApplication.exit(1)


def main():
    from PyQt6.QtWidgets import QApplication, QLabel
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)

    # 最小窗口
    label = QLabel("兼容性测试中...")
    label.resize(300, 100)
    label.show()

    # 在事件循环启动后立即执行 mediapipe 测试
    QTimer.singleShot(0, run_mediapipe_test)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
