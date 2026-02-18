"""
快速验证 EyeCropper 在真实数据上的表现。
工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
"""
import sys
from pathlib import Path

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from data.preprocessing import EyeCropper, normalize_screen_coords, denormalize_screen_coords

def main():
    # 找到第一个可用的 Session 图像
    raw_dir = Path(__file__).resolve().parent.parent / "dataset_raw"
    test_img = None
    for session_dir in sorted(raw_dir.iterdir()):
        frames_dir = session_dir / "frames_private"
        if frames_dir.exists():
            for img_file in sorted(frames_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".png"):
                    test_img = img_file
                    break
        if test_img:
            break

    if test_img is None:
        print("错误：未找到测试图像")
        return

    print(f"测试图像: {test_img}")
    frame = cv2.imread(str(test_img))
    print(f"图像尺寸: {frame.shape}")

    # 测试 EyeCropper
    cropper = EyeCropper(eye_crop_size=128)
    result = cropper.crop_eyes(frame)
    cropper.close()

    print(f"\n检测成功: {result.success}")
    if result.success:
        print(f"左眼裁剪尺寸: {result.left_eye.shape}")
        print(f"右眼裁剪尺寸: {result.right_eye.shape}")
        assert result.left_eye.shape == (128, 128, 3), f"左眼尺寸错误: {result.left_eye.shape}"
        assert result.right_eye.shape == (128, 128, 3), f"右眼尺寸错误: {result.right_eye.shape}"
        print(f"PnP 关键点数: {len(result.pnp_points_2d)}")
        print(f"人脸边界框: {result.face_bbox}")
        print("✓ 眼部裁剪尺寸验证通过 (128x128x3)")
    else:
        print("✗ 人脸检测失败")

    # 测试坐标归一化 round-trip
    print("\n--- 坐标归一化 round-trip 测试 ---")
    tx, ty = 768.5, 432.25
    sw, sh = 1536, 864
    nx, ny = normalize_screen_coords(tx, ty, sw, sh)
    rx, ry = denormalize_screen_coords(nx, ny, sw, sh)
    print(f"原始: ({tx}, {ty})")
    print(f"归一化: ({nx:.6f}, {ny:.6f})")
    print(f"还原: ({rx:.6f}, {ry:.6f})")
    assert abs(rx - tx) < 1e-6 and abs(ry - ty) < 1e-6, "round-trip 失败"
    print("✓ 坐标归一化 round-trip 验证通过")

    print("\n全部验证通过")

if __name__ == "__main__":
    main()
