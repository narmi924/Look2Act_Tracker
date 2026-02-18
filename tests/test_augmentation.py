"""
数据增强 Property-Based Tests 与单元测试。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python -m pytest tests/test_augmentation.py -v

测试覆盖：
- Property 4: 数据增强标签一致性（水平翻转）
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data.augmentation import (
    horizontal_flip,
    brightness_jitter,
    slight_rotation,
    augment_sample,
)


# ============ hypothesis strategies ============

@st.composite
def eye_img_strategy(draw):
    """生成随机 128×128 BGR 眼部图像。"""
    return np.random.RandomState(
        draw(st.integers(0, 99999))
    ).randint(0, 256, (128, 128, 3), dtype=np.uint8)


@st.composite
def unit_gaze_vector_strategy(draw):
    """生成随机 3D 单位视线向量。"""
    # 生成非零随机向量后归一化
    x = draw(st.floats(min_value=-1, max_value=1, allow_nan=False))
    y = draw(st.floats(min_value=-1, max_value=1, allow_nan=False))
    z = draw(st.floats(min_value=0.1, max_value=1, allow_nan=False))  # z > 0（朝前）
    norm = np.sqrt(x**2 + y**2 + z**2)
    return float(x / norm), float(y / norm), float(z / norm)


# ============ Property 4：数据增强标签一致性（水平翻转） ============
# Feature: look2act-tracker, Property 4: 数据增强标签一致性（水平翻转）
#
# 对于任意眼部图像和对应的 3D 视线向量 (gx, gy, gz)，
# 水平翻转后：gx_flipped ≈ -gx，gy_flipped ≈ gy，gz_flipped ≈ gz，
# 且翻转后的向量仍为单位向量。
#
# **Validates: Requirements 1.5**


@given(img=eye_img_strategy(), gaze=unit_gaze_vector_strategy())
@settings(max_examples=100)
def test_property4_horizontal_flip_label_consistency(img, gaze):
    """Property 4：水平翻转后视线标签一致性。"""
    gx, gy, gz = gaze
    result = horizontal_flip(img, gx, gy, gz)

    # gaze_x 取反
    assert abs(result.gaze_x - (-gx)) < 1e-10, (
        f"gaze_x 翻转错误: 期望 {-gx}, 实际 {result.gaze_x}"
    )
    # gaze_y 不变
    assert abs(result.gaze_y - gy) < 1e-10, (
        f"gaze_y 应不变: 期望 {gy}, 实际 {result.gaze_y}"
    )
    # gaze_z 不变
    assert abs(result.gaze_z - gz) < 1e-10, (
        f"gaze_z 应不变: 期望 {gz}, 实际 {result.gaze_z}"
    )

    # 翻转后仍为单位向量
    norm = np.sqrt(result.gaze_x**2 + result.gaze_y**2 + result.gaze_z**2)
    assert abs(norm - 1.0) < 1e-6, (
        f"翻转后不是单位向量: norm={norm}"
    )

    # 图像确实被翻转了
    assert result.eye_img.shape == img.shape
    # 验证翻转：第一列应等于原图最后一列
    np.testing.assert_array_equal(
        result.eye_img[:, 0, :], img[:, -1, :]
    )


# ============ 单元测试 ============

def test_brightness_jitter_preserves_labels():
    """亮度抖动不改变视线标签。"""
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    result = brightness_jitter(img, 0.3, 0.2, 0.9, rng=np.random.RandomState(42))
    assert result.gaze_x == 0.3
    assert result.gaze_y == 0.2
    assert result.gaze_z == 0.9
    assert result.eye_img.shape == (128, 128, 3)


def test_slight_rotation_preserves_labels():
    """轻微旋转不改变视线标签。"""
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    result = slight_rotation(img, 0.1, 0.2, 0.97, rng=np.random.RandomState(0))
    assert result.gaze_x == 0.1
    assert result.gaze_y == 0.2
    assert result.gaze_z == 0.97
    assert result.eye_img.shape == (128, 128, 3)


def test_brightness_clamp():
    """亮度抖动后像素值应在 [0, 255] 范围内。"""
    img = np.full((128, 128, 3), 250, dtype=np.uint8)
    result = brightness_jitter(img, 0, 0, 1, delta_range=(20, 20))
    assert result.eye_img.max() <= 255
    assert result.eye_img.min() >= 0

    img2 = np.full((128, 128, 3), 5, dtype=np.uint8)
    result2 = brightness_jitter(img2, 0, 0, 1, delta_range=(-20, -20))
    assert result2.eye_img.min() >= 0


def test_augment_sample_deterministic():
    """相同 rng 种子应产生相同结果。"""
    img = np.random.RandomState(0).randint(0, 256, (128, 128, 3), dtype=np.uint8)
    r1 = augment_sample(img, 0.3, 0.2, 0.9, rng=np.random.RandomState(42))
    r2 = augment_sample(img, 0.3, 0.2, 0.9, rng=np.random.RandomState(42))
    np.testing.assert_array_equal(r1.eye_img, r2.eye_img)
    assert r1.gaze_x == r2.gaze_x
    assert r1.gaze_y == r2.gaze_y
    assert r1.gaze_z == r2.gaze_z
