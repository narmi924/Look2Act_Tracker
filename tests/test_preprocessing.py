"""
数据预处理 Property-Based Tests 与单元测试。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python -m pytest tests/test_preprocessing.py -v

测试覆盖：
- Property 2: 坐标归一化 round-trip
- Property 3: 眼部裁剪尺寸不变量（基于 extract_eyes_from_mosaic）
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from data.preprocessing import (
    extract_eyes_from_mosaic,
    normalize_screen_coords,
    denormalize_screen_coords,
    DEFAULT_EYE_CROP_SIZE,
)


# ============ hypothesis strategies ============

@st.composite
def mosaic_strategy(draw):
    """生成模拟 Collector 合成图的 strategy。

    合成图布局：640 宽，高度 128 或 240。
    前两个 128x128 tile 为左右眼区域（非全黑）。
    """
    height = draw(st.sampled_from([128, 240]))
    width = 640
    # 生成非全黑的合成图（确保眼部区域有内容）
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    # 左眼区域填充随机像素（均值 > 1.0 以通过全黑检测）
    left_eye = draw(
        st.integers(min_value=30, max_value=255)
    )
    right_eye = draw(
        st.integers(min_value=30, max_value=255)
    )
    mosaic[:128, :128] = left_eye
    mosaic[:128, 128:256] = right_eye
    # 添加一些随机噪声使其更真实
    noise = np.random.RandomState(draw(st.integers(0, 9999))).randint(
        0, 30, (128, 128, 3), dtype=np.uint8
    )
    mosaic[:128, :128] = np.clip(
        mosaic[:128, :128].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    noise2 = np.random.RandomState(draw(st.integers(0, 9999))).randint(
        0, 30, (128, 128, 3), dtype=np.uint8
    )
    mosaic[:128, 128:256] = np.clip(
        mosaic[:128, 128:256].astype(np.int16) + noise2, 0, 255
    ).astype(np.uint8)
    return mosaic


def screen_coord_strategy():
    """生成屏幕像素坐标和屏幕尺寸的 strategy。"""
    return st.tuples(
        st.floats(min_value=0, max_value=7680, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=4320, allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=7680),
        st.integers(min_value=1, max_value=4320),
    )


# ============ Property 3：眼部裁剪尺寸不变量 ============
# Feature: look2act-tracker, Property 3: 眼部裁剪尺寸不变量
#
# 对于任意包含有效眼部数据的 Collector 合成图（任意高度 128/240），
# extract_eyes_from_mosaic 输出的左右眼图像尺寸必须恒为 128×128×3。
#
# **Validates: Requirements 1.2, 1.7, 2.2**


@given(mosaic=mosaic_strategy())
@settings(max_examples=100)
def test_property3_eye_crop_size_invariant(mosaic: np.ndarray):
    """Property 3：眼部裁剪尺寸不变量。

    验证 extract_eyes_from_mosaic 对任意合法合成图输出恒为 128×128×3。
    """
    left_eye, right_eye = extract_eyes_from_mosaic(mosaic)

    # 眼部区域非全黑，应成功提取
    assert left_eye is not None, "左眼提取失败（非全黑输入）"
    assert right_eye is not None, "右眼提取失败（非全黑输入）"

    # 尺寸不变量
    assert left_eye.shape == (128, 128, 3), (
        f"左眼尺寸错误: {left_eye.shape}，期望 (128, 128, 3)"
    )
    assert right_eye.shape == (128, 128, 3), (
        f"右眼尺寸错误: {right_eye.shape}，期望 (128, 128, 3)"
    )

    # 数据类型
    assert left_eye.dtype == np.uint8
    assert right_eye.dtype == np.uint8


# ============ Property 2：坐标归一化 round-trip ============
# Feature: look2act-tracker, Property 2: 坐标归一化 round-trip
#
# 对于任意屏幕像素坐标 (target_x, target_y) 和屏幕尺寸 (screen_w, screen_h)，
# 归一化后再反归一化应还原原始像素坐标（允许浮点误差 1e-6）。
#
# **Validates: Requirements 1.7**


@given(data=screen_coord_strategy())
@settings(max_examples=100)
def test_property2_normalize_round_trip(data):
    """Property 2：坐标归一化 round-trip。

    验证 normalize → denormalize 的可逆性。
    """
    target_x, target_y, screen_w, screen_h = data

    # 归一化
    norm_x, norm_y = normalize_screen_coords(target_x, target_y, screen_w, screen_h)

    # 反归一化
    restored_x, restored_y = denormalize_screen_coords(norm_x, norm_y, screen_w, screen_h)

    # round-trip 精度验证
    assert abs(restored_x - target_x) < 1e-6, (
        f"X 坐标 round-trip 失败: {target_x} -> {norm_x} -> {restored_x}"
    )
    assert abs(restored_y - target_y) < 1e-6, (
        f"Y 坐标 round-trip 失败: {target_y} -> {norm_y} -> {restored_y}"
    )



# ============ 单元测试：边界情况 ============

def test_extract_eyes_none_input():
    """输入 None 时应返回 (None, None)。"""
    le, re = extract_eyes_from_mosaic(None)
    assert le is None and re is None


def test_extract_eyes_empty_input():
    """输入空数组时应返回 (None, None)。"""
    le, re = extract_eyes_from_mosaic(np.array([]))
    assert le is None and re is None


def test_extract_eyes_too_small():
    """输入尺寸不足时应返回 (None, None)。"""
    small = np.zeros((50, 100, 3), dtype=np.uint8)
    le, re = extract_eyes_from_mosaic(small)
    assert le is None and re is None


def test_extract_eyes_all_black():
    """全黑合成图（Collector 未检测到眼部）应返回 (None, None)。"""
    black = np.zeros((128, 640, 3), dtype=np.uint8)
    le, re = extract_eyes_from_mosaic(black)
    assert le is None and re is None


def test_normalize_invalid_screen_size():
    """屏幕尺寸为 0 或负数时应抛出 ValueError。"""
    with pytest.raises(ValueError):
        normalize_screen_coords(100, 100, 0, 864)
    with pytest.raises(ValueError):
        normalize_screen_coords(100, 100, 1536, -1)
    with pytest.raises(ValueError):
        denormalize_screen_coords(0.5, 0.5, 0, 864)


def test_normalize_center_point():
    """屏幕中心点归一化后应为 (0.5, 0.5)。"""
    nx, ny = normalize_screen_coords(768, 432, 1536, 864)
    assert abs(nx - 0.5) < 1e-9
    assert abs(ny - 0.5) < 1e-9
