"""
数据管道 Property-Based Tests 与单元测试。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python -m pytest Look2Act_Tracker_Project/tests/test_data_pipeline.py -v
"""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from data.pipeline import (
    DataPipeline,
    REQUIRED_COLUMNS,
    PROCESSED_COLUMNS,
    SessionData,
)


# ============ hypothesis strategies ============

def _valid_flag_strategy():
    """生成 valid 字段值：0 或 1。"""
    return st.sampled_from([0, 1])


def _labels_row_strategy(user_id: int, session_id: str):
    """生成单行 labels.csv 数据的 strategy。"""
    return st.fixed_dictionaries({
        "user_id": st.just(user_id),
        "session_id": st.just(session_id),
        "frame_idx": st.integers(min_value=0, max_value=9999),
        "timestamp_ms": st.integers(min_value=0, max_value=999999),
        "img_path": st.just("frames_private/000000.jpg"),
        "target_id": st.integers(min_value=0, max_value=25),
        "target_x": st.floats(min_value=0, max_value=1920, allow_nan=False),
        "target_y": st.floats(min_value=0, max_value=1080, allow_nan=False),
        "target_elapsed_ms": st.integers(min_value=0, max_value=5000),
        "valid": _valid_flag_strategy(),
        "face_score": st.floats(min_value=0, max_value=1, allow_nan=False),
        "lm_score": st.floats(min_value=0, max_value=1, allow_nan=False),
        "head_yaw": st.floats(min_value=-180, max_value=180, allow_nan=False),
        "head_pitch": st.floats(min_value=-180, max_value=180, allow_nan=False),
        "head_roll": st.floats(min_value=-180, max_value=180, allow_nan=False),
        "distance_proxy": st.floats(min_value=1, max_value=50, allow_nan=False),
        "illumination_mean": st.floats(min_value=0, max_value=255, allow_nan=False),
        "device_id": st.just("TEST-DEVICE"),
        "camera_name": st.just("Camera_0"),
        "glass_id": st.sampled_from(["glass", "no_glass"]),
        "screen_w": st.just(1536),
        "screen_h": st.just(864),
        "frame_w": st.just(1920),
        "frame_h": st.just(1080),
    })


@st.composite
def labels_dataframe_strategy(draw):
    """生成包含随机行数的 labels DataFrame。"""
    n_rows = draw(st.integers(min_value=1, max_value=30))
    user_id = draw(st.integers(min_value=1, max_value=100))
    session_id = f"S_TEST_{user_id}"

    rows = [draw(_labels_row_strategy(user_id, session_id)) for _ in range(n_rows)]

    # 确保 frame_idx 唯一
    for i, row in enumerate(rows):
        row["frame_idx"] = i

    # valid=0 的行 img_path 可以为空（模拟真实数据）
    for row in rows:
        if row["valid"] == 0:
            # 50% 概率 img_path 为空
            if draw(st.booleans()):
                row["img_path"] = None

    return pd.DataFrame(rows)


def _create_meta_json(user_id: int, session_id: str) -> dict:
    """创建最小 meta.json 内容。"""
    return {
        "user_id": str(user_id),
        "session_id": session_id,
        "glass_id": "glass",
        "device_id": "TEST-DEVICE",
        "camera_name": "Camera_0",
        "screen_w": 1536,
        "screen_h": 864,
        "frame_w": 1920,
        "frame_h": 1080,
        "grid": {"rows": 5, "cols": 5},
        "eye_crop_size": 128,
        "distance_init_proxy": 13.0,
    }


# ============ Property 1：数据管道字段完整性 ============
# Feature: look2act-tracker, Property 1: 数据管道字段完整性
#
# 对于任意有效的原始 Session 目录，DataPipeline 解析 labels.csv 后：
# 1. 能正确解析全部 24 个必需字段
# 2. get_valid_labels() 返回的行数 == valid=1 的行数
# 3. valid=0 的行被正确过滤
#
# **Validates: Requirements 1.1, 1.3, 1.4**


@given(df=labels_dataframe_strategy())
@settings(max_examples=100)
def test_property1_pipeline_field_completeness(df: pd.DataFrame, tmp_path_factory):
    """Property 1：数据管道字段完整性。

    验证 DataPipeline.load_session() 能正确解析随机生成的 labels.csv，
    且 get_valid_labels() 返回的行数与 valid=1 的行数一致。
    """
    # 准备临时 Session 目录
    tmp_dir = tmp_path_factory.mktemp("session")
    session_dir = tmp_dir / "test_session"
    session_dir.mkdir()

    # 写入 labels.csv
    df.to_csv(session_dir / "labels.csv", index=False)

    # 写入 meta.json
    user_id = int(df["user_id"].iloc[0])
    session_id = str(df["session_id"].iloc[0])
    meta = _create_meta_json(user_id, session_id)
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # 加载 Session
    pipeline = DataPipeline()
    session = pipeline.load_session(session_dir)

    # 验证 1：labels 包含全部 24 个必需字段
    for col in REQUIRED_COLUMNS:
        assert col in session.labels.columns, f"缺少必需字段: {col}"
    assert len(session.labels.columns) >= 24, (
        f"字段数不足 24，实际为 {len(session.labels.columns)}"
    )

    # 验证 2：valid_count == valid=1 的行数
    expected_valid = int((df["valid"] == 1).sum())
    assert session.valid_count == expected_valid, (
        f"valid_count 不匹配: 期望 {expected_valid}, 实际 {session.valid_count}"
    )

    # 验证 3：get_valid_labels() 返回的行数 == valid=1 的行数
    valid_labels = pipeline.get_valid_labels(session)
    assert len(valid_labels) == expected_valid, (
        f"get_valid_labels() 行数不匹配: 期望 {expected_valid}, 实际 {len(valid_labels)}"
    )

    # 验证 4：skipped_count == valid=0 的行数
    expected_skipped = int((df["valid"] == 0).sum())
    assert session.skipped_count == expected_skipped, (
        f"skipped_count 不匹配: 期望 {expected_skipped}, 实际 {session.skipped_count}"
    )

    # 验证 5：valid_labels 中所有行的 valid 字段都为 1
    if len(valid_labels) > 0:
        assert (valid_labels["valid"] == 1).all(), (
            "get_valid_labels() 返回了 valid=0 的行"
        )


# ============ 单元测试：边界情况 ============

def test_load_session_missing_labels(tmp_path):
    """labels.csv 不存在时应抛出 FileNotFoundError。"""
    session_dir = tmp_path / "empty_session"
    session_dir.mkdir()
    meta = _create_meta_json(1, "S_TEST")
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    pipeline = DataPipeline()
    with pytest.raises(FileNotFoundError, match="labels.csv"):
        pipeline.load_session(session_dir)


def test_load_session_missing_meta(tmp_path):
    """meta.json 不存在时应抛出 FileNotFoundError。"""
    session_dir = tmp_path / "no_meta_session"
    session_dir.mkdir()
    df = pd.DataFrame([{col: 0 for col in REQUIRED_COLUMNS}])
    df.to_csv(session_dir / "labels.csv", index=False)

    pipeline = DataPipeline()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        pipeline.load_session(session_dir)


def test_load_session_missing_columns(tmp_path):
    """labels.csv 缺少必需字段时应抛出 ValueError。"""
    session_dir = tmp_path / "bad_columns_session"
    session_dir.mkdir()

    # 只写入部分字段
    df = pd.DataFrame({"user_id": [1], "session_id": ["S_TEST"]})
    df.to_csv(session_dir / "labels.csv", index=False)

    meta = _create_meta_json(1, "S_TEST")
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    pipeline = DataPipeline()
    with pytest.raises(ValueError, match="缺少必需字段"):
        pipeline.load_session(session_dir)


def test_load_session_all_valid(tmp_path):
    """所有行 valid=1 时，skipped_count 应为 0。"""
    session_dir = tmp_path / "all_valid_session"
    session_dir.mkdir()

    row = {col: 0 for col in REQUIRED_COLUMNS}
    row.update({
        "user_id": 1, "session_id": "S_TEST", "valid": 1,
        "img_path": "frames_private/000000.jpg",
        "device_id": "DEV", "camera_name": "CAM", "glass_id": "glass",
        "screen_w": 1536, "screen_h": 864, "frame_w": 1920, "frame_h": 1080,
    })
    df = pd.DataFrame([row] * 5)
    df.to_csv(session_dir / "labels.csv", index=False)

    meta = _create_meta_json(1, "S_TEST")
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    pipeline = DataPipeline()
    session = pipeline.load_session(session_dir)
    assert session.valid_count == 5
    assert session.skipped_count == 0


def test_load_session_all_invalid(tmp_path):
    """所有行 valid=0 时，valid_count 应为 0。"""
    session_dir = tmp_path / "all_invalid_session"
    session_dir.mkdir()

    row = {col: 0 for col in REQUIRED_COLUMNS}
    row.update({
        "user_id": 1, "session_id": "S_TEST", "valid": 0,
        "img_path": "frames_private/000000.jpg",
        "device_id": "DEV", "camera_name": "CAM", "glass_id": "glass",
        "screen_w": 1536, "screen_h": 864, "frame_w": 1920, "frame_h": 1080,
    })
    df = pd.DataFrame([row] * 3)
    df.to_csv(session_dir / "labels.csv", index=False)

    meta = _create_meta_json(1, "S_TEST")
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    pipeline = DataPipeline()
    session = pipeline.load_session(session_dir)
    assert session.valid_count == 0
    assert session.skipped_count == 3
