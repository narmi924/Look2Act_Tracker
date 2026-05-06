"""测试设置页面的配置管理功能。

不涉及 GUI 交互，仅测试配置的加载、保存和转换逻辑。
"""
import tempfile
from pathlib import Path

import pytest
import yaml
import numpy as np

from main import parse_args
from src.tracker.pipeline import SystemConfig
from src.ui.settings_page import (
    detect_supported_camera_resolutions,
    format_resolution,
    parse_resolution,
    sort_resolutions,
)


def test_system_config_default_values():
    """测试 SystemConfig 的默认值。"""
    config = SystemConfig()
    
    assert config.camera_index == 0
    assert config.camera_width == 640
    assert config.camera_height == 480
    assert config.camera_backend == "dshow"
    assert config.eye_crop_size == 128
    assert config.checkpoint_path == "checkpoints/best_model.pth"
    assert config.use_ipex is False
    assert config.use_onnx is False
    assert config.normalized_backend == "classic"
    assert config.calibration_path == "calibration_classic.json"
    assert config.deep_gaze_space == "head"
    assert config.normalized_deep_pose_input == "live"
    assert config.normalized_smoother_type == "kalman"
    assert config.screen_w_mm == 344.0
    assert config.screen_h_mm == 194.0
    assert config.smoother_alpha == 0.3
    assert config.target_fps == 30


def test_system_config_backend_paths():
    """测试 tracker backend 与校准文件路径分离。"""
    classic = SystemConfig(tracker_backend="classic")
    deep_pog = SystemConfig(tracker_backend="deep_pog")
    deep = SystemConfig(tracker_backend="deep")
    unknown = SystemConfig(tracker_backend="bad")

    assert classic.normalized_backend == "classic"
    assert classic.calibration_path == "calibration_classic.json"
    assert deep_pog.normalized_backend == "deep_pog"
    assert deep_pog.calibration_path == "calibration_deep_pog.json"
    assert deep.normalized_backend == "deep"
    assert deep.calibration_path == "calibration_deep.json"
    assert unknown.normalized_backend == "classic"
    assert unknown.calibration_path == "calibration_classic.json"


def test_system_config_smoother_type_normalization():
    """测试平滑器类型归一化。"""
    assert SystemConfig(smoother_type="kalman").normalized_smoother_type == "kalman"
    assert SystemConfig(smoother_type="ema").normalized_smoother_type == "ema"
    assert SystemConfig(smoother_type="none").normalized_smoother_type == "none"
    assert SystemConfig(smoother_type="bad").normalized_smoother_type == "kalman"


def test_main_config_argument_selects_experiment_yaml():
    args = parse_args(["--config", "configs/experiments/system_deep_camera_zero_720.yaml"])

    assert args.config == "configs/experiments/system_deep_camera_zero_720.yaml"


def test_resolution_format_parse_and_sort():
    assert format_resolution(1280, 720) == "1280x720"
    assert parse_resolution("1920 x 1080") == (1920, 1080)
    assert sort_resolutions([(1920, 1080), (640, 480), (640, 480)]) == [
        (640, 480),
        (1920, 1080),
    ]


def test_detect_supported_camera_resolutions_uses_actual_frame_size():
    import cv2

    class FakeCapture:
        def __init__(self, *args):
            self.width = 640
            self.height = 480
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = int(value)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = int(value)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return self.width
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.height
            return 0

        def read(self):
            if (self.width, self.height) == (1920, 1080):
                return True, np.zeros((720, 1280, 3), dtype=np.uint8)
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    detected = detect_supported_camera_resolutions(
        0,
        "auto",
        candidates=((640, 480), (1920, 1080)),
        capture_factory=lambda *args: FakeCapture(*args),
    )

    assert detected == [(640, 480), (1280, 720)]


def test_system_config_deep_pose_input_normalization():
    """测试 deep pose input 归一化。"""
    assert SystemConfig(deep_pose_input="live").normalized_deep_pose_input == "live"
    assert SystemConfig(deep_pose_input="zero").normalized_deep_pose_input == "zero"
    assert SystemConfig(deep_pose_input="bad").normalized_deep_pose_input == "live"


def test_system_config_deep_ray_origin_normalization():
    """测试 deep ray origin 归一化。"""
    assert SystemConfig(deep_ray_origin="face_translation").normalized_deep_ray_origin == "face_translation"
    assert SystemConfig(deep_ray_origin="zero_origin").normalized_deep_ray_origin == "zero_origin"
    assert SystemConfig(deep_ray_origin="bad").normalized_deep_ray_origin == "face_translation"


def test_system_config_from_yaml():
    """测试从 YAML 文件加载配置。"""
    # 创建临时 YAML 文件
    config_data = {
        'camera': {
            'index': 1,
            'width': 1280,
            'height': 720,
            'backend': 'auto',
        },
        'face_detection': {
            'eye_crop_size': 128,
            'min_detection_confidence': 0.6,
            'min_tracking_confidence': 0.6,
        },
        'model': {
            'checkpoint_path': 'models/custom.pth',
            'use_ipex': True,
            'use_onnx': True,
            'onnx_path': 'models/custom.onnx',
            'deep_gaze_space': 'camera',
            'deep_pose_input': 'zero',
            'deep_ray_origin': 'zero_origin',
        },
        'geometry': {
            'screen_w_mm': 400.0,
            'screen_h_mm': 250.0,
        },
        'smoother': {
            'alpha': 0.5,
            'type': 'ema',
        },
        'tracker': {
            'target_fps': 60,
        },
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        # 加载配置
        config = SystemConfig.from_yaml(temp_path)
        
        # 验证摄像头配置
        assert config.camera_index == 1
        assert config.camera_width == 1280
        assert config.camera_height == 720
        assert config.camera_backend == 'auto'
        
        # 验证模型配置
        assert config.checkpoint_path == 'models/custom.pth'
        assert config.use_ipex is True
        assert config.use_onnx is True
        assert config.onnx_path == 'models/custom.onnx'
        assert config.deep_gaze_space == 'camera'
        assert config.deep_pose_input == 'zero'
        assert config.deep_ray_origin == 'zero_origin'
        
        # 验证几何配置
        assert config.screen_w_mm == 400.0
        assert config.screen_h_mm == 250.0
        
        # 验证平滑配置
        assert config.smoother_alpha == 0.5
        assert config.smoother_type == 'ema'
        
        # 验证追踪配置
        assert config.target_fps == 60
        
    finally:
        # 清理临时文件
        Path(temp_path).unlink()


def test_system_config_yaml_round_trip():
    """测试配置的保存和加载往返一致性。"""
    # 创建自定义配置
    original_config = SystemConfig(
        camera_index=2,
        camera_width=1920,
        camera_height=1080,
        camera_backend='auto',
        checkpoint_path='test_model.pth',
        use_ipex=True,
        use_onnx=False,
        deep_gaze_space='camera',
        deep_pose_input='zero',
        deep_ray_origin='zero_origin',
        screen_w_mm=500.0,
        screen_h_mm=300.0,
        smoother_alpha=0.7,
        smoother_type='none',
        target_fps=45,
    )
    
    # 构建 YAML 数据（模拟 SettingsPage._handle_save 的逻辑）
    config_data = {
        'camera': {
            'index': original_config.camera_index,
            'width': original_config.camera_width,
            'height': original_config.camera_height,
            'backend': original_config.camera_backend,
        },
        'face_detection': {
            'eye_crop_size': original_config.eye_crop_size,
            'min_detection_confidence': original_config.min_detection_confidence,
            'min_tracking_confidence': original_config.min_tracking_confidence,
        },
        'model': {
            'checkpoint_path': original_config.checkpoint_path,
            'use_ipex': original_config.use_ipex,
            'use_onnx': original_config.use_onnx,
            'onnx_path': original_config.onnx_path,
            'deep_gaze_space': original_config.deep_gaze_space,
            'deep_pose_input': original_config.deep_pose_input,
            'deep_ray_origin': original_config.deep_ray_origin,
        },
        'geometry': {
            'screen_w_mm': original_config.screen_w_mm,
            'screen_h_mm': original_config.screen_h_mm,
        },
        'smoother': {
            'alpha': original_config.smoother_alpha,
            'type': original_config.smoother_type,
        },
        'tracker': {
            'target_fps': original_config.target_fps,
            'timer_interval_ms': int(1000 / original_config.target_fps),
        },
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True)
        temp_path = f.name
    
    try:
        # 加载配置
        loaded_config = SystemConfig.from_yaml(temp_path)
        
        # 验证关键字段一致
        assert loaded_config.camera_index == original_config.camera_index
        assert loaded_config.camera_width == original_config.camera_width
        assert loaded_config.camera_height == original_config.camera_height
        assert loaded_config.camera_backend == original_config.camera_backend
        assert loaded_config.checkpoint_path == original_config.checkpoint_path
        assert loaded_config.use_ipex == original_config.use_ipex
        assert loaded_config.use_onnx == original_config.use_onnx
        assert loaded_config.deep_gaze_space == original_config.deep_gaze_space
        assert loaded_config.deep_pose_input == original_config.deep_pose_input
        assert loaded_config.deep_ray_origin == original_config.deep_ray_origin
        assert loaded_config.screen_w_mm == original_config.screen_w_mm
        assert loaded_config.screen_h_mm == original_config.screen_h_mm
        assert loaded_config.smoother_alpha == original_config.smoother_alpha
        assert loaded_config.smoother_type == original_config.smoother_type
        assert loaded_config.target_fps == original_config.target_fps
        
    finally:
        # 清理临时文件
        Path(temp_path).unlink()


def test_system_config_partial_yaml():
    """测试部分字段的 YAML 配置（其余使用默认值）。"""
    config_data = {
        'camera': {
            'index': 3,
        },
        'model': {
            'use_onnx': True,
        },
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = SystemConfig.from_yaml(temp_path)
        
        # 验证指定的字段
        assert config.camera_index == 3
        assert config.use_onnx is True
        
        # 验证未指定的字段使用默认值
        assert config.camera_width == 640  # 默认值
        assert config.camera_height == 480  # 默认值
        assert config.use_ipex is False  # 默认值
        
    finally:
        Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
