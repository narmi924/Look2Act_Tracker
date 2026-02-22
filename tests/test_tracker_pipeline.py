"""TrackerPipeline 集成测试。

测试端到端推理管道的初始化、运行和错误处理。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker.pipeline import SystemConfig, TrackerPipeline


def test_system_config_from_yaml():
    """测试从 YAML 文件加载系统配置。"""
    config_path = Path(__file__).parent.parent / "configs" / "system_config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"配置文件不存在: {config_path}")
    
    config = SystemConfig.from_yaml(str(config_path))
    
    assert config.camera_index >= 0
    assert config.camera_width > 0
    assert config.camera_height > 0
    assert config.eye_crop_size == 128
    assert 0.0 < config.smoother_alpha <= 1.0


def test_tracker_pipeline_initialization():
    """测试 TrackerPipeline 初始化（不启动线程）。"""
    config = SystemConfig(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        checkpoint_path="checkpoints/best_model.pth",
    )
    
    pipeline = TrackerPipeline(
        model_path=config.checkpoint_path,
        config=config,
    )
    
    # 验证初始状态
    assert not pipeline.is_running()
    assert pipeline.get_latest_result() is None
    assert pipeline.get_latest_frame() is None


def test_tracker_result_dataclass():
    """测试 TrackerResult 数据类。"""
    from tracker.pipeline import TrackerResult
    
    # 有效结果
    result = TrackerResult(
        gaze_point=(100.0, 200.0),
        valid=True,
        fps=30.0,
        timings={'face_detection': 5.0, 'gaze_regression': 10.0},
        error_message=None,
        face_detected=True,
    )
    
    assert result.gaze_point == (100.0, 200.0)
    assert result.valid is True
    assert result.fps == 30.0
    assert result.face_detected is True
    assert result.error_message is None
    
    # 无效结果（未检测到人脸）
    result_no_face = TrackerResult(
        gaze_point=None,
        valid=False,
        fps=30.0,
        timings={},
        error_message="未检测到人脸",
        face_detected=False,
    )
    
    assert result_no_face.gaze_point is None
    assert result_no_face.valid is False
    assert result_no_face.face_detected is False
    assert "未检测到人脸" in result_no_face.error_message


def test_error_callback():
    """测试错误回调机制。"""
    config = SystemConfig(camera_index=999)  # 无效摄像头索引
    
    error_messages = []
    
    def error_callback(msg: str):
        error_messages.append(msg)
    
    pipeline = TrackerPipeline(
        model_path="checkpoints/best_model.pth",
        config=config,
        error_callback=error_callback,
    )
    
    # 尝试初始化（应该失败）
    success = pipeline.initialize()
    
    assert not success
    assert len(error_messages) > 0
    assert "摄像头" in error_messages[0]


@pytest.mark.skipif(
    not Path("checkpoints/best_model.pth").exists(),
    reason="需要训练好的模型权重"
)
def test_process_frame_with_mock_frame():
    """测试单帧处理（使用模拟图像）。"""
    config = SystemConfig(
        camera_index=0,
        checkpoint_path="checkpoints/best_model.pth",
    )
    
    pipeline = TrackerPipeline(
        model_path=config.checkpoint_path,
        config=config,
    )
    
    # 手动初始化子模块（不启动摄像头）
    from vision.face_detector import FaceDetector
    from vision.head_pose import HeadPoseEstimator
    from models.gaze_net import GazeNet
    from tracker.smoother import GazeSmoother
    
    pipeline.face_detector = FaceDetector()
    pipeline.head_pose_estimator = HeadPoseEstimator(frame_size=(640, 480))
    pipeline.gaze_model = GazeNet()
    pipeline.gaze_model.eval()
    pipeline.smoother = GazeSmoother(alpha=0.3)
    
    # 创建模拟图像（纯黑色，无人脸）
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = pipeline.process_frame(mock_frame)
    
    # 无人脸时应返回无效结果
    assert result.valid is False
    assert result.face_detected is False
    assert 'face_detection' in result.timings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
