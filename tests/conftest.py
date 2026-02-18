"""
Look2Act Tracker 测试配置
工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env pytest tests/
"""
import sys
from pathlib import Path

from hypothesis import settings, HealthCheck

# 将 src 加入 Python 路径，方便测试中直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# hypothesis 全局默认配置：每个 property test 至少运行 100 次
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")

# CI 环境可用更多迭代
settings.register_profile(
    "ci",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)
