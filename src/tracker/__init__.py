"""实时追踪管道模块。"""

from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult
from src.tracker.smoother import GazeSmoother
from src.tracker.classic import (
    ClassicKalmanSmoother,
    ClassicScreenSmoother,
    detect_pupil_centroid,
    fuse_eye_features,
    normalize_iris_offset,
)

__all__ = [
    'TrackerPipeline',
    'TrackerResult',
    'SystemConfig',
    'GazeSmoother',
    'ClassicKalmanSmoother',
    'ClassicScreenSmoother',
    'detect_pupil_centroid',
    'fuse_eye_features',
    'normalize_iris_offset',
]
