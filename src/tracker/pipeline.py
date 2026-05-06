"""端到端实时推理管道。

串联完整推理链路：
摄像头 → FaceDetector → 眼部裁剪 → GazeModel → HeadPoseEstimator 
→ 坐标转换 → ScreenGeometry → GazeSmoother → GazePoint

在独立线程中运行推理循环，监控各模块耗时，输出 FPS 和各阶段延迟。

错误处理与容错：
- 摄像头帧获取失败：重试 + 错误日志 + UI 通知
- 未检测到人脸：返回上一帧结果
- 视线无效：clamp 到屏幕边缘
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import yaml

from src.geometry.coordinate import transform_gaze_to_camera
from src.geometry.screen_geometry import ScreenGeometry
from src.models.gaze_net import GazeNet, GazeNetPoG, GazeNetV2
from src.tracker.classic import (
    ClassicKalmanSmoother,
    absolute_pupil_point,
    detect_pupil_centroid,
    fuse_eye_features,
)
from src.tracker.smoother import GazeSmoother
from src.vision.face_detector import FaceDetector
from src.vision.head_pose import HeadPoseEstimator


# 配置 logger
logger = logging.getLogger(__name__)


def _print(msg: str):
    """输出消息。"""
    print(msg)


@dataclass
class TrackerResult:
    """推理结果。"""
    gaze_point: Optional[tuple[float, float]]  # 屏幕像素坐标
    valid: bool
    fps: float
    timings: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）
    error_message: Optional[str] = None  # 错误信息（用于 UI 通知）
    face_detected: bool = True  # 是否检测到人脸
    raw_point: Optional[tuple[float, float]] = None
    calibrated_point: Optional[tuple[float, float]] = None
    backend: str = "deep"
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """系统配置。"""
    # 界面配置
    language: str = "bilingual"
    window_mode: str = "adaptive"  # "fullscreen" 或 "adaptive"

    # 摄像头配置
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_backend: str = "dshow"
    
    # 人脸检测配置
    eye_crop_size: int = 128
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # 模型配置
    checkpoint_path: str = "checkpoints/best_model.pth"
    use_ipex: bool = False
    use_onnx: bool = False
    onnx_path: str = "checkpoints/gaze_net.onnx"
    model_version: str = "auto"  # "auto" 从 checkpoint 自动检测, "v1", "v2"
    tracker_backend: str = "classic"  # "classic" 体验模式, "deep" 研究模型
    deep_gaze_space: str = "head"  # "head" 原链路, "camera" 跳过 PnP 旋转实验
    deep_pose_input: str = "live"  # "live" 使用 PnP 姿态, "zero" 用零向量做消融
    deep_ray_origin: str = "face_translation"  # "face_translation" 或 "zero_origin"
    
    # 几何配置
    screen_w_mm: float = 344.0
    screen_h_mm: float = 194.0
    screen_distance_mm: float = 500.0
    cam_above_screen_mm: float = 5.0
    
    # 平滑配置
    smoother_alpha: float = 0.3
    smoother_type: str = "kalman"  # classic 默认 Kalman，deep 默认使用 EMA
    
    # 追踪配置
    target_fps: int = 30
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> SystemConfig:
        """从 YAML 文件加载配置。"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            language=data.get('ui', {}).get('language', 'bilingual'),
            camera_index=data.get('camera', {}).get('index', 0),
            camera_width=data.get('camera', {}).get('width', 640),
            camera_height=data.get('camera', {}).get('height', 480),
            camera_backend=data.get('camera', {}).get('backend', 'dshow'),
            eye_crop_size=data.get('face_detection', {}).get('eye_crop_size', 128),
            min_detection_confidence=data.get('face_detection', {}).get('min_detection_confidence', 0.5),
            min_tracking_confidence=data.get('face_detection', {}).get('min_tracking_confidence', 0.5),
            checkpoint_path=data.get('model', {}).get('checkpoint_path', 'checkpoints/best_model.pth'),
            use_ipex=data.get('model', {}).get('use_ipex', False),
            use_onnx=data.get('model', {}).get('use_onnx', False),
            onnx_path=data.get('model', {}).get('onnx_path', 'checkpoints/gaze_net.onnx'),
            model_version=data.get('model', {}).get('model_version', 'auto'),
            tracker_backend=data.get('tracker', {}).get('backend', 'classic'),
            deep_gaze_space=data.get('model', {}).get('deep_gaze_space', 'head'),
            deep_pose_input=data.get('model', {}).get('deep_pose_input', 'live'),
            deep_ray_origin=data.get('model', {}).get('deep_ray_origin', 'face_translation'),
            screen_w_mm=data.get('geometry', {}).get('screen_w_mm', 344.0),
            screen_h_mm=data.get('geometry', {}).get('screen_h_mm', 194.0),
            screen_distance_mm=data.get('geometry', {}).get('screen_distance_mm', 500.0),
            cam_above_screen_mm=data.get('geometry', {}).get('cam_above_screen_mm', 5.0),
            smoother_alpha=data.get('smoother', {}).get('alpha', 0.3),
            smoother_type=data.get('smoother', {}).get('type', 'kalman'),
            target_fps=data.get('tracker', {}).get('target_fps', 30),
            window_mode=data.get('ui', {}).get('window_mode', 'adaptive'),
        )

    @property
    def normalized_backend(self) -> str:
        backend = (self.tracker_backend or "classic").lower()
        return backend if backend in {"classic", "deep", "deep_pog"} else "classic"

    @property
    def calibration_path(self) -> str:
        if self.normalized_backend == "classic":
            return "calibration_classic.json"
        if self.normalized_backend == "deep_pog":
            return "calibration_deep_pog.json"
        return "calibration_deep.json"

    @property
    def normalized_smoother_type(self) -> str:
        smoother_type = (self.smoother_type or "kalman").lower()
        return smoother_type if smoother_type in {"kalman", "ema", "none"} else "kalman"

    @property
    def normalized_deep_pose_input(self) -> str:
        pose_input = (self.deep_pose_input or "live").lower()
        return pose_input if pose_input in {"live", "zero"} else "live"

    @property
    def normalized_deep_ray_origin(self) -> str:
        origin = (self.deep_ray_origin or "face_translation").lower()
        return origin if origin in {"face_translation", "zero_origin"} else "face_translation"


class TrackerPipeline:
    """端到端实时推理管道。
    
    参数:
        model_path: 模型权重路径
        config: 系统配置对象
        error_callback: 错误回调函数（用于 UI 通知），接收错误消息字符串
    """
    
    def __init__(
        self, 
        model_path: str, 
        config: SystemConfig,
        error_callback: Optional[Callable[[str], None]] = None
    ):
        self.model_path = model_path
        self.config = config
        self.error_callback = error_callback
        
        # 子模块（在 initialize 中初始化）
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_detector: Optional[FaceDetector] = None
        self.head_pose_estimator: Optional[HeadPoseEstimator] = None
        self.gaze_model = None  # GazeNet 或 GazeNetV2
        self.model_version: str = "v1"  # 实际检测到的模型版本
        self.onnx_session = None  # ONNX Runtime session
        self.onnx_input_names: list[str] = []  # ONNX 输入名列表
        self.onnx_output_name: Optional[str] = None
        self.screen_geometry: Optional[ScreenGeometry] = None
        self.smoother: Optional[GazeSmoother] = None
        self.classic_smoother: Optional[ClassicKalmanSmoother] = None
        
        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # 最新结果（线程安全访问）
        self._latest_result: Optional[TrackerResult] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._last_valid_result: Optional[TrackerResult] = None  # 上一帧有效结果（用于容错）
        
        # FPS 计算
        self._frame_times: list[float] = []
        self._max_frame_times = 30
        
        # 校准模式标志：校准时禁用 clamp，保留原始坐标用于仿射拟合
        self._calibration_mode = False
        
        # 错误处理计数器
        self._camera_fail_count = 0
        self._max_camera_retries = 3
        self._no_face_count = 0
        self._camera_disconnected = False
        
    def _notify_error(self, message: str) -> None:
        """通知 UI 层错误信息。
        
        参数:
            message: 错误消息
        """
        _print(f"错误：{message}")
        if self.error_callback is not None:
            try:
                self.error_callback(message)
            except Exception as e:
                _print(f"错误回调执行失败: {e}")
    
    def initialize(self) -> bool:
        """初始化所有子模块：摄像头、检测器、模型、几何模型。
        
        返回:
            初始化是否成功
        """
        try:
            # 1. 打开摄像头
            if self.config.camera_backend == "dshow":
                self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.config.camera_index)
            
            if not self.cap.isOpened():
                error_msg = f"错误：无法打开摄像头 {self.config.camera_index}"
                self._notify_error(error_msg)
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _print(
                "摄像头已打开："
                f"请求 {self.config.camera_width}x{self.config.camera_height}，"
                f"实际 {actual_w}x{actual_h}"
            )
            
            # 2. 初始化人脸检测器
            self.face_detector = FaceDetector(
                eye_crop_size=self.config.eye_crop_size,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                refine_landmarks=self.config.normalized_backend == "classic",
            )
            _print("人脸检测器已初始化")
            
            # 3. 初始化头部姿态估计器
            self.head_pose_estimator = HeadPoseEstimator(
                frame_size=(actual_w, actual_h)
            )
            _print("头部姿态估计器已初始化")
            
            # 4. 加载视线模型（classic 后端不需要 CNN 权重）
            if self.config.normalized_backend == "classic":
                self.model_version = "classic"
                print("使用 Classic Tracker（pupil/iris feature + calibration）")
            elif self.config.use_onnx:
                # 使用 ONNX Runtime
                print("使用 ONNX Runtime 推理")
                import onnxruntime as ort
                
                onnx_path = self.config.onnx_path
                if not Path(onnx_path).exists():
                    print(f"错误：ONNX 模型文件不存在 {onnx_path}")
                    return False
                
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=['CPUExecutionProvider']
                )
                # 检测 ONNX 模型版本：V2 有 3 个输入，V1 有 1 个
                inputs = self.onnx_session.get_inputs()
                self.onnx_input_names = [inp.name for inp in inputs]
                self.onnx_output_name = self.onnx_session.get_outputs()[0].name
                
                output_shape = self.onnx_session.get_outputs()[0].shape
                output_dim = output_shape[-1] if output_shape else None
                if self.config.normalized_backend == "deep_pog" and output_dim != 2:
                    print(f"错误：deep_pog 需要 2D PoG ONNX 输出，但当前输出维度为 {output_dim}")
                    return False
                if self.config.normalized_backend == "deep_pog" or output_dim == 2:
                    self.model_version = "pog_v1"
                    print(f"ONNX Deep PoG 模型已加载：{onnx_path}（双眼 + head pose -> 2D PoG）")
                elif len(inputs) == 3 and 'left_eye' in self.onnx_input_names:
                    self.model_version = "v2"
                    print(f"ONNX V2 模型已加载：{onnx_path}（双眼 + head pose）")
                else:
                    self.model_version = "v1"
                    print(f"ONNX V1 模型已加载：{onnx_path}（单眼）")
                
            else:
                # 使用 PyTorch
                # 先加载 checkpoint 检测版本
                detected_version = self.config.model_version
                config_from_ckpt = {}
                state_dict = None
                
                if Path(self.model_path).exists():
                    checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                    if isinstance(checkpoint, dict):
                        if detected_version == "auto":
                            detected_version = checkpoint.get("model_version", "v1")
                        config_from_ckpt = checkpoint.get("config", {})
                        state_dict = checkpoint.get("model_state_dict", checkpoint)
                    else:
                        if detected_version == "auto":
                            detected_version = "v1"
                        state_dict = checkpoint
                else:
                    if detected_version == "auto":
                        detected_version = "pog_v1" if self.config.normalized_backend == "deep_pog" else "v1"
                    print(f"警告：模型文件不存在 {self.model_path}，使用随机初始化权重")

                if self.config.normalized_backend == "deep_pog" and detected_version != "pog_v1":
                    print(f"错误：deep_pog 需要 pog_v1 checkpoint，但当前模型版本为 {detected_version}")
                    return False
                
                self.model_version = detected_version
                model_cfg = config_from_ckpt.get("model", {})
                channels = model_cfg.get("channels", [32, 64, 128, 256])
                
                if self.model_version == "pog_v1":
                    self.gaze_model = GazeNetPoG(
                        num_channels=channels,
                        head_pose_dim=model_cfg.get("head_pose_dim", 3),
                        fusion_dim=model_cfg.get("fusion_dim", 128),
                        dropout=model_cfg.get("dropout", 0.3),
                    )
                    print("使用 GazeNetPoG（双眼 + head pose -> 2D PoG）")
                elif self.model_version == "v2":
                    self.gaze_model = GazeNetV2(
                        num_channels=channels,
                        head_pose_dim=model_cfg.get("head_pose_dim", 3),
                        fusion_dim=model_cfg.get("fusion_dim", 128),
                        dropout=model_cfg.get("dropout", 0.3),
                    )
                    print(f"使用 GazeNetV2（双眼 + head pose 融合）")
                else:
                    self.gaze_model = GazeNet(num_channels=channels)
                    print(f"使用 GazeNet V1（单眼）")
                
                if state_dict is not None and isinstance(state_dict, dict):
                    # 如果 state_dict 还包含其他 key（非纯 state_dict），尝试提取
                    if 'model_state_dict' in state_dict:
                        self.gaze_model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        self.gaze_model.load_state_dict(state_dict)
                    print(f"视线模型已加载：{self.model_path}")
                
                self.gaze_model.eval()
                
                # IPEX 优化（可选）
                if self.config.use_ipex:
                    try:
                        import intel_extension_for_pytorch as ipex
                        self.gaze_model = ipex.optimize(self.gaze_model)
                        print("IPEX 优化已启用")
                    except (ImportError, AttributeError, Exception) as e:
                        print(f"警告：IPEX 优化失败，跳过: {e}")
                        self.config.use_ipex = False
            
            # 5. 初始化屏幕几何模型
            # 获取屏幕逻辑分辨率（与 PyQt6 一致，避免高 DPI 下物理/逻辑分辨率不匹配）
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                screen = app.primaryScreen()
                if screen is not None:
                    geo = screen.geometry()
                    screen_w_px = geo.width()
                    screen_h_px = geo.height()
                else:
                    screen_w_px, screen_h_px = 1920, 1080
            else:
                # 回退：使用 tkinter（注意高 DPI 下可能返回物理分辨率）
                import tkinter as tk
                root = tk.Tk()
                screen_w_px = root.winfo_screenwidth()
                screen_h_px = root.winfo_screenheight()
                root.destroy()
            
            self.screen_geometry = ScreenGeometry(
                screen_w_px=screen_w_px,
                screen_h_px=screen_h_px,
                screen_w_mm=self.config.screen_w_mm,
                screen_h_mm=self.config.screen_h_mm,
                camera_matrix=self.head_pose_estimator.camera_matrix,
            )
            
            # Keep the online screen plane aligned with the training-time
            # camera/screen geometry used for label generation.
            screen_origin = np.array([
                -self.config.screen_w_mm / 2.0,
                self.config.cam_above_screen_mm,
                self.config.screen_distance_mm,
            ])
            screen_normal = np.array([0.0, 0.0, -1.0])  # 屏幕法向量指向摄像头
            screen_x_axis = np.array([1.0, 0.0, 0.0])
            screen_y_axis = np.array([0.0, 1.0, 0.0])
            
            self.screen_geometry.setup_plane(
                screen_origin_mm=screen_origin,
                screen_normal=screen_normal,
                screen_x_axis=screen_x_axis,
                screen_y_axis=screen_y_axis,
            )
            print(f"屏幕几何模型已初始化：屏幕 {screen_w_px}x{screen_h_px} px")
            
            # 6. 初始化平滑滤波器
            self.smoother = GazeSmoother(alpha=self.config.smoother_alpha)
            self.classic_smoother = ClassicKalmanSmoother()
            print("平滑滤波器已初始化")
            
            return True
            
        except Exception as e:
            error_msg = f"初始化失败：{e}"
            self._notify_error(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def process_frame(self, frame_bgr: np.ndarray) -> TrackerResult:
        """处理单帧，返回注视点和各阶段耗时。
        
        容错策略：
        - 未检测到人脸：返回上一帧有效结果
        - 视线无效（无交点）：已通过 clamp_to_screen=True 处理
        
        参数:
            frame_bgr: BGR 格式输入图像
            
        返回:
            TrackerResult 包含注视点和性能指标
        """
        timings = {}
        backend = self.config.normalized_backend
        
        # 1. 人脸检测
        t0 = time.perf_counter()
        face_result = self.face_detector.detect(frame_bgr)
        timings['face_detection'] = (time.perf_counter() - t0) * 1000
        
        if not face_result.detected:
            self._no_face_count += 1
            
            # 容错：返回上一帧有效结果
            if self._last_valid_result is not None:
                logger.debug(f"未检测到人脸（连续 {self._no_face_count} 帧），使用上一帧结果")
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="未检测到人脸，使用上一帧结果",
                    face_detected=False,
                    backend=backend,
                )
            
            # 无历史结果，返回无效
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="未检测到人脸",
                face_detected=False,
                backend=backend,
            )
        
        # 检测到人脸，重置计数器
        self._no_face_count = 0

        if self.config.normalized_backend == "classic":
            return self._process_classic_result(face_result, timings)
        
        # 2. 头部姿态估计
        t0 = time.perf_counter()
        head_pose = self.head_pose_estimator.estimate(face_result.pnp_points_2d)
        timings['head_pose'] = (time.perf_counter() - t0) * 1000
        
        if not head_pose.valid:
            # 容错：返回上一帧有效结果
            if self._last_valid_result is not None:
                logger.debug("头部姿态估计失败，使用上一帧结果")
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="头部姿态估计失败",
                    face_detected=True,
                    backend=backend,
                )
            
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="头部姿态估计失败",
                face_detected=True,
                backend=backend,
            )
        
        # 3. 视线回归（CNN 模型推理）
        t0 = time.perf_counter()
        
        left_eye = face_result.left_eye_crop
        right_eye = face_result.right_eye_crop
        
        if left_eye is None or right_eye is None:
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="眼部裁剪失败",
                    face_detected=True,
                    backend=backend,
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="眼部裁剪失败",
                face_detected=True,
                backend=backend,
            )
        
        try:
            # BGR → RGB（与训练时 _img_to_tensor 一致）
            left_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
            
            # 转换为张量并归一化
            left_tensor = torch.from_numpy(left_rgb).permute(2, 0, 1).float() / 255.0
            right_tensor = torch.from_numpy(right_rgb).permute(2, 0, 1).float() / 255.0
            
            # 构建 head pose 向量 (yaw, pitch, roll)，单位：度
            if self.config.normalized_deep_pose_input == "zero":
                head_pose_vec = np.zeros(3, dtype=np.float32)
            else:
                head_pose_vec = np.array([
                    head_pose.yaw, head_pose.pitch, head_pose.roll
                ], dtype=np.float32)
            
            if self.model_version in {"v2", "pog_v1"}:
                left_batch = left_tensor.unsqueeze(0)
                right_batch = right_tensor.unsqueeze(0)
                pose_batch = torch.from_numpy(head_pose_vec).unsqueeze(0)
                
                if self.config.use_onnx:
                    gaze_vector = self.onnx_session.run(
                        [self.onnx_output_name],
                        {
                            'left_eye': left_batch.numpy(),
                            'right_eye': right_batch.numpy(),
                            'head_pose': pose_batch.numpy(),
                        }
                    )[0][0]
                else:
                    with torch.no_grad():
                        gaze_out = self.gaze_model(left_batch, right_batch, pose_batch)
                    gaze_vector = gaze_out[0].cpu().numpy()
            else:
                batch = torch.stack([left_tensor, right_tensor], dim=0)
                if self.config.use_onnx:
                    gaze_vectors = self.onnx_session.run(
                        [self.onnx_output_name],
                        {self.onnx_input_names[0]: batch.numpy()}
                    )[0]
                    gaze_vector = gaze_vectors.mean(axis=0)
                else:
                    with torch.no_grad():
                        gaze_vectors = self.gaze_model(batch)
                    gaze_vector = gaze_vectors.mean(dim=0).cpu().numpy()
            
            timings['gaze_regression'] = (time.perf_counter() - t0) * 1000
            
        except Exception as e:
            logger.error(f"视线回归失败: {e}")
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message=f"视线回归失败: {e}",
                    face_detected=True,
                    backend=backend,
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message=f"视线回归失败: {e}",
                face_detected=True,
                backend=backend,
            )
        
        if backend == "deep_pog":
            return self._process_deep_pog_output(
                output=gaze_vector,
                head_pose=head_pose,
                timings=timings,
            )

        # 4. Convert the predicted 3D gaze vector into a raw screen-space point.
        t0 = time.perf_counter()
        d = gaze_vector.astype(np.float64)
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="视线方向为零向量",
                    face_detected=True,
                    backend=backend,
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="视线方向为零向量",
                face_detected=True,
                backend=backend,
            )
        
        d = d / norm_d
        ray_origin, ray_direction = self._compute_deep_ray(d, head_pose)
        timings['coordinate_transform'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        clamp_to_screen = not self._calibration_mode
        intersection = self.screen_geometry.ray_plane_intersect(ray_origin, ray_direction)
        timings['ray_plane_intersect'] = (time.perf_counter() - t0) * 1000

        if intersection is None:
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="视线射线未与屏幕平面相交",
                    face_detected=True,
                    backend=backend,
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="视线射线未与屏幕平面相交",
                face_detected=True,
                backend=backend,
            )

        raw_point = self.screen_geometry.world_to_screen_px(intersection, clamp=clamp_to_screen)
        
        # 5. 时序平滑
        t0 = time.perf_counter()
        if self.config.normalized_smoother_type == "none" or self.smoother is None:
            smoothed_point = raw_point
        else:
            smoothed_point = self.smoother.update(raw_point)
        timings['smoothing'] = (time.perf_counter() - t0) * 1000
        
        # 保存为有效结果（用于后续容错）
        result = TrackerResult(
            gaze_point=smoothed_point,
            valid=True,
            fps=self._calculate_fps(),
            timings=timings,
            error_message=None,
            face_detected=True,
            raw_point=raw_point,
            backend="deep",
            debug={
                "model_version": self.model_version,
                "onnx_inputs": list(self.onnx_input_names),
                "deep_gaze_space": self.config.deep_gaze_space,
                "deep_pose_input": self.config.normalized_deep_pose_input,
                "deep_ray_origin": self.config.normalized_deep_ray_origin,
                "ray_origin": ray_origin.tolist(),
                "ray_direction": ray_direction.tolist(),
                "gaze_vector": d.tolist(),
                "head_pose": {
                    "yaw": float(head_pose.yaw),
                    "pitch": float(head_pose.pitch),
                    "roll": float(head_pose.roll),
                },
                "screen_geometry": self.get_diagnostics().get("screen_geometry", {}),
            },
        )
        self._last_valid_result = result
        
        return result

    def _select_deep_ray_origin(self, face_translation: np.ndarray) -> np.ndarray:
        """Select the 3D deep ray origin for geometry-contract experiments."""
        if self.config.normalized_deep_ray_origin == "zero_origin":
            return np.zeros(3, dtype=np.float64)
        return np.asarray(face_translation, dtype=np.float64).flatten()

    def _compute_deep_ray(self, gaze_direction: np.ndarray, head_pose) -> tuple[np.ndarray, np.ndarray]:
        """Compute a 3D gaze ray according to the configured gaze-space contract."""
        d = np.asarray(gaze_direction, dtype=np.float64).flatten()
        norm = np.linalg.norm(d)
        if norm > 1e-12:
            d = d / norm
        if self.config.deep_gaze_space == "camera":
            return self._select_deep_ray_origin(head_pose.translation_vec), d
        ray_origin, ray_direction = transform_gaze_to_camera(
            d,
            head_pose.rotation_matrix,
            head_pose.translation_vec,
        )
        return self._select_deep_ray_origin(ray_origin), ray_direction

    def _process_deep_pog_output(
        self,
        output: np.ndarray,
        head_pose,
        timings: dict[str, float],
    ) -> TrackerResult:
        """Map a direct PoG model output to screen pixels without 3D ray geometry."""
        t0 = time.perf_counter()
        pog = np.asarray(output, dtype=np.float64).flatten()
        if pog.shape[0] < 2 or not np.all(np.isfinite(pog[:2])):
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="Deep PoG 输出无效",
                    face_detected=True,
                    backend="deep_pog",
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="Deep PoG 输出无效",
                face_detected=True,
                backend="deep_pog",
            )

        raw_norm = (float(pog[0]), float(pog[1]))
        screen_w = self.screen_geometry.screen_w_px if self.screen_geometry is not None else 1920
        screen_h = self.screen_geometry.screen_h_px if self.screen_geometry is not None else 1080
        raw_x = raw_norm[0] * screen_w
        raw_y = raw_norm[1] * screen_h
        if not self._calibration_mode:
            raw_x = float(np.clip(raw_x, 0.0, screen_w - 1.0))
            raw_y = float(np.clip(raw_y, 0.0, screen_h - 1.0))
        raw_point = (float(raw_x), float(raw_y))
        timings["pog_to_screen"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.config.normalized_smoother_type == "none" or self.smoother is None:
            smoothed_point = raw_point
        else:
            smoothed_point = self.smoother.update(raw_point)
        timings["smoothing"] = (time.perf_counter() - t0) * 1000

        result = TrackerResult(
            gaze_point=smoothed_point,
            valid=True,
            fps=self._calculate_fps(),
            timings=timings,
            error_message=None,
            face_detected=True,
            raw_point=raw_point,
            backend="deep_pog",
            debug={
                "model_version": self.model_version,
                "output_mode": "normalized_point_of_gaze",
                "raw_norm_point": raw_norm,
                "onnx_inputs": list(self.onnx_input_names),
                "deep_pose_input": self.config.normalized_deep_pose_input,
                "head_pose": {
                    "yaw": float(head_pose.yaw),
                    "pitch": float(head_pose.pitch),
                    "roll": float(head_pose.roll),
                },
                "screen_geometry": self.get_diagnostics().get("screen_geometry", {}),
            },
        )
        self._last_valid_result = result
        return result

    def _process_classic_result(
        self,
        face_result,
        timings: dict[str, float],
    ) -> TrackerResult:
        """Process a detected face with the Eye_Touch classic backend."""
        t0 = time.perf_counter()
        left_pupil = detect_pupil_centroid(getattr(face_result, "left_eye_roi", None))
        right_pupil = detect_pupil_centroid(getattr(face_result, "right_eye_roi", None))
        left_abs = absolute_pupil_point(left_pupil, getattr(face_result, "left_eye_origin", None))
        right_abs = absolute_pupil_point(right_pupil, getattr(face_result, "right_eye_origin", None))
        frame_size = getattr(face_result, "frame_size", None) or (
            self.config.camera_width,
            self.config.camera_height,
        )
        feature = fuse_eye_features(
            left_abs,
            right_abs,
            camera_width=int(frame_size[0]),
            camera_height=int(frame_size[1]),
            method="eyetouch_pupil",
        )
        timings["classic_feature"] = (time.perf_counter() - t0) * 1000

        if feature is None:
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="classic 眼部特征提取失败",
                    face_detected=True,
                    backend="classic",
                    debug={"left_pupil": left_pupil, "right_pupil": right_pupil, "left_abs": left_abs, "right_abs": right_abs},
                )
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="classic 眼部特征提取失败",
                face_detected=True,
                backend="classic",
                debug={"left_pupil": left_pupil, "right_pupil": right_pupil, "left_abs": left_abs, "right_abs": right_abs},
            )

        raw_point = feature.point

        result = TrackerResult(
            gaze_point=raw_point,
            valid=True,
            fps=self._calculate_fps(),
            timings=timings,
            error_message=None,
            face_detected=True,
            raw_point=raw_point,
            backend="classic",
            debug={
                "feature_method": feature.method,
                "feature_confidence": feature.confidence,
                "smoother_type": self.config.normalized_smoother_type,
                "left_pupil": left_pupil,
                "right_pupil": right_pupil,
                "left_abs": left_abs,
                "right_abs": right_abs,
                "frame_size": frame_size,
                "calibration_mode": self._calibration_mode,
            },
        )
        self._last_valid_result = result
        return result
    
    def _calculate_fps(self) -> float:
        """计算实时 FPS（基于最近 N 帧的时间间隔）。"""
        if len(self._frame_times) < 2:
            return 0.0
        
        time_diffs = [
            self._frame_times[i] - self._frame_times[i - 1]
            for i in range(1, len(self._frame_times))
        ]
        avg_time = sum(time_diffs) / len(time_diffs)
        
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0
    
    def run(self) -> None:
        """主循环：持续从摄像头读取帧并处理。在独立线程中运行。
        
        错误处理：
        - 摄像头帧获取失败：重试最多 3 次，失败后标记摄像头断开并通知 UI
        """
        print("推理管道已启动")
        
        while self._running:
            frame_start = time.perf_counter()
            
            # 读取帧（带重试机制）
            ret = False
            frame = None
            
            for retry in range(self._max_camera_retries):
                ret, frame = self.cap.read()
                if ret:
                    self._camera_fail_count = 0  # 重置失败计数
                    break
                
                logger.warning(f"摄像头帧获取失败，重试 {retry + 1}/{self._max_camera_retries}")
                time.sleep(0.01)
            
            if not ret or frame is None:
                self._camera_fail_count += 1
                logger.error(f"摄像头帧获取失败（连续 {self._camera_fail_count} 次）")
                
                # 标记摄像头断开并通知 UI
                if not self._camera_disconnected:
                    self._camera_disconnected = True
                    self._notify_error("摄像头断开连接，请检查设备")
                
                time.sleep(0.1)
                continue
            
            # 摄像头恢复正常
            if self._camera_disconnected:
                self._camera_disconnected = False
                print("摄像头已恢复连接")
            
            # 处理帧
            try:
                result = self.process_frame(frame)
            except Exception as e:
                logger.error(f"帧处理异常: {e}")
                import traceback
                traceback.print_exc()
                
                # 容错：使用上一帧结果
                if self._last_valid_result is not None:
                    result = TrackerResult(
                        gaze_point=self._last_valid_result.gaze_point,
                        valid=True,
                        fps=self._calculate_fps(),
                        timings={},
                        error_message=f"帧处理异常: {e}",
                        face_detected=False,
                        backend=self.config.normalized_backend,
                    )
                else:
                    result = TrackerResult(
                        gaze_point=None,
                        valid=False,
                        fps=self._calculate_fps(),
                        timings={},
                        error_message=f"帧处理异常: {e}",
                        face_detected=False,
                        backend=self.config.normalized_backend,
                    )
            
            # 更新 FPS 计算
            self._frame_times.append(frame_start)
            if len(self._frame_times) > self._max_frame_times:
                self._frame_times.pop(0)
            
            # 线程安全地更新最新结果
            with self._lock:
                self._latest_result = result
                self._latest_frame = frame.copy()
        
        print("推理管道已停止")
    
    def start(self) -> bool:
        """启动推理线程。
        
        返回:
            启动是否成功
        """
        if self._running:
            print("警告：推理管道已在运行")
            return False
        
        if not self.initialize():
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return True
    
    def stop(self) -> None:
        """停止推理线程并释放资源。"""
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.face_detector is not None:
            self.face_detector.close()
            self.face_detector = None

        if self.smoother is not None:
            self.smoother.reset()
        if self.classic_smoother is not None:
            self.classic_smoother.reset()

        self._last_valid_result = None
        self._latest_result = None
        self._latest_frame = None
        self._frame_times.clear()
        
        print("推理管道资源已释放")
    
    def get_latest_result(self) -> Optional[TrackerResult]:
        """获取最新的推理结果（线程安全）。"""
        with self._lock:
            return self._latest_result
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的摄像头帧（线程安全）。"""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None
    
    def is_running(self) -> bool:
        """检查推理管道是否正在运行。"""
        return self._running

    def set_calibration_mode(self, enabled: bool) -> None:
        """设置校准模式。校准模式下禁用坐标 clamp，保留原始值。"""
        self._calibration_mode = enabled
        if self.smoother is not None:
            self.smoother.reset()
        if self.classic_smoother is not None:
            self.classic_smoother.reset()

    def get_diagnostics(self) -> dict[str, object]:
        """Return runtime diagnostics for UI/debug displays."""
        diag: dict[str, object] = {
            "backend": self.config.normalized_backend,
            "model_version": self.model_version,
            "onnx_inputs": list(self.onnx_input_names),
            "calibration_path": self.config.calibration_path,
            "camera": {
                "index": self.config.camera_index,
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "backend": self.config.camera_backend,
            },
            "deep_gaze_space": self.config.deep_gaze_space,
            "deep_pose_input": self.config.normalized_deep_pose_input,
            "deep_ray_origin": self.config.normalized_deep_ray_origin,
        }
        if self.screen_geometry is not None:
            diag["screen_geometry"] = {
                "screen_w_px": self.screen_geometry.screen_w_px,
                "screen_h_px": self.screen_geometry.screen_h_px,
                "screen_w_mm": self.screen_geometry.screen_w_mm,
                "screen_h_mm": self.screen_geometry.screen_h_mm,
                "screen_distance_mm": self.config.screen_distance_mm,
                "cam_above_screen_mm": self.config.cam_above_screen_mm,
            }
        return diag
