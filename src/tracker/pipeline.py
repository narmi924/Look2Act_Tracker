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
from src.models.gaze_net import GazeNet, GazeNetV2
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


@dataclass
class SystemConfig:
    """系统配置。"""
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
    
    # 几何配置
    screen_w_mm: float = 344.0
    screen_h_mm: float = 194.0
    
    # 界面配置
    main_window_fullscreen: bool = False
    
    # 平滑配置
    smoother_alpha: float = 0.3
    
    # 追踪配置
    target_fps: int = 30
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> SystemConfig:
        """从 YAML 文件加载配置。"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
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
            screen_w_mm=data.get('geometry', {}).get('screen_w_mm', 344.0),
            screen_h_mm=data.get('geometry', {}).get('screen_h_mm', 194.0),
            smoother_alpha=data.get('smoother', {}).get('alpha', 0.3),
            target_fps=data.get('tracker', {}).get('target_fps', 30),
            main_window_fullscreen=data.get('ui', {}).get('main_window_fullscreen', False),
        )


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
            _print(f"摄像头已打开：{actual_w}x{actual_h}")
            
            # 2. 初始化人脸检测器
            self.face_detector = FaceDetector(
                eye_crop_size=self.config.eye_crop_size,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            _print("人脸检测器已初始化")
            
            # 3. 初始化头部姿态估计器
            self.head_pose_estimator = HeadPoseEstimator(
                frame_size=(actual_w, actual_h)
            )
            _print("头部姿态估计器已初始化")
            
            # 4. 加载视线模型
            if self.config.use_onnx:
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
                
                if len(inputs) == 3 and 'left_eye' in self.onnx_input_names:
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
                        detected_version = "v1"
                    print(f"警告：模型文件不存在 {self.model_path}，使用随机初始化权重")
                
                self.model_version = detected_version
                model_cfg = config_from_ckpt.get("model", {})
                channels = model_cfg.get("channels", [32, 64, 128, 256])
                
                if self.model_version == "v2":
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
            # 获取屏幕分辨率
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
            
            # 设置默认屏幕平面参数（假设摄像头在屏幕上方中央）
            # 这些参数应该通过校准优化，这里使用简化的默认值
            camera_z_mm = 500.0  # 摄像头距离屏幕 50cm
            screen_origin = np.array([
                -self.config.screen_w_mm / 2.0,
                -self.config.screen_h_mm / 2.0,
                camera_z_mm
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
            print(f"屏幕几何模型已初始化：{screen_w_px}x{screen_h_px} px")
            
            # 6. 初始化平滑滤波器
            self.smoother = GazeSmoother(alpha=self.config.smoother_alpha)
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
                )
            
            # 无历史结果，返回无效
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="未检测到人脸",
                face_detected=False,
            )
        
        # 检测到人脸，重置计数器
        self._no_face_count = 0
        
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
                )
            
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="头部姿态估计失败",
                face_detected=True,
            )
        
        # 3. 视线回归（左右眼批处理）
        t0 = time.perf_counter()
        
        # 准备输入：左右眼合并为 batch=2
        left_eye = face_result.left_eye_crop
        right_eye = face_result.right_eye_crop
        
        if left_eye is None or right_eye is None:
            # 容错：返回上一帧有效结果
            if self._last_valid_result is not None:
                logger.debug("眼部裁剪失败，使用上一帧结果")
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="眼部裁剪失败",
                    face_detected=True,
                )
            
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="眼部裁剪失败",
                face_detected=True,
            )
        
        try:
            # 转换为张量并归一化
            left_tensor = torch.from_numpy(left_eye).permute(2, 0, 1).float() / 255.0
            right_tensor = torch.from_numpy(right_eye).permute(2, 0, 1).float() / 255.0
            
            # 构建 head pose 向量 (yaw, pitch, roll)，单位：度
            head_pose_vec = np.array([
                head_pose.yaw, head_pose.pitch, head_pose.roll
            ], dtype=np.float32)
            
            if self.model_version == "v2":
                # V2：双眼 + head pose 融合
                left_batch = left_tensor.unsqueeze(0)    # (1, 3, 128, 128)
                right_batch = right_tensor.unsqueeze(0)   # (1, 3, 128, 128)
                pose_batch = torch.from_numpy(head_pose_vec).unsqueeze(0)  # (1, 3)
                
                if self.config.use_onnx:
                    gaze_vector = self.onnx_session.run(
                        [self.onnx_output_name],
                        {
                            'left_eye': left_batch.numpy(),
                            'right_eye': right_batch.numpy(),
                            'head_pose': pose_batch.numpy(),
                        }
                    )[0][0]  # (3,)
                else:
                    with torch.no_grad():
                        gaze_out = self.gaze_model(left_batch, right_batch, pose_batch)
                    gaze_vector = gaze_out[0].cpu().numpy()  # (3,)
            else:
                # V1：左右眼批处理取均值
                batch = torch.stack([left_tensor, right_tensor], dim=0)  # (2, 3, 128, 128)
                
                if self.config.use_onnx:
                    batch_np = batch.numpy()
                    gaze_vectors = self.onnx_session.run(
                        [self.onnx_output_name],
                        {self.onnx_input_names[0]: batch_np}
                    )[0]
                    gaze_vector = gaze_vectors.mean(axis=0)  # (3,)
                else:
                    with torch.no_grad():
                        gaze_vectors = self.gaze_model(batch)  # (2, 3)
                    gaze_vector = gaze_vectors.mean(dim=0).cpu().numpy()  # (3,)
            
            timings['gaze_regression'] = (time.perf_counter() - t0) * 1000
            
        except Exception as e:
            logger.error(f"视线回归失败: {e}")
            # 容错：返回上一帧有效结果
            if self._last_valid_result is not None:
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message=f"视线回归失败: {e}",
                    face_detected=True,
                )
            
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message=f"视线回归失败: {e}",
                face_detected=True,
            )
        
        # 4. 坐标转换
        t0 = time.perf_counter()
        ray_origin, ray_direction = transform_gaze_to_camera(
            gaze_vector=gaze_vector,
            rotation_matrix=head_pose.rotation_matrix,
            translation_vec=head_pose.translation_vec,
        )
        timings['coordinate_transform'] = (time.perf_counter() - t0) * 1000
        
        # 5. 射线-平面求交（已启用 clamp_to_screen，自动处理边界情况）
        t0 = time.perf_counter()
        gaze_point = self.screen_geometry.get_gaze_point(
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            clamp_to_screen=True,  # 视线无效时 clamp 到屏幕边缘
        )
        timings['ray_plane_intersect'] = (time.perf_counter() - t0) * 1000
        
        if gaze_point is None:
            # 容错：返回上一帧有效结果
            if self._last_valid_result is not None:
                logger.debug("射线-平面求交失败，使用上一帧结果")
                return TrackerResult(
                    gaze_point=self._last_valid_result.gaze_point,
                    valid=True,
                    fps=self._calculate_fps(),
                    timings=timings,
                    error_message="视线与屏幕无交点",
                    face_detected=True,
                )
            
            return TrackerResult(
                gaze_point=None,
                valid=False,
                fps=self._calculate_fps(),
                timings=timings,
                error_message="视线与屏幕无交点",
                face_detected=True,
            )
        
        # 6. 时序平滑
        t0 = time.perf_counter()
        smoothed_point = self.smoother.update(gaze_point)
        timings['smoothing'] = (time.perf_counter() - t0) * 1000
        
        # 保存为有效结果（用于后续容错）
        result = TrackerResult(
            gaze_point=smoothed_point,
            valid=True,
            fps=self._calculate_fps(),
            timings=timings,
            error_message=None,
            face_detected=True,
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
                    )
                else:
                    result = TrackerResult(
                        gaze_point=None,
                        valid=False,
                        fps=self._calculate_fps(),
                        timings={},
                        error_message=f"帧处理异常: {e}",
                        face_detected=False,
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
        if not self._running:
            return
        
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
