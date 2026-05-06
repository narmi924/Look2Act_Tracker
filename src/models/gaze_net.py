"""GazeNet 系列模型：视线方向/屏幕点回归 CNN。

V1 (GazeNet)：
    4 层 Conv + BN + ReLU + MaxPool → AdaptiveAvgPool → FC → L2 归一化
    输入：(B, 3, 128, 128) 单眼图像
    输出：(B, 3) 单位视线向量

V2 (GazeNetV2)：
    双眼共享 CNN 特征提取 + head pose 融合
    输入：左眼 (B, 3, 128, 128) + 右眼 (B, 3, 128, 128) + head_pose (B, 3)
    输出：(B, 3) 单位视线向量
    参数量 < 2.5M

PoG (GazeNetPoG)：
    双眼共享 CNN 特征提取 + head pose 融合
    输入同 V2
    输出：(B, 2) 归一化屏幕坐标 [x, y]，范围 [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeNet(nn.Module):
    """V1：轻量级单眼视线回归 CNN。

    参数:
        num_channels: 四层卷积的通道数列表，默认 [32, 64, 128, 256]
    """

    def __init__(self, num_channels: list[int] | None = None):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 256]

        assert len(num_channels) == 4, "需要恰好 4 层卷积通道配置"

        # 4 层 Conv + BN + ReLU + MaxPool
        self.features = nn.Sequential(
            # 第 1 层: 3 -> num_channels[0], 128x128 -> 64x64
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第 2 层: num_channels[0] -> num_channels[1], 64x64 -> 32x32
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第 3 层: num_channels[1] -> num_channels[2], 32x32 -> 16x16
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第 4 层: num_channels[2] -> num_channels[3], 16x16 -> 8x8
            nn.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 自适应平均池化 -> 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层: num_channels[3] -> 3
        self.fc = nn.Linear(num_channels[3], 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出 L2 归一化的 3D 视线向量。

        参数:
            x: 输入张量，形状 (B, 3, 128, 128)

        返回:
            形状 (B, 3) 的单位视线向量
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 归一化，确保输出为单位向量
        x = F.normalize(x, p=2, dim=1)
        return x


class GazeNetV2(nn.Module):
    """V2：双眼共享 CNN + head pose 融合的视线回归模型。

    架构：
    - 共享 CNN backbone 分别提取左右眼特征 (各 256 维)
    - 拼接双眼特征 + 3 维 head pose → 515 维
    - FC 融合层 → Dropout → FC 输出 → L2 归一化

    参数:
        num_channels: 四层卷积的通道数列表，默认 [32, 64, 128, 256]
        head_pose_dim: 头部姿态维度，默认 3 (yaw, pitch, roll)
        fusion_dim: 融合层隐藏维度，默认 128
        dropout: Dropout 概率，默认 0.3
    """

    def __init__(
        self,
        num_channels: list[int] | None = None,
        head_pose_dim: int = 3,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 256]

        assert len(num_channels) == 4, "需要恰好 4 层卷积通道配置"

        # 共享 CNN backbone（左右眼共用权重）
        self.eye_features = nn.Sequential(
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.eye_pool = nn.AdaptiveAvgPool2d(1)

        # 融合层：左眼(256) + 右眼(256) + head_pose(3) → fusion_dim → 3
        feat_dim = num_channels[3] * 2 + head_pose_dim  # 515
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3),
        )

    def _extract_eye_features(self, eye_img: torch.Tensor) -> torch.Tensor:
        """提取单眼 CNN 特征。

        参数:
            eye_img: (B, 3, 128, 128)

        返回:
            (B, 256) 特征向量
        """
        x = self.eye_features(eye_img)
        x = self.eye_pool(x)
        return x.view(x.size(0), -1)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        参数:
            left_eye: 左眼图像 (B, 3, 128, 128)
            right_eye: 右眼图像 (B, 3, 128, 128)
            head_pose: 头部姿态 (B, 3)，包含 yaw/pitch/roll（度）

        返回:
            (B, 3) 单位视线向量
        """
        # 共享 backbone 提取双眼特征
        left_feat = self._extract_eye_features(left_eye)   # (B, 256)
        right_feat = self._extract_eye_features(right_eye)  # (B, 256)

        # 拼接双眼特征 + head pose
        fused = torch.cat([left_feat, right_feat, head_pose], dim=1)  # (B, 515)

        # 融合回归
        out = self.fusion(fused)

        # L2 归一化
        out = F.normalize(out, p=2, dim=1)
        return out


class GazeNetPoG(nn.Module):
    """双眼 + head pose 的 Point-of-Gaze 回归模型。

    该模型不预测 3D gaze ray，而是直接预测屏幕归一化坐标。
    运行时可跳过 ray-plane intersection，用于建立可工作的 ML baseline。
    """

    def __init__(
        self,
        num_channels: list[int] | None = None,
        head_pose_dim: int = 3,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 256]

        assert len(num_channels) == 4, "需要恰好 4 层卷积通道配置"

        self.eye_features = nn.Sequential(
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.eye_pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = num_channels[3] * 2 + head_pose_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
            nn.Sigmoid(),
        )

    def _extract_eye_features(self, eye_img: torch.Tensor) -> torch.Tensor:
        x = self.eye_features(eye_img)
        x = self.eye_pool(x)
        return x.view(x.size(0), -1)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        left_feat = self._extract_eye_features(left_eye)
        right_feat = self._extract_eye_features(right_eye)
        fused = torch.cat([left_feat, right_feat, head_pose], dim=1)
        return self.fusion(fused)
