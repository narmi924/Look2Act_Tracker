"""GazeNet：轻量级三维视线方向回归 CNN。

架构：4 层 Conv + BN + ReLU + MaxPool → AdaptiveAvgPool → FC → L2 归一化
输入：(B, 3, 128, 128) 眼部图像
输出：(B, 3) 单位视线向量 (gx, gy, gz)
参数量 < 2M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeNet(nn.Module):
    """轻量级视线回归 CNN。

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
