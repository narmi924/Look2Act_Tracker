"""视线估计损失函数。"""

import torch


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """角度损失：计算预测向量与目标向量之间的角度（弧度）。

    loss = mean(arccos(clamp(dot(pred, target), -1+eps, 1-eps)))

    参数:
        pred: 预测视线向量，形状 (B, 3)，应为单位向量
        target: 目标视线向量，形状 (B, 3)，应为单位向量

    返回:
        标量张量，批次平均角度误差（弧度）
    """
    eps = 1e-7
    # 逐样本点积
    cos_sim = torch.sum(pred * target, dim=1)
    # clamp 防止 arccos 数值不稳定
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    # 角度误差（弧度）
    angle = torch.acos(cos_sim)
    return angle.mean()
