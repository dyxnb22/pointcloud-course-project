"""
SE-Block 通道注意力模块 — 嵌入 PointNet 特征提取层（代码增强）

在 Baseline PointNet 的全局特征提取后插入轻量级 SE-Block，
通过"通道重标定（channel recalibration）"强化重要特征通道，
从而在不显著增加参数量的前提下提升分类精度。

参考文献：
    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    Args:
        channels (int): 输入特征维度（通道数）。
        reduction (int): 压缩比，默认 16。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x (torch.Tensor): 形状 (B, C) 的全局特征向量。

        Returns:
            torch.Tensor: 通道注意力加权后的特征，形状 (B, C)。
        """
        scale = self.fc(x)
        return x * scale


class PointNetSE(nn.Module):
    """带 SE-Block 通道注意力的 PointNet 分类网络（简化版）。

    在原始 PointNet 全局最大池化后的 1024 维特征上插入 SEBlock，
    再经三层全连接完成 40 类分类。

    Args:
        num_classes (int): 分类数目，默认 40（ModelNet40）。
        se_reduction (int): SE-Block 压缩比，默认 16。
    """

    def __init__(self, num_classes: int = 40, se_reduction: int = 16):
        super().__init__()

        # --- 共享 MLP（逐点特征提取）---
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # --- SE-Block 通道注意力（代码增强核心）---
        self.se = SEBlock(channels=1024, reduction=se_reduction)

        # --- 分类头 ---
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x (torch.Tensor): 输入点云，形状 (B, 3, N)。

        Returns:
            torch.Tensor: 分类 log-softmax 输出，形状 (B, num_classes)。
        """
        # 逐点特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局最大池化 → (B, 1024)
        x = torch.max(x, dim=2)[0]

        # SE-Block 通道注意力加权
        x = self.se(x)

        # 分类全连接层
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
