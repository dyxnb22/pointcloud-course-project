"""
带 SE-Block 通道注意力的 PointNet（代码增强版）
在原始 PointNet 特征提取层后插入轻量级 SE-Block，
以增强对不同通道特征的自适应权重分配，提升分类精度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    通过全局平均池化 + 两层全连接网络计算各通道的重要性权重，
    再对输入特征进行逐通道缩放（channel-wise scaling）。

    Args:
        channels (int): 输入特征的通道数。
        reduction (int): 压缩比例，默认 16。
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        if channels < reduction:
            raise ValueError(
                f"channels ({channels}) must be >= reduction ({reduction})"
            )
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): shape (B, C) 的特征向量。

        Returns:
            torch.Tensor: 经通道注意力加权后的特征，shape 与输入相同。
        """
        scale = self.fc(x)
        return x * scale


class PointNetWithSE(nn.Module):
    """插入 SE-Block 的 PointNet 分类网络（代码增强版）。

    在 PointNet 全局特征提取（max-pool 后的 1024 维向量）之后
    加入 SE-Block，再经三层 MLP 输出分类 logits。

    Args:
        num_classes (int): 分类类别数，默认 40（ModelNet40）。
        se_reduction (int): SE-Block 压缩比，默认 16。
    """

    def __init__(self, num_classes=40, se_reduction=16):
        super(PointNetWithSE, self).__init__()
        # 逐点特征提取（shared MLP）
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # SE-Block 通道注意力（代码增强核心）
        self.se = SEBlock(1024, reduction=se_reduction)

        # 分类 MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): shape (B, 3, N) 的点云张量。

        Returns:
            torch.Tensor: shape (B, num_classes) 的分类 logits。
        """
        # 逐点特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局最大池化 → (B, 1024)
        x = torch.max(x, dim=2)[0]

        # SE-Block 通道注意力加权（代码增强）
        x = self.se(x)

        # 分类 MLP
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
