"""
SE-Block 通道注意力 + 数据增强版 PointNet（分类）
=================================================
在原始 PointNet 特征提取层（全局特征 1024-d）之后插入轻量级
SE-Block（Squeeze-and-Excitation Block），用于通道维度的自注意力校正。

依赖：与 pointnet.pytorch 原始环境相同（torch + numpy）

使用方式（在 pointnet.pytorch 目录下执行）：

    python experiments/attention/pointnet_attention.py \
        --dataset data/modelnet40_ply_hdf5_2048 \
        --nepoch 20

详见 experiments/attention/README.md。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 将 experiments/augmentation 加入路径，复用增强数据集
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "augmentation"))

from dataset_augmented import AugmentedModelNetDataset  # noqa: E402


# ---------------------------------------------------------------------------
# SE-Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block（通道注意力）。

    对输入特征向量执行全局 squeeze → 两层 FC excitation → 逐元素缩放。

    Args:
        channels: 特征维度。
        reduction: 压缩比（默认 16，即瓶颈维度 = channels // reduction）。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, C) 的全局特征向量。

        Returns:
            经通道注意力校正后的特征，shape (B, C)。
        """
        scale = self.fc(x)
        return x * scale


# ---------------------------------------------------------------------------
# T-Net（空间变换网络）
# ---------------------------------------------------------------------------

class TNet(nn.Module):
    """PointNet 中的 T-Net（k×k 变换矩阵预测）。"""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        # 初始化为单位矩阵
        identity = (
            torch.eye(self.k, dtype=x.dtype, device=x.device)
            .view(1, self.k * self.k)
            .expand(B, -1)
        )
        x = x + identity
        return x.view(B, self.k, self.k)


# ---------------------------------------------------------------------------
# PointNet 特征提取（含 SE-Block）
# ---------------------------------------------------------------------------

class PointNetFeatSE(nn.Module):
    """PointNet 特征提取器，在全局特征后插入 SE-Block。

    Args:
        global_feat: 是否仅返回全局特征（分类用 True，分割用 False）。
        se_reduction: SE-Block 压缩比。
    """

    def __init__(self, global_feat: bool = True, se_reduction: int = 16) -> None:
        super().__init__()
        self.global_feat = global_feat
        self.tnet3 = TNet(k=3)
        self.tnet64 = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # SE-Block：对 1024-d 全局特征做通道注意力
        self.se = SEBlock(channels=1024, reduction=se_reduction)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (B, 3, N)

        Returns:
            全局特征 (B, 1024) 以及 T-Net 变换矩阵（用于正则化损失）。
        """
        B, D, N = x.size()

        # Input T-Net
        trans = self.tnet3(x)           # (B, 3, 3)
        x = torch.bmm(trans, x)         # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        local_feat = x                        # 保留局部特征（分割用）

        # Feature T-Net
        trans64 = self.tnet64(x)             # (B, 64, 64)
        x = torch.bmm(trans64, x)            # (B, 64, N)

        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))          # (B, 1024, N)
        x = torch.max(x, 2)[0]              # (B, 1024) 全局最大池化

        # SE-Block 通道注意力校正
        x = self.se(x)                       # (B, 1024)

        if self.global_feat:
            return x, trans64
        # 分割模式：将全局特征与局部特征拼接
        x = x.unsqueeze(2).expand(-1, -1, N)
        return torch.cat([local_feat, x], dim=1), trans64


# ---------------------------------------------------------------------------
# 分类网络
# ---------------------------------------------------------------------------

class PointNetClsSE(nn.Module):
    """带 SE-Block 的 PointNet 分类模型。

    Args:
        num_classes: 分类类别数（ModelNet40 = 40）。
        se_reduction: SE-Block 压缩比。
    """

    def __init__(self, num_classes: int = 40, se_reduction: int = 16) -> None:
        super().__init__()
        self.feat = PointNetFeatSE(global_feat=True, se_reduction=se_reduction)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        feat, trans64 = self.feat(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans64


# ---------------------------------------------------------------------------
# 正则化损失（Feature Transform Regularizer）
# ---------------------------------------------------------------------------

def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """||I - A·Aᵀ||_F² 正则化，确保特征变换矩阵接近正交。"""
    B, K, _ = trans.size()
    I = torch.eye(K, dtype=trans.dtype, device=trans.device).unsqueeze(0)
    loss = torch.mean(torch.norm(I - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2)))
    return loss


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 数据集
    train_set = AugmentedModelNetDataset(
        root=args.dataset,
        split="train",
        npoints=args.num_points,
        augment=True,
    )
    test_set = AugmentedModelNetDataset(
        root=args.dataset,
        split="test",
        npoints=args.num_points,
        augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"训练集大小：{len(train_set)}，测试集大小：{len(test_set)}")

    # 模型
    model = PointNetClsSE(num_classes=40, se_reduction=args.se_reduction).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.nepoch + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            pred, trans64 = model(points)
            loss = F.nll_loss(pred, labels)
            loss += feature_transform_regularizer(trans64) * 0.001
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_cls = pred.argmax(dim=1)
            correct += (pred_cls == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        # ---- 验证 ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                pred, _ = model(points)
                pred_cls = pred.argmax(dim=1)
                val_correct += (pred_cls == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch:3d}/{args.nepoch}] "
            f"Loss: {total_loss / len(train_loader):.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            if args.outdir:
                os.makedirs(args.outdir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pth"))
                print(f"  → 最佳模型已保存（Val Acc: {best_acc:.4f}）")

    print(f"\n训练完成，最佳验证精度：{best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet + SE-Block 分类训练（数据增强）")
    parser.add_argument("--dataset", type=str, required=True, help="modelnet40_ply_hdf5_2048 目录路径")
    parser.add_argument("--nepoch", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_points", type=int, default=2500, help="每样本采样点数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--se_reduction", type=int, default=16, help="SE-Block 压缩比")
    parser.add_argument("--outdir", type=str, default="results/attention", help="模型与日志输出目录")
    args = parser.parse_args()
    train(args)
