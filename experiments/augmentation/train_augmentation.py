"""
数据增强训练脚本。

使用随机旋转 + 抖动加噪训练 PointNet Baseline，
记录精度与 Loss 便于与 Baseline 结果对比。

运行方式（在项目根目录）
------------------------
    python experiments/augmentation/train_augmentation.py \
        --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
        --nepoch 20 \
        --batchsize 32
"""

import argparse
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# 确保 pointnet.pytorch 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pointnet.pytorch'))
sys.path.insert(0, os.path.dirname(__file__))

from pointnet.dataset import ModelNet40Dataset
from pointnet.model import PointNetCls, feature_transform_regularizer
from dataset_augment import ModelNet40AugDataset


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(args):
    train_dataset = ModelNet40AugDataset(
        root=args.dataset, split='train',
        npoints=args.num_points, data_augmentation=True,
    )
    test_dataset = ModelNet40Dataset(
        root=args.dataset, split='test',
        npoints=args.num_points, data_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize,
        shuffle=True, num_workers=4, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batchsize,
        shuffle=False, num_workers=4,
    )

    num_classes = len(train_dataset.classes)
    model = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_acc = 0.0
    for epoch in range(1, args.nepoch + 1):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for points, target in train_loader:
            points = points.transpose(2, 1).to(device)
            target = target[:, 0].to(device)

            optimizer.zero_grad()
            pred, trans, trans_feat = model(points)
            loss = nn.functional.nll_loss(pred, target)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_choice = pred.max(1)[1]
            correct += pred_choice.eq(target).sum().item()
            total += target.size(0)

        scheduler.step()
        train_acc = correct / total

        # 验证
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for points, target in test_loader:
                points = points.transpose(2, 1).to(device)
                target = target[:, 0].to(device)
                pred, _, _ = model(points)
                pred_choice = pred.max(1)[1]
                v_correct += pred_choice.eq(target).sum().item()
                v_total += target.size(0)

        val_acc = v_correct / v_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_augmentation.pth')

        print(f"[Epoch {epoch:03d}/{args.nepoch}] "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    print(f"\n==> Best val accuracy (augmentation): {best_acc:.4f}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to ModelNet40 HDF5 folder')
    parser.add_argument('--nepoch', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=2500)
    parser.add_argument('--feature_transform', action='store_true')
    args = parser.parse_args()
    train(args)
