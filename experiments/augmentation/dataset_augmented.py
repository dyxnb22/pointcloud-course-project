"""
数据增强模块 — PointNet 数据加载增强版本
=======================================
在原始 ModelNet40 数据集基础上加入：
  1. random_rotate_point_cloud：随机绕 Z 轴旋转点云
  2. jitter_point_cloud：对点坐标施加高斯抖动噪声

使用方式（在 pointnet.pytorch/utils/train_classification.py 中替换数据加载）：

    from experiments.augmentation.dataset_augmented import (
        random_rotate_point_cloud,
        jitter_point_cloud,
        AugmentedModelNetDataset,
    )

    train_dataset = AugmentedModelNetDataset(
        root='data/modelnet40_ply_hdf5_2048',
        split='train',
        npoints=2500,
    )
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 增强函数
# ---------------------------------------------------------------------------

def random_rotate_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """随机将点云绕 Z 轴（向上轴）旋转 [0, 2π)。

    Args:
        point_cloud: shape (N, 3) 的 float32 ndarray。

    Returns:
        旋转后的点云，shape (N, 3)。
    """
    theta = np.random.uniform(0, 2 * np.pi)
    cos_val = np.cos(theta)
    sin_val = np.sin(theta)
    rotation_matrix = np.array(
        [
            [cos_val, -sin_val, 0.0],
            [sin_val,  cos_val, 0.0],
            [0.0,      0.0,     1.0],
        ],
        dtype=np.float32,
    )
    return point_cloud @ rotation_matrix.T


def jitter_point_cloud(
    point_cloud: np.ndarray,
    sigma: float = 0.01,
    clip: float = 0.05,
) -> np.ndarray:
    """对点云每个坐标值添加截断高斯噪声。

    Args:
        point_cloud: shape (N, 3) 的 float32 ndarray。
        sigma: 高斯分布标准差（控制噪声强度）。
        clip: 噪声截断范围，超出 ±clip 的值被截断。

    Returns:
        添加抖动后的点云，shape (N, 3)。
    """
    assert clip > 0, "clip 必须大于 0"
    N, C = point_cloud.shape
    noise = np.clip(
        sigma * np.random.randn(N, C).astype(np.float32),
        -clip,
        clip,
    )
    return point_cloud + noise


# ---------------------------------------------------------------------------
# 增强数据集包装器
# ---------------------------------------------------------------------------

class AugmentedModelNetDataset(Dataset):
    """在 pointnet.pytorch 的 ModelNetDataset 基础上添加数据增强。

    该类从 HDF5 格式的 ModelNet40 文件中读取数据，并在训练阶段
    自动应用随机旋转与抖动增强。

    Args:
        root: modelnet40_ply_hdf5_2048 目录路径。
        split: 'train' 或 'test'。
        npoints: 每个样本采样的点数（默认 2500）。
        augment: 是否启用数据增强（仅在 split='train' 时建议开启）。
        jitter_sigma: jitter_point_cloud 的 sigma 参数（高斯噪声标准差）。
        jitter_clip: jitter_point_cloud 的 clip 参数（噪声截断范围）。
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        npoints: int = 2500,
        augment: bool = True,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
    ) -> None:
        import glob
        import h5py
        import os

        self.npoints = npoints
        self.augment = augment and (split == "train")
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

        # 读取所有 HDF5 分片
        pattern = os.path.join(root, f"ply_data_{split}*.h5")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"未找到 {split} 数据文件，请检查路径：{pattern}"
            )

        all_data = []
        all_label = []
        for fname in files:
            with h5py.File(fname, "r") as f:
                all_data.append(f["data"][:].astype(np.float32))
                all_label.append(f["label"][:].astype(np.int64))

        self.data = np.concatenate(all_data, axis=0)    # (N, 2048, 3)
        self.label = np.concatenate(all_label, axis=0)  # (N, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        point_set = self.data[idx].copy()
        label = self.label[idx]

        # 随机采样 npoints 个点
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # 归一化到单位球
        point_set -= point_set.mean(axis=0)
        dist = np.max(np.sqrt((point_set ** 2).sum(axis=1)))
        point_set /= dist

        # 数据增强（仅训练集）
        if self.augment:
            point_set = random_rotate_point_cloud(point_set)
            point_set = jitter_point_cloud(
                point_set,
                sigma=self.jitter_sigma,
                clip=self.jitter_clip,
            )

        point_set = torch.from_numpy(point_set.T)   # (3, N) — PointNet 输入格式
        label = torch.tensor(label.item(), dtype=torch.long)
        return point_set, label
