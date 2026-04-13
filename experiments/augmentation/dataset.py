"""
数据增强模块 — 供 PointNet ModelNet40 分类实验使用

在 Baseline 数据加载的基础上增加两种增强手段：
  1. random_rotate_point_cloud : 沿 Z 轴随机旋转（0 ~ 2π）
  2. jitter_point_cloud        : 向每个点坐标添加高斯噪声
"""

import numpy as np


def random_rotate_point_cloud(point_cloud):
    """沿 Z 轴随机旋转点云（数据增强）。

    Args:
        point_cloud (np.ndarray): 形状 (N, 3) 的点云数组。

    Returns:
        np.ndarray: 旋转后的点云，形状不变 (N, 3)。
    """
    angle = np.random.uniform(0, 2 * np.pi)  # 半开区间 [0, 2π)，均匀采样
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # 绕 Z 轴的旋转矩阵
    rotation_matrix = np.array(
        [
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ],
        dtype=np.float32,
    )
    return point_cloud @ rotation_matrix.T


def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    """向点云每个点坐标添加高斯噪声（数据增强）。

    Args:
        point_cloud (np.ndarray): 形状 (N, 3) 的点云数组。
        sigma (float): 高斯标准差，默认 0.01。
        clip (float): 噪声截断范围，默认 0.05。

    Returns:
        np.ndarray: 加噪后的点云，形状不变 (N, 3)。
    """
    noise = np.clip(
        sigma * np.random.randn(*point_cloud.shape).astype(np.float32),
        -clip,
        clip,
    )
    return point_cloud + noise


# ---------------------------------------------------------------------------
# 使用示例（Colab 笔记本 / 训练脚本中调用）
# ---------------------------------------------------------------------------
# from experiments.augmentation.dataset import random_rotate_point_cloud, jitter_point_cloud
#
# # 在 DataLoader 的 __getitem__ 中：
# if self.augment:
#     points = random_rotate_point_cloud(points)
#     points = jitter_point_cloud(points)
