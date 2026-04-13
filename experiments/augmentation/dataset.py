"""
数据增强模块（Data Augmentation）
在 PointNet Baseline 基础上，为数据加载器引入随机旋转和抖动加噪，
以提升模型的泛化能力和对噪声的鲁棒性。
"""

import numpy as np


def random_rotate_point_cloud(point_cloud):
    """对点云绕 Y 轴进行随机旋转（Random Rotation around Y-axis）。

    Args:
        point_cloud (np.ndarray): shape (N, 3) 的点云数组。

    Returns:
        np.ndarray: 旋转后的点云，shape 与输入相同。
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_angle, 0,  sin_angle],
        [0,         1,  0        ],
        [-sin_angle, 0, cos_angle],
    ])
    rotated = point_cloud @ rotation_matrix.T
    return rotated


def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    """对点云每个点添加高斯随机抖动（Jitter with Gaussian noise）。

    Args:
        point_cloud (np.ndarray): shape (N, 3) 的点云数组。
        sigma (float): 高斯噪声标准差，默认 0.01。
        clip (float): 噪声截断范围，默认 0.05。

    Returns:
        np.ndarray: 加噪后的点云，shape 与输入相同。
    """
    assert clip > 0, "clip must be greater than 0"
    noise = np.clip(sigma * np.random.randn(*point_cloud.shape), -clip, clip)
    return point_cloud + noise
