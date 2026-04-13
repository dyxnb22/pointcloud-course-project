"""
数据增强模块 — 在 PointNet Baseline 数据加载基础上添加随机旋转和抖动加噪。

使用方法
--------
在训练脚本中用 ``ModelNet40AugDataset`` 替换原始的 ModelNet40Dataset：

    from experiments.augmentation.dataset_augment import ModelNet40AugDataset
    train_dataset = ModelNet40AugDataset(root=DATA_DIR, split='train', npoints=2500)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pointnet.dataset import ModelNet40Dataset as _BaseDataset
except ImportError:
    _BaseDataset = None


# ---------------------------------------------------------------------------
# 增强函数
# ---------------------------------------------------------------------------

def random_rotate_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """绕 Y 轴随机旋转点云（±180°）。

    Parameters
    ----------
    point_cloud : np.ndarray, shape (N, 3)

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # 旋转矩阵（绕 Y 轴）
    R = np.array([
        [ cos_a, 0, sin_a],
        [     0, 1,     0],
        [-sin_a, 0, cos_a],
    ], dtype=np.float32)
    return point_cloud @ R.T


def jitter_point_cloud(point_cloud: np.ndarray,
                       sigma: float = 0.01,
                       clip: float = 0.05) -> np.ndarray:
    """向点云每个点加高斯噪声（抖动加噪）。

    Parameters
    ----------
    point_cloud : np.ndarray, shape (N, 3)
    sigma       : 噪声标准差，默认 0.01
    clip        : 噪声截断范围，默认 0.05

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    noise = np.clip(
        sigma * np.random.randn(*point_cloud.shape),
        -clip, clip
    ).astype(np.float32)
    return point_cloud + noise


# ---------------------------------------------------------------------------
# 增强版数据集
# ---------------------------------------------------------------------------

class ModelNet40AugDataset(Dataset):
    """对 ModelNet40Dataset 进行封装，在训练集上自动应用随机旋转 + 抖动加噪。

    参数与 pointnet.pytorch 原始 ModelNet40Dataset 完全兼容。
    当 split='test' 时不施加增强，与 Baseline 评估一致。
    """

    def __init__(self, root: str, split: str = 'train',
                 npoints: int = 2500, data_augmentation: bool = True):
        if _BaseDataset is None:
            raise ImportError(
                "无法导入 pointnet.dataset.ModelNet40Dataset，"
                "请先执行 bash colab_final/colab_setup.sh 安装依赖。"
            )
        self._base = _BaseDataset(
            root=root,
            split=split,
            npoints=npoints,
            data_augmentation=False,  # 由本类统一控制增强
        )
        self.split = split
        self.augment = data_augmentation and (split == 'train')

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        points, label = self._base[idx]

        if isinstance(points, torch.Tensor):
            points = points.numpy()

        if self.augment:
            points = random_rotate_point_cloud(points.astype(np.float32))
            points = jitter_point_cloud(points)

        return torch.from_numpy(points), label
