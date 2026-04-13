"""
SE-Block 通道注意力增强版 PointNet。

"代码增强"：在 PointNet 全局特征提取（1024-dim）之后插入轻量级
SE-Block，通过 Squeeze-and-Excitation 机制对通道进行加权，
提升分类判别力。

使用方法
--------
直接用 ``PointNetClsSE`` 替换原始 ``PointNetCls``::

    from experiments.attention.pointnet_se import PointNetClsSE
    model = PointNetClsSE(k=40)
"""

import torch
import torch.nn as nn

try:
    from pointnet.model import PointNetCls, feature_transform_regularizer
except ImportError:
    PointNetCls = None
    feature_transform_regularizer = None


# ---------------------------------------------------------------------------
# SE-Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """轻量级通道注意力模块（Squeeze-and-Excitation Block）。

    Parameters
    ----------
    channels  : 输入特征维度
    reduction : 压缩比，默认 16
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C)

        Returns
        -------
        Tensor, shape (B, C)  — 经通道注意力加权后的特征
        """
        scale = self.fc(x)
        return x * scale


# ---------------------------------------------------------------------------
# SE 增强版分类网络
# ---------------------------------------------------------------------------

class PointNetClsSE(nn.Module):
    """在 PointNet 全局特征后插入 SE-Block 的分类网络。

    当 ``pointnet`` 包不可用时会抛出 ``ImportError``；
    请先执行 ``bash colab_final/colab_setup.sh`` 安装依赖。

    Parameters
    ----------
    k                  : 分类数，ModelNet40 为 40
    feature_transform  : 是否使用特征变换矩阵正则化
    reduction          : SE-Block 压缩比
    """

    def __init__(self, k: int = 40, feature_transform: bool = False,
                 reduction: int = 16):
        super().__init__()
        if PointNetCls is None:
            raise ImportError(
                "无法导入 pointnet.model，"
                "请先执行 bash colab_final/colab_setup.sh 安装依赖。"
            )
        self._base = PointNetCls(k=k, feature_transform=feature_transform)

        # SE-Block 插入 1024-dim 全局特征之后；分类头复用 _base 中的层
        self.se = SEBlock(channels=1024, reduction=reduction)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor, shape (B, 3, N)

        Returns
        -------
        pred   : Tensor, shape (B, k)  — log-softmax 输出
        trans  : 输入变换矩阵（3×3）
        trans_feat : 特征变换矩阵（64×64），或 None
        """
        # 特征提取：调用 _base 的 PointNetfeat（1024-dim 全局特征）
        feat, trans, trans_feat = self._base.feat(x)

        # SE-Block 通道注意力
        feat = self.se(feat)

        # 分类头：复用 _base 中已有的层，避免重复定义
        x = nn.functional.relu(self._base.bn1(self._base.fc1(feat)))
        x = nn.functional.relu(self._base.bn2(self._base.dropout(self._base.fc2(x))))
        x = self._base.fc3(x)
        return nn.functional.log_softmax(x, dim=1), trans, trans_feat
