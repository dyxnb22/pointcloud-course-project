# 数据增强实验

在 Baseline 基础上，对数据加载器引入随机旋转（Random Rotation）和抖动加噪（Jitter），观察对分类精度的影响。

## 主要修改

- [`dataset_augmented.py`](dataset_augmented.py) 中实现 `random_rotate_point_cloud` 函数（随机绕 Z 轴旋转点云）
- [`dataset_augmented.py`](dataset_augmented.py) 中实现 `jitter_point_cloud` 函数（高斯抖动加噪）
- [`dataset_augmented.py`](dataset_augmented.py) 中实现 `AugmentedModelNetDataset`（训练阶段自动应用上述增强）

## 运行方式

```bash
bash colab_final/train_augmented.sh
```

或直接调用（需先完成环境准备），将 `AugmentedModelNetDataset` 替换原数据集：

```python
from experiments.augmentation.dataset_augmented import AugmentedModelNetDataset

train_set = AugmentedModelNetDataset(
    root='pointnet.pytorch/data/modelnet40_ply_hdf5_2048',
    split='train',
    npoints=2500,
    augment=True,
)
```

## 记录指标

- 最终 Accuracy：
- 相比 Baseline 变化：
