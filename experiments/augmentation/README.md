# 数据增强实验

在 Baseline 基础上，对数据加载器引入随机旋转（Random Rotation）和抖动加噪（Jitter），观察对分类精度的影响。

## 主要修改

- `dataset.py` 中增加 `random_rotate_point_cloud` 函数（实现见 [`dataset.py`](dataset.py)）
- `dataset.py` 中增加 `jitter_point_cloud` 函数（实现见 [`dataset.py`](dataset.py)）

## 记录指标

- 最终 Accuracy：
- 相比 Baseline 变化：
