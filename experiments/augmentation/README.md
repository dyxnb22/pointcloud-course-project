# 数据增强实验

在 Baseline 基础上，对数据加载器引入随机旋转（Random Rotation）和抖动加噪（Jitter），观察对分类精度的影响。

## 主要修改

- [`dataset.py`](dataset.py) 中实现 `random_rotate_point_cloud` 函数（沿 Z 轴随机旋转）
- [`dataset.py`](dataset.py) 中实现 `jitter_point_cloud` 函数（高斯噪声抖动加噪）

## 记录指标

- 最终 Accuracy：
- 相比 Baseline 变化：
