# 数据增强实验（Advanced Requirements 2.2 — 数据修改）

在 Baseline 基础上，引入**增强型数据增强策略**：全 SO(3) 随机旋转 + 带裁剪的高斯抖动，
观察对分类精度及鲁棒性的影响。

## 修改动机

Baseline 训练仅对点云施加绕 Y 轴随机旋转（1 自由度）和轻微高斯抖动（σ=0.02）。
这导致模型对非直立姿态和噪声过于敏感（局部泛化能力弱）。

增强型策略：
- **全 SO(3) 旋转**：对训练点云施加任意三维方向的随机旋转（通过高斯矩阵 QR 分解
  均匀采样 SO(3) 空间），让模型学习真正的旋转不变特征表示；
- **带裁剪的抖动**：噪声从 N(0, 0.04) 采样后裁剪至 [−0.05, 0.05]，比 Baseline
  σ=0.02 更激进，但通过裁剪避免极端畸变，模拟真实扫描噪声。

## 实现位置

`colab_final/train_advanced.py` 中的 `random_so3_rotation()` 和 `jitter_with_clip()`
函数，以及 `ModelNetH5Dataset.__getitem__()` 中的 `enhanced_aug` 分支。

## 关键代码

```python
def random_so3_rotation() -> np.ndarray:
    """Uniform SO(3) rotation via QR decomposition."""
    H = np.random.randn(3, 3).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def jitter_with_clip(points, sigma=0.04, clip=0.05):
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise.astype(np.float32)
```

## 运行命令

```bash
bash colab_final/train_advanced.sh
# 或使用镜像脚本
bash scripts/train_advanced.sh
```

脚本中 `--enhanced_aug` 标志同时启用 SO(3) 旋转和裁剪抖动。

## 记录指标

- 最终 Accuracy：（运行后填写 `cls_advanced/final_accuracy.txt`）
- 相比 Baseline 变化：（预期 +0.5~1.5 pp on ModelNet40）
