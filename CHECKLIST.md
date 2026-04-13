# 项目实现清单（中文）

> 仓库：`dyxnb22/pointcloud-course-project`  
> 分支：`main`  
> 更新日期：2026-04-13

---

## 一、数据增强（Data Augmentation）

### 实现状态

| 序号 | 功能 | 实现文件 | 函数 / 类名 | 状态 |
|:---:|:---|:---|:---|:---:|
| 1 | 沿 Z 轴随机旋转（Random Rotation） | `experiments/augmentation/dataset.py` | `random_rotate_point_cloud` | ✅ 已实现 |
| 2 | 高斯抖动加噪（Jitter） | `experiments/augmentation/dataset.py` | `jitter_point_cloud` | ✅ 已实现 |

### 文件级证据

- **`experiments/augmentation/dataset.py`**
  - `random_rotate_point_cloud(point_cloud)`：生成随机旋转矩阵（0 ~ 2π），对点云做矩阵乘法，返回旋转后的 `(N, 3)` 数组。
  - `jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05)`：对每个点坐标叠加高斯噪声并做截断，增强对噪声的鲁棒性。

- **`experiments/augmentation/README.md`**（设计说明）
  - 说明了两个函数的接入位置（`dataset.py` 的 `__getitem__`），以及需要记录的对比指标。

### 在训练流程中如何使用

```python
# 在 DataLoader 的 __getitem__ 中启用：
from experiments.augmentation.dataset import (
    random_rotate_point_cloud,
    jitter_point_cloud,
)

if self.augment:
    points = random_rotate_point_cloud(points)
    points = jitter_point_cloud(points)
```

---

## 二、代码增强（Code Enhancement）

### 实现状态

| 序号 | 增强内容 | 实现文件 | 函数 / 类名 | 状态 |
|:---:|:---|:---|:---|:---:|
| 1 | SE-Block 通道注意力模块 | `experiments/attention/pointnet_se.py` | `SEBlock` | ✅ 已实现 |
| 2 | 嵌入 SE-Block 的 PointNet 分类网络 | `experiments/attention/pointnet_se.py` | `PointNetSE` | ✅ 已实现 |

### 文件级证据

- **`experiments/attention/pointnet_se.py`**
  - `SEBlock(channels, reduction=16)`：标准 Squeeze-and-Excitation 结构——两层全连接（含 ReLU + Sigmoid）对全局特征做通道重标定，以极少参数量提升特征判别能力。
  - `PointNetSE(num_classes=40, se_reduction=16)`：在原始 PointNet 全局最大池化（1024 维）之后紧接 `SEBlock`，再经三层全连接输出分类 log-softmax。相比 Baseline 的主要结构改动就是插入了这一 SE-Block。

- **`experiments/attention/README.md`**（设计说明）
  - 给出了 SE-Block 的参考结构及接入位置，以及需要记录的精度对比指标。

### SE-Block 核心代码摘要

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x)   # 学习每个通道的重要程度
        return x * scale     # 通道加权
```

---

## 三、运行脚本汇总

| 脚本文件 | 用途 |
|:---|:---|
| `colab_final/colab_setup.sh` | 环境准备（克隆依赖仓库 + 安装 + 下载 ModelNet40） |
| `colab_final/train_baseline.sh` | 训练原始 PointNet Baseline |
| `colab_final/train_augmentation.sh` | 训练带数据增强的 PointNet |
| `colab_final/train_dgcnn.sh` | 训练 DGCNN（SOTA 对比实验） |
| `scripts/train_baseline.sh` | 同上（scripts 镜像版） |
| `scripts/train_dgcnn.sh` | 同上（scripts 镜像版） |

---

## 四、完整实验对比清单

- [x] **Baseline**：原始 PointNet，在 ModelNet40 上训练 20 epoch
  - 代码入口：`colab_final/train_baseline.sh`
  - 指标记录：`results/baseline/`

- [x] **数据增强实验**：Baseline + Random Rotation + Jitter
  - 增强实现：`experiments/augmentation/dataset.py`
  - 运行入口：`colab_final/train_augmentation.sh`
  - 指标记录：`results/augmentation/`

- [x] **代码增强实验（SE-Attention）**：数据增强 + SE-Block 通道注意力
  - 模型实现：`experiments/attention/pointnet_se.py`
  - 指标记录：`results/attention/`

- [x] **SOTA 对比（DGCNN）**：Dynamic Graph CNN
  - 运行入口：`colab_final/train_dgcnn.sh`
  - 指标记录：`results/dgcnn_sota/`

---

## 五、结论

| 类别 | 是否已有实现文件 | 关键文件路径 |
|:---|:---:|:---|
| 数据增强 | ✅ 是 | `experiments/augmentation/dataset.py` |
| 代码增强（SE-Block 注意力） | ✅ 是 | `experiments/attention/pointnet_se.py` |
| Colab 运行入口（增强训练） | ✅ 是 | `colab_final/train_augmentation.sh` |
