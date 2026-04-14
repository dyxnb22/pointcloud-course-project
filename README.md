# Point Cloud Course Project（PointNet vs DGCNN）

本仓库用于完成课程项目的**两阶段任务**：

- **第一阶段（Google Colab）**：环境搭建、Baseline 训练、进阶修改（数据增强 + SE-Attention）、SOTA 对比（DGCNN）
- **第二阶段（报告与展示）**：收集日志、整理结果表、MeshLab 可视化与误分类分析

---

## 仓库结构

```
pointcloud-course-project/
├── README.md
├── .gitignore
├── requirements.txt
├── colab_final/        # Colab 最终运行代码集中目录（提交入口）
├── notebooks/          # Google Colab 笔记本
├── scripts/            # 一键运行脚本
│   ├── colab_setup.sh
│   ├── train_baseline.sh
│   ├── train_cross_dataset.sh
│   └── train_dgcnn.sh
├── experiments/        # 各阶段实验记录
│   ├── baseline/
│   ├── augmentation/
│   ├── attention/
│   └── dgcnn_sota/
├── results/            # 训练日志与结果对比
├── assets/meshlab/     # MeshLab 可视化截图
└── report/             # 报告与 PPT 素材
```

---

## Colab 最终提交入口

如需将“最终在 Colab 运行的代码”统一放在一个目录，请直接使用 [`colab_final/`](colab_final/)。

该目录内已集中放置可直接执行的关键脚本，并附带使用说明文档：

- `colab_final/README.md`
- `colab_final/colab_setup.sh`
- `colab_final/train_baseline.sh`
- `colab_final/train_cross_dataset.sh`
- `colab_final/train_dgcnn.sh`

---

## 1. 环境准备（Google Colab T4 GPU）

1. 新建 Colab Notebook
2. 依次点击 **代码执行程序 → 更改运行时类型 → 硬件加速器：T4 GPU**
3. 在 Colab 单元格中执行：

```bash
# 克隆 PointNet（基线模型）
!git clone https://github.com/fxia22/pointnet.pytorch.git

# 克隆 DGCNN（SOTA 对比模型）
!git clone https://github.com/WangYueFt/dgcnn.git

# 安装 PointNet 依赖（Colab 已内置 PyTorch）
!pip install -e ./pointnet.pytorch
```

或直接运行一键脚本：

```bash
bash scripts/colab_setup.sh
```

---

## 2. Baseline 训练（PointNet，ModelNet40 主线）

### 2.1 下载数据集

```bash
bash scripts/colab_setup.sh
```

> `colab_setup.sh` 已内置 ModelNet40 多镜像下载与自动回退机制，并会基于 ModelNet40 自动构建第二数据集 `modelnet10_ply_hdf5_2048`（无需额外下载）。

### 2.2 训练命令

```bash
!python scripts/train_classification_h5.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --dataset_type modelnet40
```

或使用脚本：

```bash
bash scripts/train_baseline.sh
```

> **注意**：本仓库使用 `scripts/train_classification_h5.py` 作为 PointNet 的 HDF5 兼容训练入口。

---

## 3. 跨数据集与鲁棒性测试（基础要求 6 & 7）

PointNet 典型缺点：**局部特征提取能力弱、对噪声敏感**。

测试方法：
1. 在测试集数据加载函数中加入高斯抖动（Gaussian Jitter）
2. 使用第二分类数据集进行跨数据集测试（本仓库提供基于 ModelNet40 自动构建的 ModelNet10 子集），记录准确率变化作为“缺点分析”依据

跨数据集训练命令（ModelNet10 子集）：

```bash
bash scripts/train_cross_dataset.sh
```

---

## 4. 进阶修改（消融实验，占 20%）

### A. 数据增强

在 `dataset.py` 中加入：

- `random_rotate_point_cloud`：随机旋转
- `jitter_point_cloud`：抖动加噪

重新训练并记录精度，详见 [`experiments/augmentation/`](experiments/augmentation/)。

### B. SE-Block 通道注意力

在 `pointnet.py` 特征提取层后插入轻量级 SE-Block，再次训练并记录精度变化，详见 [`experiments/attention/`](experiments/attention/)。

---

## 5. SOTA 对比：DGCNN（基础要求 8）

```bash
!cd dgcnn/pytorch && python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40
```

或使用脚本：

```bash
bash scripts/train_dgcnn.sh
```

详见 [`experiments/dgcnn_sota/`](experiments/dgcnn_sota/)。

---

## 6. 必须保存的文件

| 文件 | 说明 |
|---|---|
| `loss.txt` | 每轮训练损失 |
| `accuracy.txt` | 每轮验证精度 |
| `best_model.pth` | 最佳权重文件 |

在 Colab 左侧文件栏找到后下载，课程提交压缩包中须包含这些文件。

---

## 7. 实验对比表

| 实验 | Accuracy | 备注 |
|---|---:|---|
| Baseline（原始 PointNet）| | |
| Baseline + 数据增强 | | |
| Baseline + 数据增强 + SE-Attention | | |
| SOTA（DGCNN） | | |

完整模板见 [`results/metrics_template.csv`](results/metrics_template.csv)。

---

## 8. MeshLab 可视化（基础要求 5）

1. 在本机安装 [MeshLab](https://www.meshlab.net/)
2. 从 ModelNet40 中挑选典型 `.off` 文件（如 `chair`、`airplane`）
3. 调整视角截图，保存至 [`assets/meshlab/`](assets/meshlab/)
4. 对比展示：
   - 原始 PointNet 误分类案例（局部细节不足）
   - 加入 Attention / 使用 DGCNN 后正确分类的改进原因

---

## 参考资料

- [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [ModelNet40 数据集](https://modelnet.cs.princeton.edu/)
- [MeshLab](https://www.meshlab.net/)
