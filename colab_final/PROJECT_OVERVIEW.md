# 项目概述（基于 colab_final）

本项目面向 **3D 点云分类** 的课程实践，围绕 PointNet 基线模型进行复现与扩展，并通过 DGCNN 进行对比实验。`colab_final` 目录是“最终在 Google Colab 上一键运行”的整合入口，包含环境准备、训练脚本、结果对比与打包脚本，便于复现与提交。

---

## 1. 项目做了什么

### 1.1 任务目标
- **输入**：三维点云（HDF5 格式点集）
- **输出**：点云所属类别（分类任务）
- **数据集**：ModelNet40（主线）、ModelNet10 子集（跨数据集/鲁棒性验证）
- **方法路线**：
  - PointNet Baseline 复现
  - PointNet Advanced（2.2 高级要求）扩展
  - DGCNN 对比实验

### 1.2 Colab 一键复现流程
`colab_setup.sh` 完成以下关键步骤：
- 克隆依赖仓库：`pointnet.pytorch` 与 `dgcnn`
- 安装 PointNet 依赖
- 从多个镜像自动下载 **ModelNet40**
- **自动构建 ModelNet10 子集**（从 ModelNet40 过滤 10 类并重写 HDF5 列表）
- 为 DGCNN 创建数据软链接

---

## 2. 实验内容与实现效果

### 2.1 PointNet Baseline（ModelNet40）
- 脚本：`train_baseline.sh`
- 训练入口：`train_classification_h5.py`
- 输出：`cls/`（模型权重、最优权重、metrics.csv、loss.txt、accuracy.txt）

**效果**：
- ModelNet40 **最终测试精度 76.9%**，best=80.5%
- 相比论文 89.2%，仍存在约 12.3pp 差距

### 2.2 跨数据集验证（ModelNet10 子集）
- 脚本：`train_cross_dataset.sh`
- 数据：由 ModelNet40 自动裁剪得到 ModelNet10 子集
- 输出：`cls_cross/` 与完整训练日志

**效果**：
- ModelNet10 **最终测试精度 81.6%**，best=86.6%
- 体现一定泛化能力，但后期出现回落

### 2.3 Advanced Requirement 2.2（PointNet 改进）
- 脚本：`train_advanced.sh`
- 核心改动：
  - **label smoothing**（降低过度自信，缓解过拟合）
  - **scale augment**（提升尺度鲁棒性）
  - **feature transform**（启用 64D T-Net 对齐）
  - **CSV/TXT 指标记录** + **MeshLab 点云导出**

**效果**（ModelNet40）：
- **最终测试精度 80.8%**，相对 Baseline +3.9pp
- 训练更稳定，曲线波动更小

### 2.4 Advanced（ModelNet10 子集）
- 脚本：`train_advanced_modelnet10.sh`
- 输出：`cls_cross_advanced/` + metrics.csv

**效果**：
- ModelNet10 **最终测试精度 88.0%**，相对 Baseline +6.4pp

### 2.5 DGCNN 对比实验
- 脚本：`train_dgcnn.sh`
- 模型：DGCNN（局部图卷积建模）

**效果**：
- ModelNet40 **最终测试精度 84.7%**
- 相比论文 92.9%，仍有差距，但优于 PointNet 系列

---

## 3. 结果可视化与产物打包

### 3.1 训练曲线对比
- `plot_compare.py` 自动读取 `metrics.csv`，生成三张曲线图：
  - `*_loss.png`
  - `*_accuracy.png`
  - `*_lr.png`

### 3.2 打包输出
- `package_modelnet10_compare.sh`：
  - 收集 `cls_cross/` 与 `cls_cross_advanced/`
  - 生成 ModelNet10 Baseline vs Advanced 曲线图
  - 输出 `modelnet10_compare.zip`
- `package_final.sh`：
  - 汇总 PointNet / DGCNN 关键产物
  - 自动生成 Baseline vs Advanced 曲线图
  - 输出 `final_submission.zip`，并附带 `MANIFEST.txt`

---

## 4. 效果与分析要点（基于 colab_final 结果）

### 4.1 关键效果总结
| 方法 | 数据集 | 最终测试精度 | 备注 |
| --- | --- | --- | --- |
| PointNet Baseline | ModelNet40 | **76.9%** | 低于论文 89.2% |
| PointNet Baseline（跨数据集） | ModelNet10 | **81.6%** | best=86.6% |
| PointNet Advanced | ModelNet10 | **88.0%** | 相对 Baseline +6.4pp |
| DGCNN | ModelNet40 | **84.7%** | 低于论文 92.9% |
| PointNet Advanced (2.2) | ModelNet40 | **80.8%** | 相对 Baseline +3.9pp |

### 4.2 误差来源与局限性
项目总结指出：
- 训练轮数偏少（主要 20 epoch）
- 数据增强与优化配置与论文存在差异
- PointNet 局部结构建模较弱，易过拟合

### 4.3 改进方向
建议方向包括：
- 引入层次化局部建模（PointNet++）
  - 或基于图结构的局部关系建模（DGCNN）
- 加强数据增强、调参和训练策略（更长训练、cosine 调度、weight decay）

---

## 5. 推荐使用顺序（Colab）
1. `bash colab_final/colab_setup.sh`
2. `bash colab_final/train_baseline.sh`
3. `bash colab_final/train_cross_dataset.sh`
4. `bash colab_final/train_dgcnn.sh`
5. `bash colab_final/train_advanced.sh`
6. `bash colab_final/train_advanced_modelnet10.sh`
7. `bash colab_final/package_modelnet10_compare.sh`
8. `bash colab_final/package_final.sh`

---

## 6. 结论
该项目完整实现了 **PointNet 基线复现 + 高级改进 + DGCNN 对比** 的实验闭环，并提供了 **一键 Colab 复现** 与 **结果打包** 能力。Advanced 实验在 ModelNet40 和 ModelNet10 上均实现明显提升，验证了改动方向的有效性；同时也暴露了 PointNet 在局部结构建模与训练稳定性上的局限，说明进一步引入局部建模和更系统的训练策略仍有提升空间。
