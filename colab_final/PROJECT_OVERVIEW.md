# 基于 `colab_final` 的项目说明与实现效果

## 1. 项目做了什么

本项目面向课程作业的**3D 点云分类**任务，目标是复现并对比 PointNet 与 DGCNN 在 ModelNet 数据集上的分类表现，并在此基础上完成跨数据集验证与高级扩展实验。`colab_final/` 目录是“最终可在 Google Colab 直接运行”的集中入口，包含环境准备、训练、对比与打包脚本，保证一键复现。

核心工作包括：

- **PointNet Baseline**：在 ModelNet40 上训练基线分类模型；
- **跨数据集验证**：将 ModelNet40 自动裁剪为 ModelNet10 子集，在子集上重新训练评估；
- **Advanced 2.2 扩展**：引入 label smoothing、尺度增强、feature transform、权重衰减、学习率调度，并记录 CSV 指标；
- **SOTA 对比**：使用 DGCNN 在 ModelNet40 上训练作为对照；
- **结果整理与可视化**：生成训练曲线对比图，并收集全部产物打包。

## 2. Colab 入口与数据准备

`colab_final/colab_setup.sh` 负责一键完成环境与数据：

- 克隆依赖仓库：`pointnet.pytorch` 与 `dgcnn`；
- 安装 PointNet 依赖（`pip install -e`）；
- 多镜像下载 ModelNet40 HDF5 数据并解压；
- 基于 ModelNet40 自动构建 **ModelNet10 子集**；
- 为 DGCNN 建立数据集软链接。

## 3. 训练流程与实验设计

### 3.1 PointNet Baseline（ModelNet40）

脚本：`colab_final/train_baseline.sh`

- 训练 20 epoch，输出 `cls/` 目录；
- 生成 `metrics.csv`、`loss.txt`、`accuracy.txt`；
- 自动更新 `best_model.pth`。

### 3.2 跨数据集验证（ModelNet10 子集）

脚本：`colab_final/train_cross_dataset.sh`

- 在 ModelNet10 子集上重新训练；
- 输出 `cls_cross/` 目录；
- 用于评估跨数据集泛化能力。

### 3.3 Advanced 2.2（ModelNet40）

脚本：`colab_final/train_advanced.sh`

实现的扩展包括：

- **label smoothing**：缓解过拟合；
- **scale augment**：随机缩放增强尺度鲁棒性；
- **feature transform**：启用 T-Net 对齐；
- **权重衰减 + cosine 调度**；
- **CSV/TXT 指标记录**；
- **MeshLab 点云导出**（带预测/真值注释）。

输出目录：`cls_advanced/`（含 `metrics.csv`、`best_model.pth`、`meshlab_ply/*.ply`）。

### 3.4 Advanced（ModelNet10 子集）

脚本：`colab_final/train_advanced_modelnet10.sh`

- 以相同高级配置在 ModelNet10 子集上训练；
- 输出目录：`cls_cross_advanced/`。

### 3.5 DGCNN 对比

脚本：`colab_final/train_dgcnn.sh`

- 在 ModelNet40 上训练 DGCNN（20 epoch）；
- 输出目录：`dgcnn/pytorch/checkpoints/dgcnn_test/`。

## 4. 结果与实现效果（来自 `colab_final/README.md`）

### 4.1 精度汇总

| 方法 | 数据集 | 最终测试精度 | 说明 |
| --- | --- | ---: | --- |
| PointNet Baseline | ModelNet40 | 76.9% | 20 epoch 下后期有回落（best=80.5%） |
| PointNet Baseline | ModelNet10 | 81.6% | 跨数据集有一定泛化能力（best=86.6%） |
| PointNet Advanced | ModelNet10 | 88.0% | 相比 Baseline final 提升 +6.4pp |
| DGCNN | ModelNet40 | 84.7% | 整体优于 PointNet，体现局部结构建模优势 |
| PointNet Advanced (2.2) | ModelNet40 | 80.8% | 相比 Baseline final 提升 +3.9pp |

### 4.2 训练曲线对比

`colab_final/plot_compare.py` 会读取 `metrics.csv` 并生成三张对比图：

- `curve_compare_loss.png`
- `curve_compare_accuracy.png`
- `curve_compare_lr.png`

ModelNet10 的对比图会生成：

- `curve_compare_modelnet10_loss.png`
- `curve_compare_modelnet10_accuracy.png`
- `curve_compare_modelnet10_lr.png`

### 4.3 失败案例与局限

根据 `meshlab_ply` 导出样本分析：

- PointNet 对家具/相似形状易混淆（如 `night_stand` ↔ `dresser` 等）；
- 全局汇聚对局部结构敏感度不足；
- 后期易出现过拟合（训练精度上升、测试精度回落）。

## 5. 打包与交付产物

### 5.1 ModelNet10 对比包

脚本：`colab_final/package_modelnet10_compare.sh`

- 收集 `cls_cross` 与 `cls_cross_advanced`；
- 自动生成对比曲线图；
- 输出 `modelnet10_compare/` 与 `modelnet10_compare.zip`。

### 5.2 最终提交包

脚本：`colab_final/package_final.sh`

- 汇总 PointNet 与 DGCNN 产物；
- 自动生成训练曲线图；
- 输出 `final/` 与 `final_submission.zip`；
- 包含 README、对比图、模型权重、日志等提交证据。

## 6. 总结

`colab_final` 将完整的**点云分类训练、跨数据集验证、Advanced 改进、SOTA 对比、可视化分析、结果打包**流程整合为可复现的 Colab 入口。最终效果表现为：

- Advanced 相比 Baseline 在 ModelNet40 与 ModelNet10 上都有稳定提升；
- DGCNN 整体精度高于 PointNet；
- 训练曲线与 MeshLab 样本提供了直观的模型表现与失败案例证据；
- 一键脚本确保了可重复运行与提交材料完整性。
