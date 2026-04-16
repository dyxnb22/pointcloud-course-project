# Colab 最终运行文件夹说明

English: This folder is the consolidated Colab-ready entry for final submission.

这个文件夹用于集中放置"最终在 Google Colab 上运行"的代码与脚本，便于直接提交与复现。

## 文件说明

- `colab_setup.sh`: 一键克隆依赖仓库并安装环境，同时用多镜像回退准备 ModelNet40，并自动构建 ModelNet10 子集作为第二数据集
- `train_classification_h5.py`: PointNet 分类训练入口（兼容 `modelnet40_ply_hdf5_2048`）
- `train_baseline.sh`: 运行 PointNet Baseline 训练（ModelNet40 主线）
- `train_cross_dataset.sh`: 运行 PointNet 跨数据集训练（ModelNet10 子集）
- `train_dgcnn.sh`: 运行 DGCNN 对比实验训练
- `train_advanced.sh`: **[2.2]** 运行 PointNet 高级扩展实验（label smoothing + scale augment + feature transform + CSV 指标记录）
- `package_final.sh`: 一键收集产物到 `final/` 并打包 `final_submission.zip`

## 建议使用顺序（Colab）

> 两种调用方式均支持：
> - 从仓库根目录运行：`bash colab_final/<脚本名>.sh`
> - 将本文件夹上传到 Colab 后进入目录运行：`bash <脚本名>.sh`

1. 上传或克隆本仓库到 Colab 运行目录
2. 执行环境准备：

```bash
bash colab_final/colab_setup.sh
```

3. 执行 Baseline 训练（ModelNet40）：

```bash
bash colab_final/train_baseline.sh
```

4. 执行跨数据集训练（ModelNet10 子集）：

```bash
bash colab_final/train_cross_dataset.sh
```

5. 执行 DGCNN 对比实验：

```bash
bash colab_final/train_dgcnn.sh
```

6. **[2.2]** 执行高级扩展实验（Advanced Requirement 2.2）：

```bash
bash colab_final/train_advanced.sh
```

7. 一键打包提交文件（创建 `final/` 并生成压缩包）：

```bash
bash colab_final/package_final.sh
```

> 打包脚本会尝试自动运行 `plot_compare.py`（输入 `cls/metrics.csv` 与 `cls_advanced/metrics.csv`）生成 3 张独立图片：`curve_compare_loss.png`、`curve_compare_accuracy.png`、`curve_compare_lr.png`，并一起放入 `final_submission.zip`。

---

## 训练输出文件位置（Output File Locations）

运行各脚本后，产出文件均位于 **Colab 工作目录**（即运行 `bash` 命令所在目录）的以下路径：

| 脚本 | 产出文件路径 | 说明 |
|------|-------------|------|
| `train_baseline.sh` | `cls/cls_model_<epoch>.pth` | PointNet Baseline 每轮模型权重 |
| `train_cross_dataset.sh` | `cls_cross/cls_model_<epoch>.pth` | 跨数据集（ModelNet10）每轮模型权重（独立目录） |
| `train_cross_dataset.sh` | `cls_cross/best_model.pth` | 跨数据集最优模型（按每轮 test_acc 自动更新） |
| `train_cross_dataset.sh` | `cls_cross/metrics.csv` | 每轮 `epoch,train_loss,train_acc,test_loss,test_acc,lr` |
| `train_cross_dataset.sh` | `cls_cross/loss.txt` / `cls_cross/accuracy.txt` | 每轮 loss/accuracy 文本记录 |
| `train_dgcnn.sh` | `dgcnn/pytorch/checkpoints/dgcnn_test/models/model.t7` | DGCNN 最佳模型权重 |
| `train_dgcnn.sh` | `dgcnn/pytorch/checkpoints/dgcnn_test/run.log` | DGCNN 训练日志 |
| `train_advanced.sh` | `cls_advanced/cls_model_<epoch>.pth` | Advanced 每轮模型权重 |
| `train_advanced.sh` | `cls_advanced/best_model.pth` | Advanced 最优模型（按每轮 test_acc 自动更新） |
| `train_advanced.sh` | `cls_advanced/metrics.csv` | 每轮 `epoch,train_loss,train_acc,test_loss,test_acc,lr` 指标 |
| `train_advanced.sh` | `cls_advanced/loss.txt` / `cls_advanced/accuracy.txt` | 每轮 loss/accuracy 文本记录 |
| `train_advanced.sh` | `cls_advanced/meshlab_ply/*.ply` | 每轮导出的 MeshLab 点云样本（含预测/真值标签注释） |

> **提示**：Colab 重启后文件会丢失，请及时通过 `files.download()` 或 Google Drive 保存以上文件。

---

## Results（实验结果）

### 精度汇总

| 方法                    | 数据集                     | 最终测试精度 | 论文报告精度 | 差距分析                 |
| ----------------------- | -------------------------- | ------------ | ------------ | ------------------------ |
| PointNet Baseline       | ModelNet40                 | 74.9%        | 89.2%        | （填：偏高/偏低，原因）  |
| PointNet Baseline       | ModelNet10 (cross-dataset) | 82.7%        | —            | （填：泛化表现说明）     |
| DGCNN                   | ModelNet40                 | 84.7%        | 92.9%        | （填：偏高/偏低，原因）  |
| PointNet Advanced (2.2) | ModelNet40                 | 77.6%        | —            | （填：与 baseline 对比） |

### 训练曲线对比（Baseline vs Advanced）

![curve_compare_loss](curve_compare_loss.png)
![curve_compare_accuracy](curve_compare_accuracy.png)
![curve_compare_lr](curve_compare_lr.png)



### 论文结果对比分析

（填：你的结果与论文数值的差距，以及可能原因，如 epoch 数、数据增强、硬件差异等）

### 失败案例与方法局限性分析

（填：至少列出 2–3 类容易分类错误的点云类别，分析原因；
 说明 PointNet 的主要局限，如局部结构捕捉不足、尺度敏感等；
 提出可能的改进方向）

### Advanced 2.2 改进效果分析

（填：label smoothing + scale augment + feature transform 带来了多少精度提升？
 分析每项改动的贡献，是否符合预期，原因是什么）

## 完成项清单（Submission Checklist）

### 2.1 基础要求（Basic Requirements）

- [ ] **1. 项目介绍**：任务定义（3D点云分类）、输入输出（点云→类别）、数据集（ModelNet40/10）、参考论文（PointNet）、目标动机、技术挑战已在 README/报告中说明
- [ ] **2. 环境部署**：`colab_setup.sh` 可一键运行，README 中有逐步执行说明（pip install / conda 等）
- [ ] **3. Demo 可运行**：`train_baseline.sh` 能跑通并输出 loss/accuracy；README 中每步命令有说明
- [ ] **4. 模型训练**：已完成 baseline 训练（ModelNet40），`cls/cls_model_*.pth` 文件已产出并保存
- [ ] **5. 与论文对比**：已记录最终测试精度，与论文结果（89.2%）进行了数值对比，并分析误差原因；有训练曲线或日志截图
- [ ] **6. 其他数据集验证**：已运行 `train_cross_dataset.sh`（ModelNet10），记录了跨数据集精度，说明泛化结论
- [ ] **7. 缺点分析与改进**：已分析 PointNet 的局限性（如局部结构、尺度等），列出失败案例并给出改进思路
- [ ] **8. 实现并对比 SOTA 方法**：已运行 `train_dgcnn.sh`（DGCNN），并与 PointNet baseline 进行了精度对比，分析了方法差异

### 2.2 高级要求（Advanced Requirements）

- [ ] **方法扩展实现**：已运行 `train_advanced.sh`，启用了 label smoothing + scale augment + feature transform
- [ ] **CSV 指标记录**：`cls_advanced/metrics.csv` 已产出，包含完整的每轮 `epoch,train_loss,train_acc,test_acc`
- [ ] **改进动机说明**：已在报告/README 中说明每项改动的动机（为什么这样改、解决什么问题）
- [ ] **结果对比与分析**：已将 Advanced 精度与 Baseline 精度对比，分析改进效果；若精度未提升也要给出分析
- [ ] **证据文件齐全**：`cls_advanced/metrics.csv` 和最终模型权重已下载保存，作为提交证据

---

## Advanced Requirement 2.2 — 方法扩展说明

### 动机（Motivation）

PointNet 基线在 ModelNet40 上的分类精度受两方面限制：
1. **过拟合**：分类头使用标准 one-hot cross-entropy，模型容易对训练集标签过于自信，泛化能力有限。
2. **尺度不变性不足**：基线仅做了旋转和随机抖动增强，未考虑点云在真实场景中尺度的自然变化，导致对不同采集距离/分辨率的样本鲁棒性不足。

### 改动内容（What Changed）

在 `train_classification_h5.py` 中新增/扩展若干可选 CLI 标志，核心行为保持向后兼容：

| 标志 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--label_smoothing` | float | `0.0` | 标签平滑系数；推荐从 `0.05` 起调参，降低过拟合风险 |
| `--scale_augment` | 开关 | 关闭 | 训练时对每个点云随机缩放 ×[0.8, 1.25]，提升尺度鲁棒性 |
| `--log_csv` | str | `""` | 指定 CSV 文件路径；若设置则每轮写入 `epoch,train_loss,train_acc,test_loss,test_acc,lr`，并同步生成 `loss.txt/accuracy.txt` |
| `--meshlab_dir` | str | `""` | 指定目录后，每轮导出 MeshLab 可直接打开的 `.ply` 点云样本 |
| `--meshlab_samples_per_epoch` | int | `0` | 每轮导出的测试样本数（与 `--meshlab_dir` 配合） |
| `--weight_decay` | float | `0.0` | Adam 权重衰减，抑制过拟合 |
| `--scheduler` | str | `step` | 学习率调度器，可选 `step/cosine/none` |

具体实现要点：
- `label_smoothing_loss()` 函数：当 `smoothing=0` 时等价于 `F.nll_loss`（向后兼容）；否则用平滑分布替代 one-hot 目标。
- `ModelNetH5Dataset` 新增 `scale_augment` 参数，仅在 `data_augmentation=True` 时生效（测试集不受影响）。
- 每个 epoch 末尾执行完整测试评估：保存 train/test loss、train/test acc、当前学习率，并自动更新 `best_model.pth`。
- 开启 `--meshlab_dir` 后，每轮都会导出带预测/真值标签注释的 `.ply` 文件，便于直接在 MeshLab 可视化分析误分类。

### 如何运行（How to Run）

```bash
# 1. 确保环境已准备好（从仓库根目录运行）
bash colab_final/colab_setup.sh

# 2. 一键启动高级实验（从仓库根目录运行）
bash colab_final/train_advanced.sh

# 或：进入 colab_final 文件夹后直接运行
# bash train_advanced.sh
```

等价的完整命令（手动调参参考）：

```bash
python colab_final/train_classification_h5.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch 20 \
  --dataset_type modelnet40 \
  --feature_transform \
  --label_smoothing 0.05 \
  --scale_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --min_lr 0.00001 \
  --outf cls_advanced \
  --log_csv cls_advanced/metrics.csv \
  --meshlab_dir cls_advanced/meshlab_ply \
  --meshlab_samples_per_epoch 6
```

### 预期输出文件（Expected Output Files）

| 文件 | 说明 |
|------|------|
| `cls_advanced/cls_model_<epoch>.pth` | 每轮保存的模型权重 |
| `cls_advanced/best_model.pth` | 每轮评估后自动更新的最佳权重 |
| `cls_advanced/metrics.csv` | 每轮 `epoch,train_loss,train_acc,test_loss,test_acc,lr` |
| `cls_advanced/loss.txt` | 每轮 `epoch,train_loss,test_loss` |
| `cls_advanced/accuracy.txt` | 每轮 `epoch,train_acc,test_acc` |
| `cls_advanced/meshlab_ply/*.ply` | 每轮导出的 MeshLab 点云样本 |

`metrics.csv` 示例：

```
epoch,train_loss,train_acc,test_loss,test_acc,lr
0,2.1234,0.4512,1.9988,0.5031,0.00080000
1,1.8765,0.5234,1.7450,0.5612,0.00076121
...
```

---

## 备注

- 该目录是 Colab 运行入口的集中版本，便于课程提交时统一整理。
- `colab_setup.sh` 内置 ModelNet40 多镜像回退，并会基于 ModelNet40 自动构建 `modelnet10_ply_hdf5_2048`，无需额外下载第二数据集。
- 训练输出（如 `cls_advanced/metrics.csv`、`cls_model_*.pth`）请在 Colab 中及时下载保存。
