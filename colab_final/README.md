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

## 建议使用顺序（Colab）

1. 上传或克隆本仓库到 Colab 运行目录
2. 执行环境准备：

```bash
bash colab_final/colab_setup.sh
```

3. 执行 Baseline 训练（ModelNet40）：

```bash
bash colab_final/train_baseline.sh
```

4. 执行跨数据集训练（ModelNet10 子集，可选）：

```bash
bash colab_final/train_cross_dataset.sh
```

5. 执行 DGCNN 对比实验（可选）：

```bash
bash colab_final/train_dgcnn.sh
```

6. **[2.2]** 执行高级扩展实验（Advanced Requirement 2.2）：

```bash
bash colab_final/train_advanced.sh
```

## 备注

- 该目录是 Colab 运行入口的集中版本，便于课程提交时统一整理。
- `colab_setup.sh` 内置 ModelNet40 多镜像回退，并会基于 ModelNet40 自动构建 `modelnet10_ply_hdf5_2048`，无需额外下载第二数据集。
- 训练输出（如 `loss.txt`、`accuracy.txt`、`best_model.pth`）请在 Colab 中及时下载保存。

---

## Advanced Requirement 2.2 — 方法扩展说明

### 动机（Motivation）

PointNet 基线在 ModelNet40 上的分类精度受两方面限制：
1. **过拟合**：分类头使用标准 one-hot cross-entropy，模型容易对训练集标签过于自信，泛化能力有限。
2. **尺度不变性不足**：基线仅做了旋转和随机抖动增强，未考虑点云在真实场景中尺度的自然变化，导致对不同采集距离/分辨率的样本鲁棒性不足。

### 改动内容（What Changed）

在 `train_classification_h5.py` 中新增三个可选 CLI 标志，**默认值均与基线完全一致**，不破坏已有行为：

| 标志 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `--label_smoothing` | float | `0.0` | 标签平滑系数；设为 `0.1` 时将正确类的目标概率从 1 降至 0.9，其余质量均匀分配到其他类，降低过拟合 |
| `--scale_augment` | 开关 | 关闭 | 训练时对每个点云随机缩放 ×[0.8, 1.25]，提升尺度鲁棒性 |
| `--log_csv` | str | `""` | 指定 CSV 文件路径；若设置则每轮结束后执行完整测试集评估并写入 `epoch,train_loss,train_acc,test_acc` |

具体实现要点：
- `label_smoothing_loss()` 函数：当 `smoothing=0` 时等价于 `F.nll_loss`（向后兼容）；否则用平滑分布替代 one-hot 目标。
- `ModelNetH5Dataset` 新增 `scale_augment` 参数，仅在 `data_augmentation=True` 时生效（测试集不受影响）。
- CSV 记录在每 epoch 末尾追加一行，支持实时查看训练曲线。

### 如何运行（How to Run）

```bash
# 1. 确保环境已准备好
bash colab_final/colab_setup.sh

# 2. 一键启动高级实验
bash colab_final/train_advanced.sh
```

等价的完整命令（手动调参参考）：

```bash
python colab_final/train_classification_h5.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch 20 \
  --dataset_type modelnet40 \
  --feature_transform \
  --label_smoothing 0.1 \
  --scale_augment \
  --outf cls_advanced \
  --log_csv cls_advanced/metrics.csv
```

### 预期输出文件（Expected Output Files）

| 文件 | 说明 |
|------|------|
| `cls_advanced/cls_model_<epoch>.pth` | 每轮保存的模型权重 |
| `cls_advanced/metrics.csv` | 每轮 `epoch,train_loss,train_acc,test_acc` |

`metrics.csv` 示例：

```
epoch,train_loss,train_acc,test_acc
0,2.1234,0.4512,0.5031
1,1.8765,0.5234,0.5612
...
```

### 结果汇报清单（Reporting Checklist）

- [ ] 记录 baseline（`train_baseline.sh`）最终测试精度（`final accuracy`）
- [ ] 记录高级实验（`train_advanced.sh`）最终测试精度
- [ ] 对比两组 `metrics.csv`（或训练日志），绘制或描述 loss/accuracy 曲线差异
- [ ] 说明 label smoothing + scale augment 是否带来精度提升，分析原因
- [ ] 下载并保存 `cls_advanced/metrics.csv` 和最终模型权重作为提交证据
