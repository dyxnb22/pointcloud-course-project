# Colab 最终运行文件夹说明

English: This folder is the consolidated Colab-ready entry for final submission.

这个文件夹用于集中放置“最终在 Google Colab 上运行”的代码与脚本，便于直接提交与复现。

## 文件说明

| 文件 | 说明 |
|---|---|
| `colab_setup.sh` | 一键克隆依赖仓库并安装环境，同时下载 ModelNet40 数据集 |
| `train_baseline.sh` | 运行 PointNet Baseline 训练（原始无增强） |
| `train_augmented.sh` | 运行数据增强训练（random_rotate + jitter，对应消融实验 A） |
| `train_attention.sh` | 运行 SE-Block 注意力训练（数据增强 + SE-Block，对应消融实验 B） |
| `train_dgcnn.sh` | 运行 DGCNN 对比实验训练（SOTA 对比） |

## 建议使用顺序（Colab）

1. 上传或克隆本仓库到 Colab 运行目录
2. 执行环境准备：

```bash
bash colab_final/colab_setup.sh
```

3. 执行 Baseline 训练：

```bash
bash colab_final/train_baseline.sh
```

4. 执行**数据增强**训练（消融实验 A）：

```bash
bash colab_final/train_augmented.sh
```

5. 执行 **SE-Block 注意力**训练（消融实验 B，含数据增强）：

```bash
bash colab_final/train_attention.sh
```

6. 执行 DGCNN 对比实验（可选）：

```bash
bash colab_final/train_dgcnn.sh
```

## 增强与注意力代码位置

| 功能 | 实现文件 |
|---|---|
| `random_rotate_point_cloud`（随机旋转） | `experiments/augmentation/dataset_augmented.py` |
| `jitter_point_cloud`（高斯抖动加噪） | `experiments/augmentation/dataset_augmented.py` |
| `AugmentedModelNetDataset`（增强数据集） | `experiments/augmentation/dataset_augmented.py` |
| `SEBlock`（通道注意力） | `experiments/attention/pointnet_attention.py` |
| `PointNetClsSE`（含 SE-Block 的分类模型） | `experiments/attention/pointnet_attention.py` |

## 备注

- 该目录是 Colab 运行入口的集中版本，便于课程提交时统一整理。
- 训练输出（如 `loss.txt`、`accuracy.txt`、`best_model.pth`）请在 Colab 中及时下载保存。
- SE-Block 训练结果默认保存至 `results/attention/best_model.pth`。
