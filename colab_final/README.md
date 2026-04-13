# Colab 最终运行文件夹说明

English: This folder is the consolidated Colab-ready entry for final submission.

这个文件夹用于集中放置"最终在 Google Colab 上运行"的代码与脚本，便于直接提交与复现。

## 文件说明

| 脚本 | 用途 |
|------|------|
| `colab_setup.sh` | 一键克隆依赖仓库并安装环境，同时下载 ModelNet40 数据集 |
| `train_baseline.sh` | 运行 PointNet Baseline 训练 |
| `train_augmentation.sh` | **数据增强**训练（随机旋转 + 抖动加噪） |
| `train_attention.sh` | **代码增强**训练（SE-Block 通道注意力） |
| `train_dgcnn.sh` | 运行 DGCNN 对比实验训练 |

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

4. 执行数据增强训练（对应 `experiments/augmentation/`）：

```bash
bash colab_final/train_augmentation.sh
```

5. 执行代码增强（SE-Block 注意力）训练（对应 `experiments/attention/`）：

```bash
bash colab_final/train_attention.sh
```

6. 执行 DGCNN 对比实验（可选）：

```bash
bash colab_final/train_dgcnn.sh
```

## 增强模块说明

### 数据增强（train_augmentation.sh）

调用 `experiments/augmentation/train_augmentation.py`，使用以下两个增强函数对训练数据进行在线增强：

- `random_rotate_point_cloud`：绕 Y 轴随机旋转点云（±180°）
- `jitter_point_cloud`：向每个点叠加高斯随机噪声（σ=0.01，clip=0.05）

测试集**不**施加增强，保证与 Baseline 评估标准一致。

### 代码增强 — SE-Block 通道注意力（train_attention.sh）

调用 `experiments/attention/train_attention.py`，在 PointNet 提取的 1024-dim 全局特征之后插入轻量级 **SE-Block**：

```
全局特征 (1024) → SE-Block (squeeze→excite→scale) → 分类头
```

SE-Block 实现见 `experiments/attention/pointnet_se.py`。

## 备注

- 该目录是 Colab 运行入口的集中版本，便于课程提交时统一整理。
- 训练输出（如 `best_model_augmentation.pth`、`best_model_attention.pth`）请在 Colab 中及时下载保存。
