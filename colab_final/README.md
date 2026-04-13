# Colab 最终运行文件夹说明

English: This folder is the consolidated Colab-ready entry for final submission.

## 对应项目阶段

本目录对应项目的**第一阶段（Google Colab 阶段）**的最终整合提交入口。

根据 [`README.md`](../README.md) 中的两阶段说明：

- **第一阶段（本目录负责）**：在 Google Colab（T4 GPU）上完成环境搭建、PointNet Baseline 训练、进阶修改（数据增强 + SE-Attention）以及 SOTA 对比实验（DGCNN）。
- **第二阶段**：收集训练日志、整理实验结果对比表、MeshLab 可视化与误分类分析，最终形成报告与展示材料。

`colab_final/` 是第一阶段工作的**集中收尾版本**，将环境准备、Baseline 训练、DGCNN 对比训练三个核心步骤整合为可一键复现的脚本入口，便于课程最终提交时统一复现。

---

这个文件夹用于集中放置“最终在 Google Colab 上运行”的代码与脚本，便于直接提交与复现。

## 文件说明

- `colab_setup.sh`: 一键克隆依赖仓库并安装环境，同时下载 ModelNet40 数据集
- `train_baseline.sh`: 运行 PointNet Baseline 训练
- `train_dgcnn.sh`: 运行 DGCNN 对比实验训练

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

4. 执行 DGCNN 对比实验（可选）：

```bash
bash colab_final/train_dgcnn.sh
```

## 备注

- 该目录是 Colab 运行入口的集中版本，便于课程提交时统一整理。
- 训练输出（如 `loss.txt`、`accuracy.txt`、`best_model.pth`）请在 Colab 中及时下载保存。
