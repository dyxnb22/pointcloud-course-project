# Colab 最终运行文件夹说明

English: This folder is the consolidated Colab-ready entry for final submission.

这个文件夹用于集中放置“最终在 Google Colab 上运行”的代码与脚本，便于直接提交与复现。

## 文件说明

- `colab_setup.sh`: 一键克隆依赖仓库并安装环境，同时用多镜像回退准备 ModelNet40 + ShapeNet 数据集（支持 `HF_TOKEN` / `HUGGINGFACE_TOKEN`）
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
- `colab_setup.sh` 内置多镜像回退，减少 ModelNet40/ShapeNet 下载卡死问题；若 ShapeNet 镜像全部失效，可手动上传 `shapenetcore_partanno_segmentation_benchmark_v0.zip` 到当前目录或 `/content`，或设置 `SHAPENET_ZIP_PATH` 指向该压缩包后重试（也可通过 `SHAPENET_URLS` 自定义镜像列表）。
- 训练输出（如 `loss.txt`、`accuracy.txt`、`best_model.pth`）请在 Colab 中及时下载保存。
