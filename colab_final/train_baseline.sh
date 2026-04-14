#!/usr/bin/env bash
# train_baseline.sh - 在 ModelNet40 上运行 PointNet 训练（主线）
# 可从仓库根目录运行：bash colab_final/train_baseline.sh
# 也可进入文件夹后运行：bash train_baseline.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet40_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

echo "==> 开始训练 PointNet 分类模型..."
python "${SCRIPT_DIR}/train_classification_h5.py" \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40
