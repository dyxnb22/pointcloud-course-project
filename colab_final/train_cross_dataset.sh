#!/usr/bin/env bash
# train_cross_dataset.sh - 在 ShapeNet 上运行 PointNet 分类训练（跨数据集验证）
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

echo "==> 开始训练 PointNet 分类模型（跨数据集：ShapeNet）..."
python pointnet.pytorch/utils/train_classification.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type shapenet
