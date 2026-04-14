#!/usr/bin/env bash
# train_cross_dataset.sh - 在 ModelNet10 子集上运行 PointNet 分类训练（跨数据集验证）
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

echo "==> 开始训练 PointNet 分类模型（跨数据集：ModelNet10 子集）..."
python pointnet.pytorch/utils/train_classification.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40
