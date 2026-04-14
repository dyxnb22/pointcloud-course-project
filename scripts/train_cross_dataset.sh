#!/usr/bin/env bash
# train_cross_dataset.sh - 在 ModelNet10 子集上运行 PointNet 分类训练（跨数据集验证）
# 说明：使用本仓库的 HDF5 兼容训练入口，并沿用 dataset_type=modelnet40 分支。
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

echo "==> 开始训练 PointNet 分类模型（跨数据集：ModelNet10 子集，使用 modelnet40 加载分支）..."
python scripts/train_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40
