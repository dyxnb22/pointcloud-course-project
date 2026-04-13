#!/usr/bin/env bash
# train_augmented.sh - 使用数据增强训练 PointNet（随机旋转 + 抖动加噪）
# 运行前请先执行: bash colab_final/colab_setup.sh
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

if [ ! -d "pointnet.pytorch/data/modelnet40_ply_hdf5_2048" ]; then
  echo "Error: ModelNet40 dataset not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

echo "==> 启动数据增强训练（PointNet + random_rotate + jitter）..."
# 通过 PointNet 原生训练脚本传入增强标志（需 pointnet.pytorch 支持）
# 若上游脚本不支持 --augment，请直接用 experiments/attention/pointnet_attention.py
python pointnet.pytorch/utils/train_classification.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --feature_transform
