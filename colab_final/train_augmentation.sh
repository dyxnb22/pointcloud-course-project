#!/usr/bin/env bash
# train_augmentation.sh - Train PointNet with data augmentation on ModelNet40
# 数据增强训练：在 Baseline 基础上启用随机旋转（Random Rotation）+ 抖动加噪（Jitter）
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

if [ ! -d "pointnet.pytorch/data/modelnet40_ply_hdf5_2048" ]; then
  echo "Error: ModelNet40 dataset not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

python experiments/augmentation/train_augmentation.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --batchsize=32
