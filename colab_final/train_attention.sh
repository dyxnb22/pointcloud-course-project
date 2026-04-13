#!/usr/bin/env bash
# train_attention.sh - Train PointNet + SE-Block (代码增强) on ModelNet40
# 代码增强训练：在数据增强的基础上，于全局特征后插入 SE-Block 通道注意力模块
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

if [ ! -d "pointnet.pytorch/data/modelnet40_ply_hdf5_2048" ]; then
  echo "Error: ModelNet40 dataset not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

python experiments/attention/train_attention.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --batchsize=32 \
  --se_reduction=16