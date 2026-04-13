#!/usr/bin/env bash
# train_attention.sh - 使用 SE-Block 注意力机制 + 数据增强训练 PointNet
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

echo "==> 启动 SE-Attention 训练（PointNet + SE-Block + 数据增强）..."
python experiments/attention/pointnet_attention.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch 20 \
  --batch_size 32 \
  --num_points 2500 \
  --lr 1e-3 \
  --se_reduction 16 \
  --outdir results/attention
