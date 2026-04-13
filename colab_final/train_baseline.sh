#!/usr/bin/env bash
# train_baseline.sh - Train PointNet baseline on ModelNet40
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

if [ ! -d "pointnet.pytorch/data/modelnet40_ply_hdf5_2048" ]; then
  echo "Error: ModelNet40 dataset not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

python pointnet.pytorch/utils/train_classification.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --dataset_type modelnet40
