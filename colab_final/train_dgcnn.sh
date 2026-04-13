#!/usr/bin/env bash
# train_dgcnn.sh - Train DGCNN on ModelNet40 for SOTA comparison
set -e

if [ ! -d "dgcnn/pytorch" ]; then
  echo "Error: dgcnn/pytorch not found. Please run: bash colab_final/colab_setup.sh"
  exit 1
fi

cd dgcnn/pytorch
python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40
