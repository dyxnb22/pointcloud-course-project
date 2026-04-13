#!/usr/bin/env bash
# train_dgcnn.sh — Train DGCNN on ModelNet40 for SOTA comparison
set -e

cd dgcnn/pytorch
python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40
