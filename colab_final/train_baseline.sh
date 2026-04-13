#!/usr/bin/env bash
# train_baseline.sh - Train PointNet baseline on ModelNet40
set -e

python pointnet.pytorch/utils/train_classification.py \
  --dataset pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0 \
  --nepoch=20 \
  --dataset_type modelnet40
