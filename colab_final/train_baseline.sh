#!/usr/bin/env bash
# train_baseline.sh - Train PointNet baseline on ModelNet40
set -e

python pointnet.pytorch/utils/train_classification.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --dataset_type modelnet40
