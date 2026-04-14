#!/usr/bin/env bash
# train_advanced.sh - 进阶修改训练（PointNet + SE-Attention + 增强数据增强）
# Advanced Requirements (2.2)：架构修改（SE-Block）+ 数据修改（全 SO(3) 旋转 + 裁剪抖动）
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet40_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

echo "==> 开始训练 PointNet + SE-Attention（Advanced 2.2：架构 + 数据双重修改）..."
python colab_final/train_advanced.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --outf=cls_advanced \
  --enhanced_aug
