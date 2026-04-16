#!/usr/bin/env bash
# train_advanced_modelnet10.sh - 在 ModelNet10 子集上运行 Advanced PointNet 训练（鲁棒性验证）
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

OUTPUT_DIR="cls_cross_advanced"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> 开始训练 PointNet Advanced（ModelNet10 子集，label smoothing + scale augment + feature transform）..."
python scripts/train_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --feature_transform \
  --label_smoothing 0.05 \
  --scale_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --min_lr 0.00001 \
  --outf "${OUTPUT_DIR}" \
  --log_csv "${CSV_PATH}" \
  --meshlab_dir "${OUTPUT_DIR}/meshlab_ply" \
  --meshlab_samples_per_epoch 6

echo ""
echo "==> ModelNet10 Advanced 训练完成。输出文件："
echo "    模型权重: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    最优模型: ${OUTPUT_DIR}/best_model.pth"
echo "    每轮指标: ${CSV_PATH}"
echo "    每轮loss: ${OUTPUT_DIR}/loss.txt"
echo "    每轮acc : ${OUTPUT_DIR}/accuracy.txt"
echo "    MeshLab  : ${OUTPUT_DIR}/meshlab_ply/*.ply"
