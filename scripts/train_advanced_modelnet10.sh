#!/usr/bin/env bash
# train_advanced_modelnet10.sh - Run Advanced PointNet on ModelNet10 subset (robustness validation)
set -e

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Run: bash scripts/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "Error: dataset folder ${DATASET_PATH} not found. Run: bash scripts/colab_setup.sh"
  exit 1
fi

TRAIN_SCRIPT="scripts/train_classification_h5.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "Error: training script ${TRAIN_SCRIPT} not found. Please verify repository integrity"
  exit 1
fi

OUTPUT_DIR="cls_cross_advanced"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"
DATASET_TYPE="modelnet40" # HDF5 training entry supports modelnet40 branch only; ModelNet10 subset also uses this branch

echo "==> Starting PointNet Advanced training (ModelNet10 subset, label smoothing + scale augment + feature transform)..."
python "${TRAIN_SCRIPT}" \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type "${DATASET_TYPE}" \
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
echo "==> ModelNet10 Advanced training finished. Output files:"
echo "    Model checkpoints: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    Best model       : ${OUTPUT_DIR}/best_model.pth"
echo "    Per-epoch metrics: ${CSV_PATH}"
echo "    Per-epoch loss   : ${OUTPUT_DIR}/loss.txt"
echo "    Per-epoch acc    : ${OUTPUT_DIR}/accuracy.txt"
echo "    MeshLab exports  : ${OUTPUT_DIR}/meshlab_ply/*.ply"
