#!/usr/bin/env bash
# train_cross_dataset.sh - Run PointNet classification on ModelNet10 subset (cross-dataset validation)
# Note: Uses this repo's HDF5-compatible training entry and keeps dataset_type=modelnet40.
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

OUTPUT_DIR="cls_cross"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> Starting PointNet classification training (cross-dataset: ModelNet10 subset, modelnet40 loading branch)..."
python scripts/train_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --outf "${OUTPUT_DIR}" \
  --log_csv "${CSV_PATH}"

echo ""
echo "==> Cross-dataset training finished. Output files:"
echo "    Model checkpoints: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    Best model       : ${OUTPUT_DIR}/best_model.pth"
echo "    Per-epoch metrics: ${CSV_PATH}"
echo "    Per-epoch loss   : ${OUTPUT_DIR}/loss.txt"
echo "    Per-epoch acc    : ${OUTPUT_DIR}/accuracy.txt"
