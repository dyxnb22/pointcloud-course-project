#!/usr/bin/env bash
# train_advanced.sh - Advanced Requirement 2.2
# Trains PointNet on ModelNet40 with methodological enhancements:
#   1. Label smoothing (--label_smoothing 0.05): regularises predictions.
#   2. Random scale augmentation (--scale_augment): scale factor in [0.8, 1.25].
#   3. Feature transform (--feature_transform): enables learned feature alignment.
#   4. Per-epoch CSV/TXT logging (--log_csv): writes epoch metrics and text logs.
#   5. MeshLab export (--meshlab_dir): writes per-epoch sample .ply files.
# Run from repo root: bash colab_final/train_advanced.sh
# Or from this folder: bash train_advanced.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "pointnet.pytorch" ]; then
  echo "Error: pointnet.pytorch not found. Run: bash colab_final/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet40_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "Error: dataset folder ${DATASET_PATH} not found. Run: bash colab_final/colab_setup.sh"
  exit 1
fi

OUTPUT_DIR="cls_advanced"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> [2.2] Starting PointNet advanced experiment (label smoothing + scale augment + feature transform)..."
python "${SCRIPT_DIR}/train_classification_h5.py" \
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
echo "==> [2.2] Training finished. Output files:"
echo "    Model checkpoints: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    Best model       : ${OUTPUT_DIR}/best_model.pth"
echo "    Per-epoch metrics: ${CSV_PATH}"
echo "    Per-epoch loss   : ${OUTPUT_DIR}/loss.txt"
echo "    Per-epoch acc    : ${OUTPUT_DIR}/accuracy.txt"
echo "    MeshLab exports  : ${OUTPUT_DIR}/meshlab_ply/*.ply"
