#!/usr/bin/env bash
# train_advanced.sh - Advanced Requirement 2.2
# Trains PointNet on ModelNet40 with two methodological enhancements:
#   1. Label smoothing (--label_smoothing 0.1): regularises the classifier by
#      preventing over-confident predictions, which improves generalisation.
#   2. Random scale augmentation (--scale_augment): multiplies each training
#      point cloud by a uniform random scale in [0.8, 1.25], making the model
#      robust to size/resolution variation in real-world scans.
#   3. Feature transform (--feature_transform): aligns point features via a
#      learned 64-D T-Net (already available in baseline; enabled here for the
#      advanced configuration).
#   4. Per-epoch CSV logging (--log_csv): writes epoch, train_loss, train_acc,
#      test_acc to cls_advanced/metrics.csv for easy final reporting.
#
# 可从仓库根目录运行：bash colab_final/train_advanced.sh
# 也可进入文件夹后运行：bash train_advanced.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet40_ply_hdf5_2048"
if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

OUTPUT_DIR="cls_advanced"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> [2.2] 开始训练 PointNet 高级扩展实验（label smoothing + scale augment + feature transform）..."
python "${SCRIPT_DIR}/train_classification_h5.py" \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --feature_transform \
  --label_smoothing 0.1 \
  --scale_augment \
  --outf "${OUTPUT_DIR}" \
  --log_csv "${CSV_PATH}"

echo ""
echo "==> [2.2] 训练完成。输出文件："
echo "    模型权重: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    每轮指标: ${CSV_PATH}"
