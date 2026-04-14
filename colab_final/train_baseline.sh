#!/usr/bin/env bash
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

OUTPUT_DIR="cls"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> 开始训练 PointNet baseline..."
python "${SCRIPT_DIR}/train_classification_h5.py" \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --outf "${OUTPUT_DIR}" \
  --log_csv "${CSV_PATH}"

echo ""
echo "==> baseline 训练完成。输出文件："
echo "    模型权重: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    每轮指标: ${CSV_PATH}"