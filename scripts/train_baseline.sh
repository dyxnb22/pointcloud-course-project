#!/usr/bin/env bash
# train_baseline.sh - 在 ModelNet40 上运行 PointNet 训练（主线）
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

OUTPUT_DIR="cls"
CSV_PATH="${OUTPUT_DIR}/metrics.csv"

echo "==> 开始训练 PointNet 分类模型..."
python scripts/train_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --nepoch=20 \
  --dataset_type modelnet40 \
  --outf "${OUTPUT_DIR}" \
  --log_csv "${CSV_PATH}"

echo ""
echo "==> baseline 训练完成。输出文件："
echo "    模型权重: ${OUTPUT_DIR}/cls_model_*.pth"
echo "    最优模型: ${OUTPUT_DIR}/best_model.pth"
echo "    每轮指标: ${CSV_PATH}"
echo "    每轮loss: ${OUTPUT_DIR}/loss.txt"
echo "    每轮acc : ${OUTPUT_DIR}/accuracy.txt"
