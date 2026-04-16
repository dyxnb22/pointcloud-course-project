#!/usr/bin/env bash
# eval_model10_pretrained.sh - 使用已训练权重在 ModelNet10 上单独评测并导出对比图
set -euo pipefail

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
BASELINE_MODEL="cls/best_model.pth"
ADVANCED_MODEL="cls_advanced/best_model.pth"
OUT_DIR="model10_eval"

usage() {
  echo "用法:"
  echo "  bash scripts/eval_model10_pretrained.sh [--dataset PATH] [--baseline_model PATH] [--advanced_model PATH] [--out_dir DIR]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --baseline_model)
      BASELINE_MODEL="$2"
      shift 2
      ;;
    --advanced_model)
      ADVANCED_MODEL="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash scripts/colab_setup.sh"
  exit 1
fi
if [ ! -f "${BASELINE_MODEL}" ]; then
  echo "错误：找不到 baseline 权重: ${BASELINE_MODEL}"
  exit 1
fi
if [ ! -f "${ADVANCED_MODEL}" ]; then
  echo "错误：找不到 advanced 权重: ${ADVANCED_MODEL}"
  exit 1
fi
if [ ! -f "colab_final/plot_compare.py" ]; then
  echo "错误：找不到绘图脚本 colab_final/plot_compare.py"
  exit 1
fi

BASE_OUT="${OUT_DIR}/baseline"
ADV_OUT="${OUT_DIR}/advanced"
mkdir -p "${BASE_OUT}" "${ADV_OUT}"

echo "==> [1/3] 评测 baseline 权重到 ModelNet10..."
python scripts/eval_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --model "${BASELINE_MODEL}" \
  --out_dir "${BASE_OUT}"

echo "==> [2/3] 评测 advanced 权重到 ModelNet10..."
python scripts/eval_classification_h5.py \
  --dataset "${DATASET_PATH}" \
  --model "${ADVANCED_MODEL}" \
  --out_dir "${ADV_OUT}"

echo "==> [3/3] 生成对比图..."
python colab_final/plot_compare.py \
  --baseline "${BASE_OUT}/metrics.csv" \
  --advanced "${ADV_OUT}/metrics.csv" \
  --out "${OUT_DIR}/curve_compare.png" \
  --title "ModelNet10 Eval (Pretrained Weights)"

echo ""
echo "✅ 已完成 ModelNet10 独立评测，输出目录：${OUT_DIR}"
echo "   - ${BASE_OUT}/metrics.csv"
echo "   - ${ADV_OUT}/metrics.csv"
echo "   - ${OUT_DIR}/curve_compare_loss.png"
echo "   - ${OUT_DIR}/curve_compare_accuracy.png"
echo "   - ${OUT_DIR}/curve_compare_lr.png"
