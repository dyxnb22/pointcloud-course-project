#!/usr/bin/env bash
# eval_model10_pretrained.sh - 使用已训练权重在 ModelNet10 上单独评测并导出对比图
# 可从仓库根目录运行：bash colab_final/eval_model10_pretrained.sh
# 也可进入文件夹后运行：bash eval_model10_pretrained.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="pointnet.pytorch/data/modelnet10_ply_hdf5_2048"
BASELINE_MODEL="cls/best_model.pth"
ADVANCED_MODEL="cls_advanced/best_model.pth"
OUT_DIR="model10_eval"
ZIP_PATH=""

usage() {
  echo "用法:"
  echo "  bash colab_final/eval_model10_pretrained.sh [--dataset PATH] [--baseline_model PATH] [--advanced_model PATH] [--out_dir DIR] [--zip_path FILE]"
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
    --zip_path)
      ZIP_PATH="$2"
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

if [ -z "${ZIP_PATH}" ]; then
  ZIP_PATH="${OUT_DIR%/}.zip"
fi

if [ ! -d "pointnet.pytorch" ]; then
  echo "错误：找不到 pointnet.pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
  echo "错误：找不到数据集文件夹 ${DATASET_PATH}，请先运行: bash colab_final/colab_setup.sh"
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

BASE_OUT="${OUT_DIR}/baseline"
ADV_OUT="${OUT_DIR}/advanced"
mkdir -p "${BASE_OUT}" "${ADV_OUT}"

echo "==> [1/3] 评测 baseline 权重到 ModelNet10..."
python "${SCRIPT_DIR}/eval_classification_h5.py" \
  --dataset "${DATASET_PATH}" \
  --model "${BASELINE_MODEL}" \
  --out_dir "${BASE_OUT}"

echo "==> [2/3] 评测 advanced 权重到 ModelNet10..."
python "${SCRIPT_DIR}/eval_classification_h5.py" \
  --dataset "${DATASET_PATH}" \
  --model "${ADVANCED_MODEL}" \
  --out_dir "${ADV_OUT}"

echo "==> [3/3] 生成对比图..."
python "${SCRIPT_DIR}/plot_compare.py" \
  --baseline "${BASE_OUT}/metrics.csv" \
  --advanced "${ADV_OUT}/metrics.csv" \
  --out "${OUT_DIR}/curve_compare.png" \
  --title "ModelNet10 Eval (Pretrained Weights)"

if ! command -v zip >/dev/null 2>&1; then
  echo "错误：系统未安装 zip，无法打包。请先安装 zip 命令。"
  exit 1
fi
echo "==> [4/4] 打包输出目录..."
rm -f "${ZIP_PATH}"
zip -r "${ZIP_PATH}" "${OUT_DIR}" >/dev/null

echo ""
echo "✅ 已完成 ModelNet10 独立评测，输出目录：${OUT_DIR}"
echo "   - ${BASE_OUT}/metrics.csv"
echo "   - ${ADV_OUT}/metrics.csv"
echo "   - ${OUT_DIR}/curve_compare_loss.png"
echo "   - ${OUT_DIR}/curve_compare_accuracy.png"
echo "   - ${OUT_DIR}/curve_compare_lr.png"
echo "   - ${ZIP_PATH}"
