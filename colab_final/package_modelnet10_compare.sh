#!/usr/bin/env bash
# package_modelnet10_compare.sh - Collect ModelNet10 Baseline vs Advanced outputs and package them
# Run from repo root: bash colab_final/package_modelnet10_compare.sh
# Or from this folder: bash package_modelnet10_compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="modelnet10_compare"
ZIP_PATH="modelnet10_compare.zip"
COMPARE_OUT_BASENAME="${OUTPUT_DIR}/curve_compare_modelnet10.png"
COMPARE_LOSS_PNG="${OUTPUT_DIR}/curve_compare_modelnet10_loss.png"
COMPARE_ACC_PNG="${OUTPUT_DIR}/curve_compare_modelnet10_accuracy.png"
COMPARE_LR_PNG="${OUTPUT_DIR}/curve_compare_modelnet10_lr.png"

rm -rf "${OUTPUT_DIR}" "${ZIP_PATH}"
mkdir -p "${OUTPUT_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -r "${src}" "${dst}"
    echo "[OK] Collected: ${src} -> ${dst}"
  else
    echo "[WARN] Not found: ${src}"
  fi
}

generate_compare_plot() {
  local plot_script="${SCRIPT_DIR}/plot_compare.py"
  local baseline_csv="cls_cross/metrics.csv"
  local advanced_csv="cls_cross_advanced/metrics.csv"

  if [ ! -f "${plot_script}" ]; then
    echo "[WARN] Compare plotting script not found: ${plot_script}"
    return 0
  fi

  if [ ! -f "${baseline_csv}" ] || [ ! -f "${advanced_csv}" ]; then
    echo "[WARN] Required metrics.csv files for compare plot not found, skipping"
    return 0
  fi

  echo "==> Generating ModelNet10 comparison plots (3 images)"
  if MPLBACKEND=Agg python3 "${plot_script}" \
    --baseline "${baseline_csv}" \
    --advanced "${advanced_csv}" \
    --out "${COMPARE_OUT_BASENAME}" \
    --title "ModelNet10 Curves: Baseline vs Advanced"; then
    echo "[OK] Generated: ${COMPARE_LOSS_PNG}, ${COMPARE_ACC_PNG}, ${COMPARE_LR_PNG}"
  else
    echo "[WARN] Compare plot generation failed, continuing packaging"
  fi
}

copy_if_exists "cls_cross" "${OUTPUT_DIR}/cls_cross"
copy_if_exists "cls_cross_advanced" "${OUTPUT_DIR}/cls_cross_advanced"

generate_compare_plot

MANIFEST="${OUTPUT_DIR}/MANIFEST.txt"
{
  echo "Generated at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo ""
  echo "Included files/directories:"
  (cd "${OUTPUT_DIR}" && find . -mindepth 1 | sort)
} > "${MANIFEST}"

echo "==> Packaging ${ZIP_PATH}"
zip -r "${ZIP_PATH}" "${OUTPUT_DIR}" 1>/dev/null

echo ""
echo "Packaging complete: ${REPO_ROOT}/${ZIP_PATH}"
echo "Use the following in Colab to download:"
echo "from google.colab import files"
echo "files.download('${ZIP_PATH}')"
