#!/usr/bin/env bash
# package_final.sh - Collect Colab training artifacts and package them as a submission zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FINAL_DIR="final"
ZIP_PATH="final_submission.zip"
COMPARE_OUT_BASENAME="curve_compare.png"
COMPARE_LOSS_PNG="curve_compare_loss.png"
COMPARE_ACC_PNG="curve_compare_accuracy.png"
COMPARE_LR_PNG="curve_compare_lr.png"

rm -rf "${FINAL_DIR}" "${ZIP_PATH}"
mkdir -p "${FINAL_DIR}"

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
  local plot_script="colab_final/plot_compare.py"
  local baseline_csv="cls/metrics.csv"
  local advanced_csv="cls_advanced/metrics.csv"

  if [ ! -f "${plot_script}" ]; then
    echo "[WARN] Compare plotting script not found: ${plot_script}"
    return 0
  fi

  if [ ! -f "${baseline_csv}" ] || [ ! -f "${advanced_csv}" ]; then
    echo "[WARN] Required metrics.csv files for compare plot not found, skipping"
    return 0
  fi

  echo "==> Generating compare plots (3 images)"
  if MPLBACKEND=Agg python3 "${plot_script}" \
    --baseline "${baseline_csv}" \
    --advanced "${advanced_csv}" \
    --out "${COMPARE_OUT_BASENAME}"; then
    echo "[OK] Generated: ${COMPARE_LOSS_PNG}, ${COMPARE_ACC_PNG}, ${COMPARE_LR_PNG}"
  else
    echo "[WARN] Compare plot generation failed, continuing packaging"
  fi
}

# PointNet artifact directories (common for course submission)
copy_if_exists "cls" "${FINAL_DIR}/cls"
copy_if_exists "cls_cross" "${FINAL_DIR}/cls_cross"
copy_if_exists "cls_advanced" "${FINAL_DIR}/cls_advanced"

# DGCNN key artifacts
copy_if_exists "dgcnn/pytorch/checkpoints/dgcnn_test" "${FINAL_DIR}/dgcnn_test"

# Extra evidence and documentation (if present)
copy_if_exists "assets/meshlab" "${FINAL_DIR}/assets/meshlab"
copy_if_exists "results" "${FINAL_DIR}/results"
copy_if_exists "README.md" "${FINAL_DIR}/README.md"
copy_if_exists "colab_final/README.md" "${FINAL_DIR}/colab_final_README.md"

# Training curve comparison images (package them if generated)
generate_compare_plot
copy_if_exists "${COMPARE_LOSS_PNG}" "${FINAL_DIR}/${COMPARE_LOSS_PNG}"
copy_if_exists "${COMPARE_ACC_PNG}" "${FINAL_DIR}/${COMPARE_ACC_PNG}"
copy_if_exists "${COMPARE_LR_PNG}" "${FINAL_DIR}/${COMPARE_LR_PNG}"

# Generate manifest
MANIFEST="${FINAL_DIR}/MANIFEST.txt"
{
  echo "Generated at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo ""
  echo "Included files/directories:"
  (cd "${FINAL_DIR}" && find . -mindepth 1 | sort)
} > "${MANIFEST}"

echo "==> Packaging ${ZIP_PATH}"
zip -r "${ZIP_PATH}" "${FINAL_DIR}" >/dev/null

echo ""
echo "Packaging complete: ${REPO_ROOT}/${ZIP_PATH}"
echo "Use the following in Colab to download:"
echo "from google.colab import files"
echo "files.download('${ZIP_PATH}')"
