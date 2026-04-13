#!/usr/bin/env bash
# colab_setup.sh — Clone repos and install dependencies on Google Colab (T4 GPU)
set -euo pipefail

echo "==> Cloning PointNet..."
if [ ! -d "pointnet.pytorch" ]; then
  git clone https://github.com/fxia22/pointnet.pytorch.git
else
  echo "pointnet.pytorch already exists, skip clone."
fi

echo "==> Cloning DGCNN..."
if [ ! -d "dgcnn" ]; then
  git clone https://github.com/WangYueFt/dgcnn.git
else
  echo "dgcnn already exists, skip clone."
fi

echo "==> Installing PointNet package..."
pip install -e ./pointnet.pytorch

echo "==> Downloading ModelNet40 dataset..."
DATA_DIR="pointnet.pytorch/data"
DATASET_DIR="${DATA_DIR}/modelnet40_ply_hdf5_2048"
ZIP_PATH="${DATA_DIR}/modelnet40_ply_hdf5_2048.zip"
MIRROR_TIMEOUT_SECONDS="${MIRROR_TIMEOUT_SECONDS:-90}"

if [ -d "${DATASET_DIR}" ]; then
  echo "ModelNet40 dataset already exists, skip download."
else
  mkdir -p "${DATA_DIR}"

  mirrors=(
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    "https://github.com/charlesq34/pointnet/raw/master/data/modelnet40_ply_hdf5_2048.zip"
    "https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/data/modelnet40_ply_hdf5_2048.zip"
  )

  downloaded=0
  for mirror in "${mirrors[@]}"; do
    echo "Trying mirror: ${mirror}"
    if python - "$mirror" "${ZIP_PATH}" "${MIRROR_TIMEOUT_SECONDS}" <<'PY'
import socket
import sys
import urllib.request

url = sys.argv[1]
dst = sys.argv[2]
timeout = int(sys.argv[3])
socket.setdefaulttimeout(timeout)

try:
    with urllib.request.urlopen(url, timeout=timeout) as response, open(dst, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
except Exception as e:
    print(f"[download-error] {e}", file=sys.stderr)
    sys.exit(1)
PY
    then
      downloaded=1
      break
    fi
    echo "Mirror failed, trying next..."
  done

  if [ "${downloaded}" -ne 1 ]; then
    echo "Error: all dataset mirrors failed (timeout=${MIRROR_TIMEOUT_SECONDS}s)." >&2
    exit 1
  fi

  unzip -oq "${ZIP_PATH}" -d "${DATA_DIR}"
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "Error: ModelNet40 dataset directory not found after setup: ${DATASET_DIR}" >&2
  exit 1
fi

echo "==> Setup complete."
