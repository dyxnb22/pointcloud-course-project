#!/usr/bin/env bash
# colab_setup.sh - Clone repos, install deps, and prepare ModelNet40 + ShapeNet on Colab
set -euo pipefail

echo "==> 1. Cloning repositories..."
[ ! -d "pointnet.pytorch" ] && git clone https://github.com/fxia22/pointnet.pytorch.git || echo "PointNet already exists, skip."
[ ! -d "dgcnn" ] && git clone https://github.com/WangYueFt/dgcnn.git || echo "DGCNN already exists, skip."

echo "==> 2. Installing dependencies..."
pip install -e ./pointnet.pytorch
pip install -q huggingface_hub

DATA_DIR="pointnet.pytorch/data"
mkdir -p "${DATA_DIR}"

echo "==> 3. Preparing ModelNet40..."
MODELNET_DIR="${DATA_DIR}/modelnet40_ply_hdf5_2048"
if [ -d "${MODELNET_DIR}" ] && [ "$(ls -A "${MODELNET_DIR}" 2>/dev/null)" ]; then
  echo "ModelNet40 already exists, skip."
else
  MODELNET_ZIP="${DATA_DIR}/modelnet40_ply_hdf5_2048.zip"
  MIRROR_TIMEOUT_SECONDS="${MIRROR_TIMEOUT_SECONDS:-90}"
  MIRROR_TOTAL_TIMEOUT_SECONDS="${MIRROR_TOTAL_TIMEOUT_SECONDS:-900}"

  mirrors=(
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    "https://github.com/charlesq34/pointnet/raw/master/data/modelnet40_ply_hdf5_2048.zip"
    "https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/data/modelnet40_ply_hdf5_2048.zip"
  )

  downloaded=0
  for mirror in "${mirrors[@]}"; do
    rm -f "${MODELNET_ZIP}"
    echo "Trying ModelNet40 mirror: ${mirror}"
    if python - "$mirror" "${MODELNET_ZIP}" "${MIRROR_TIMEOUT_SECONDS}" "${MIRROR_TOTAL_TIMEOUT_SECONDS}" <<'PY'
import os
import sys
import time
import urllib.request

url = sys.argv[1]
dst = sys.argv[2]
connect_timeout = int(sys.argv[3])
total_timeout = int(sys.argv[4])
start_time = time.monotonic()

try:
    with urllib.request.urlopen(url, timeout=connect_timeout) as response, open(dst, "wb") as f:
        while True:
            if time.monotonic() - start_time >= total_timeout:
                raise TimeoutError(f"total download timeout exceeded: {total_timeout}s")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
except Exception as e:
    try:
        if os.path.exists(dst):
            os.remove(dst)
    except OSError as cleanup_error:
        print(f"[cleanup-warning] failed to remove partial file: {cleanup_error}", file=sys.stderr)
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
    echo "Error: all ModelNet40 mirrors failed." >&2
    exit 1
  fi

  if ! unzip -oq "${MODELNET_ZIP}" -d "${DATA_DIR}"; then
    echo "Error: failed to extract ModelNet40 archive: ${MODELNET_ZIP}" >&2
    exit 1
  fi
  rm -f "${MODELNET_ZIP}"
fi

mkdir -p dgcnn/pytorch/data
if ! ln -sfn "$(pwd)/${MODELNET_DIR}" "dgcnn/pytorch/data/modelnet40_ply_hdf5_2048"; then
  echo "Warning: failed to create ModelNet40 symlink for DGCNN." >&2
fi

if [ ! -d "${MODELNET_DIR}" ]; then
  echo "Error: ModelNet40 dataset directory not found: ${MODELNET_DIR}" >&2
  exit 1
fi

echo "==> 4. Preparing ShapeNet..."
SHAPENET_TARGET="${DATA_DIR}/shapenetcore_partanno_segmentation_benchmark_v0"
SHAPENET_COMPAT_TARGET="pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0"

if [ -d "${SHAPENET_TARGET}" ] && [ "$(ls -A "${SHAPENET_TARGET}" 2>/dev/null)" ]; then
  echo "ShapeNet already exists, skip."
else
  python - "${DATA_DIR}" "${SHAPENET_TARGET}" <<'PY'
import os
import shutil
import sys
import zipfile

from huggingface_hub import hf_hub_download

data_dir = sys.argv[1]
target_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", ""))
temp_dir = os.path.join(data_dir, "temp_shapenet")
os.makedirs(temp_dir, exist_ok=True)
zip_name = "shapenetcore_partanno_segmentation_benchmark_v0.zip"
manual_zip_env = os.environ.get("SHAPENET_ZIP_PATH", "").strip()

mirrors = [
    "gourmet/ShapeNetCore_partanno_segmentation_benchmark_v0",
    "wjh19/ShapeNetCore_partanno_segmentation_benchmark_v0",
    "jason233/ShapeNetCore_partanno_segmentation_benchmark_v0",
]

def reset_temp_dir():
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

def extract_and_organize(zip_path, source_name):
    print(f"Using ShapeNet archive from {source_name}: {zip_path}")
    reset_temp_dir()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    os.makedirs(target_dir, exist_ok=True)
    moved = False
    for root, _, files in os.walk(temp_dir):
        if "synsetoffset2category.txt" in files:
            for item in os.listdir(root):
                src = os.path.join(root, item)
                dst = os.path.join(target_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            moved = True
            break

    if not moved:
        for item in os.listdir(temp_dir):
            src = os.path.join(temp_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

success = False
try:
    local_zip_candidates = []
    if manual_zip_env:
        local_zip_candidates.append(os.path.expanduser(manual_zip_env))
    local_zip_candidates.extend(
        [
            os.path.join(data_dir, zip_name),
            os.path.join(os.getcwd(), zip_name),
            os.path.join("/content", zip_name),
        ]
    )

    checked = set()
    for local_zip in local_zip_candidates:
        if local_zip in checked:
            continue
        checked.add(local_zip)
        if not os.path.isfile(local_zip):
            continue
        try:
            extract_and_organize(local_zip, "local file")
            success = True
            break
        except Exception as e:
            print(f"Local archive failed: {local_zip} ({e})", file=sys.stderr)

    if not success:
        for repo in mirrors:
            print(f"Trying ShapeNet mirror: {repo}")
            try:
                kwargs = {
                    "repo_id": repo,
                    "filename": zip_name,
                    "repo_type": "dataset",
                }
                if token:
                    kwargs["token"] = token
                zip_path = hf_hub_download(**kwargs)
                extract_and_organize(zip_path, f"mirror {repo}")
                success = True
                break
            except Exception as e:
                print(f"Mirror failed: {repo} ({e})", file=sys.stderr)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

if not success:
    print(
        "Error: all ShapeNet mirrors and local archives failed. "
        "You can manually upload shapenetcore_partanno_segmentation_benchmark_v0.zip "
        "to the working directory, /content, or set SHAPENET_ZIP_PATH.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
fi

mkdir -p pointnet.pytorch
if ! ln -sfn "$(pwd)/${SHAPENET_TARGET}" "${SHAPENET_COMPAT_TARGET}"; then
  echo "Warning: failed to create ShapeNet compatibility symlink for PointNet." >&2
fi

if [ ! -d "${SHAPENET_TARGET}" ]; then
  echo "Error: ShapeNet dataset directory not found: ${SHAPENET_TARGET}" >&2
  exit 1
fi

echo "==> Setup complete."
