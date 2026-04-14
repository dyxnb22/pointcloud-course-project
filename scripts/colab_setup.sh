#!/usr/bin/env bash
# colab_setup.sh - Clone repos, install deps, and prepare ModelNet40 + ShapeNet on Colab
set -euo pipefail

echo "==> 1. Cloning repositories..."
[ ! -d "pointnet.pytorch" ] && git clone https://github.com/fxia22/pointnet.pytorch.git || echo "PointNet already exists, skip."
[ ! -d "dgcnn" ] && git clone https://github.com/WangYueFt/dgcnn.git || echo "DGCNN already exists, skip."

echo "==> 2. Installing dependencies..."
pip install -e ./pointnet.pytorch

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
    "https://raw.githubusercontent.com/charlesq34/pointnet/2618f72bc1a0fd21b074096e748016960d44ef55/data/modelnet40_ply_hdf5_2048.zip"
    "https://raw.githubusercontent.com/yanx27/Pointnet_Pointnet2_pytorch/eb64fe0b4c24055559cea26299cb485dcb43d8dd/data/modelnet40_ply_hdf5_2048.zip"
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
import time
import urllib.error
import urllib.request
import zipfile

data_dir = sys.argv[1]
target_dir = sys.argv[2]
temp_dir = os.path.join(data_dir, "temp_shapenet")
os.makedirs(temp_dir, exist_ok=True)
zip_name = "shapenetcore_partanno_segmentation_benchmark_v0.zip"
manual_zip_env = os.environ.get("SHAPENET_ZIP_PATH", "").strip()
mirror_urls_env = os.environ.get("SHAPENET_URLS", "").strip()
connect_timeout = int(os.environ.get("MIRROR_TIMEOUT_SECONDS", "90"))
total_timeout = int(os.environ.get("MIRROR_TOTAL_TIMEOUT_SECONDS", "900"))
download_zip_path = os.path.join(data_dir, f".{zip_name}.download")

default_mirror_urls = [
    "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip",
    "https://raw.githubusercontent.com/charlesq34/pointnet/2618f72bc1a0fd21b074096e748016960d44ef55/shapenetcore_partanno_segmentation_benchmark_v0.zip",
]
if mirror_urls_env:
    mirror_urls = [u.strip() for u in mirror_urls_env.split(",") if u.strip()]
else:
    mirror_urls = default_mirror_urls

ALLOWED_CONTENT_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
}

def reset_temp_dir(work_dir):
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

def extract_and_organize(zip_path, source_name, work_dir, output_dir):
    print(f"Using ShapeNet archive from {source_name}: {zip_path}")
    reset_temp_dir(work_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)

    os.makedirs(output_dir, exist_ok=True)
    moved = False
    for root, _, files in os.walk(work_dir):
        if "synsetoffset2category.txt" in files:
            for item in os.listdir(root):
                src = os.path.join(root, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            moved = True
            break

    if not moved:
        for item in os.listdir(work_dir):
            src = os.path.join(work_dir, item)
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def download_file(url, dst):
    start_time = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=connect_timeout) as response:
            content_type_header = response.headers.get("Content-Type", "")
            normalized_content_type = content_type_header.lower().split(";", 1)[0].strip()
            if not normalized_content_type:
                print(f"[warning] Missing Content-Type for {url}; continuing with archive extraction validation.")
            elif normalized_content_type not in ALLOWED_CONTENT_TYPES:
                raise RuntimeError(
                    f"Unexpected content type for {url}: {content_type_header}"
                )
            timeout_reached = False
            check_timeout_every_chunks = 8
            chunk_counter = 0
            with open(dst, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    chunk_counter += 1
                    if chunk_counter >= check_timeout_every_chunks:
                        if time.monotonic() - start_time >= total_timeout:
                            timeout_reached = True
                            break
                        chunk_counter = 0
            if timeout_reached:
                safe_remove(dst)
                raise TimeoutError(f"total download timeout exceeded: {total_timeout}s")
    except urllib.error.HTTPError as e:
        safe_remove(dst)
        raise RuntimeError(f"HTTP {e.code} while downloading {url}") from e
    except Exception:
        safe_remove(dst)
        raise

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
            extract_and_organize(local_zip, local_zip, temp_dir, target_dir)
            success = True
            break
        except Exception as e:
            print(f"Local archive failed: {local_zip} ({e})", file=sys.stderr)

    if not success:
        for mirror_url in mirror_urls:
            print(f"Trying ShapeNet mirror: {mirror_url}")
            try:
                download_file(mirror_url, download_zip_path)
                extract_and_organize(download_zip_path, f"mirror {mirror_url}", temp_dir, target_dir)
                success = True
                break
            except Exception as e:
                print(f"Mirror failed: {mirror_url} ({e})", file=sys.stderr)
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(download_zip_path):
        try:
            os.remove(download_zip_path)
        except OSError as e:
            print(f"[cleanup-warning] failed to remove download file: {e}", file=sys.stderr)

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
