#!/usr/bin/env bash
# colab_setup.sh - Clone repos, install deps, and prepare ModelNet40 + ModelNet10-subset on Colab
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

echo "==> 4. Preparing ModelNet10 subset (for cross-dataset validation)..."
MODELNET10_TARGET="${DATA_DIR}/modelnet10_ply_hdf5_2048"

if [ -d "${MODELNET10_TARGET}" ] && [ "$(ls -A "${MODELNET10_TARGET}" 2>/dev/null)" ]; then
  echo "ModelNet10 subset already exists, skip."
else
  python - "${MODELNET_DIR}" "${MODELNET10_TARGET}" <<'PY'
import glob
import os
import shutil
import sys

import h5py
import numpy as np

source_dir = sys.argv[1]
target_dir = sys.argv[2]
source_dataset_name = "modelnet40_ply_hdf5_2048"
target_dataset_name = "modelnet10_ply_hdf5_2048"

selected_classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

shape_names_path = os.path.join(source_dir, "shape_names.txt")
if not os.path.isfile(shape_names_path):
    raise FileNotFoundError(f"shape_names.txt not found in {source_dir}")

with open(shape_names_path, "r", encoding="utf-8") as f:
    source_classes = [line.strip() for line in f if line.strip()]

source_class_to_id = {name: idx for idx, name in enumerate(source_classes)}
missing = [name for name in selected_classes if name not in source_class_to_id]
if missing:
    raise RuntimeError(
        f"Missing classes in ModelNet40 source: {missing}. "
        f"Available classes: {source_classes}"
    )

selected_source_ids = [source_class_to_id[name] for name in selected_classes]
old_to_new_label = {old_id: new_id for new_id, old_id in enumerate(selected_source_ids)}
label_mapping = np.full(len(source_classes), -1, dtype=np.int64)
for old_id, new_id in old_to_new_label.items():
    label_mapping[old_id] = new_id

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)


def resolve_list_file_entry(split_name: str, default_filename: str) -> str:
    # Reuse source list-path style when possible (e.g. data/modelnetXX/...),
    # otherwise safely fall back to the local output filename.
    candidate_list = os.path.join(source_dir, f"{split_name}_files.txt")
    if not os.path.isfile(candidate_list):
        return default_filename

    with open(candidate_list, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "/" not in line:
                return default_filename
            prefix = line.rsplit("/", 1)[0]
            prefix_parts = prefix.split("/")
            replaced = False
            for idx, part in enumerate(prefix_parts):
                if part == source_dataset_name:
                    prefix_parts[idx] = target_dataset_name
                    replaced = True
            if not replaced:
                return default_filename
            prefix = "/".join(prefix_parts)
            return f"{prefix}/{default_filename}"
    return default_filename


def build_split(split: str):
    h5_files = sorted(glob.glob(os.path.join(source_dir, f"ply_data_{split}*.h5")))
    if not h5_files:
        raise RuntimeError(f"No source H5 files found for split={split} in {source_dir}")

    kept_data = []
    kept_label = []
    for path in h5_files:
        with h5py.File(path, "r") as h5f:
            data = h5f["data"][:]
            label = h5f["label"][:]
        label_flat = label.reshape(-1)
        mask = np.isin(label_flat, selected_source_ids)
        if not np.any(mask):
            continue
        filtered_data = data[mask]
        remapped_label = label_mapping[label_flat[mask]].reshape(-1, 1)
        kept_data.append(filtered_data)
        kept_label.append(remapped_label)

    if not kept_data:
        raise RuntimeError(f"No samples kept for split={split}; source files may be incompatible.")

    out_data = np.concatenate(kept_data, axis=0)
    out_label = np.concatenate(kept_label, axis=0)
    out_h5 = f"ply_data_{split}0.h5"
    out_h5_path = os.path.join(target_dir, out_h5)

    with h5py.File(out_h5_path, "w") as h5f:
        h5f.create_dataset("data", data=out_data, compression="gzip", compression_opts=4)
        h5f.create_dataset("label", data=out_label, compression="gzip", compression_opts=4)

    return out_h5


train_h5 = build_split("train")
test_h5 = build_split("test")

with open(os.path.join(target_dir, "shape_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(selected_classes))

train_files_list_entry = resolve_list_file_entry("train", train_h5)
test_files_list_entry = resolve_list_file_entry("test", test_h5)
for filename, list_entry in (
    ("train_files.txt", train_files_list_entry),
    ("test_files.txt", test_files_list_entry),
):
    with open(os.path.join(target_dir, filename), "w", encoding="utf-8") as f:
        f.write(f"{list_entry}\n")

for filename, generated_h5_name in (
    ("train_hdf5_file_list.txt", train_h5),
    ("test_hdf5_file_list.txt", test_h5),
):
    with open(os.path.join(target_dir, filename), "w", encoding="utf-8") as f:
        f.write(f"{generated_h5_name}\n")

print(f"Prepared ModelNet10 subset at: {target_dir}")
PY
fi

if [ ! -d "${MODELNET10_TARGET}" ]; then
  echo "Error: ModelNet10 subset directory not found: ${MODELNET10_TARGET}" >&2
  exit 1
fi

echo "==> Setup complete."
