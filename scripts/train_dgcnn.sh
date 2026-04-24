#!/usr/bin/env bash
# train_dgcnn.sh - Run DGCNN training on ModelNet40
set -e

if [ ! -d "dgcnn/pytorch" ]; then
  echo "Error: dgcnn/pytorch not found. Run: bash scripts/colab_setup.sh"
  exit 1
fi

pip install -q h5py scikit-learn

cd dgcnn/pytorch
echo "==> Starting DGCNN training..."
python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40 \
  --epochs=20
