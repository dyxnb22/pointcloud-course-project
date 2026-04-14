#!/usr/bin/env bash
# train_dgcnn.sh - 在 ModelNet40 上运行 DGCNN 训练
set -e

if [ ! -d "dgcnn/pytorch" ]; then
  echo "错误：找不到 dgcnn/pytorch，请先运行: bash colab_final/colab_setup.sh"
  exit 1
fi

pip install -q h5py scikit-learn

cd dgcnn/pytorch
echo "==> 开始训练 DGCNN..."
python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40 \
  --epochs=20
