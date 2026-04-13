#!/usr/bin/env bash
# colab_setup.sh — Clone repos and install dependencies on Google Colab (T4 GPU)
set -e

echo "==> Cloning PointNet..."
git clone https://github.com/fxia22/pointnet.pytorch.git

echo "==> Cloning DGCNN..."
git clone https://github.com/WangYueFt/dgcnn.git

echo "==> Installing PointNet package..."
pip install -e ./pointnet.pytorch

echo "==> Downloading ModelNet40 dataset..."
cd pointnet.pytorch/scripts
bash download.sh
cd ../..

echo "==> Setup complete."
