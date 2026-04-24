# Project Overview and Implementation Outcomes (Based on `colab_final`)

## 1. What this project does

This course project focuses on **3D point cloud classification**. The goal is to reproduce and compare PointNet and DGCNN on ModelNet datasets, then extend with cross-dataset validation and advanced experiments. The `colab_final/` directory is the consolidated entry for direct Google Colab execution, including setup, training, comparison, and packaging scripts for one-command reproducibility.

Core work includes:

- **PointNet Baseline**: train baseline classifier on ModelNet40;
- **Cross-dataset validation**: auto-build a ModelNet10 subset from ModelNet40 and retrain/evaluate;
- **Advanced 2.2 extension**: add label smoothing, scale augmentation, feature transform, weight decay, LR scheduling, and CSV logging;
- **SOTA comparison**: train DGCNN on ModelNet40;
- **Result organization and visualization**: generate training-curve comparisons and package artifacts.

## 2. Colab entry and data preparation

`colab_final/colab_setup.sh` performs one-command environment and data preparation:

- Clone dependency repositories: `pointnet.pytorch` and `dgcnn`;
- Install PointNet dependencies (`pip install -e`);
- Download and extract ModelNet40 HDF5 data with multi-mirror fallback;
- Auto-build **ModelNet10 subset** from ModelNet40;
- Create dataset symlink for DGCNN.

## 3. Training workflow and experiment design

### 3.1 PointNet Baseline (ModelNet40)

Script: `colab_final/train_baseline.sh`

- Train for 20 epochs, output to `cls/`;
- Generate `metrics.csv`, `loss.txt`, `accuracy.txt`;
- Auto-update `best_model.pth`.

### 3.2 Cross-dataset validation (ModelNet10 subset)

Script: `colab_final/train_cross_dataset.sh`

- Retrain on ModelNet10 subset;
- Output to `cls_cross/`;
- Evaluate cross-dataset generalization.

### 3.3 Advanced 2.2 (ModelNet40)

Script: `colab_final/train_advanced.sh`

Extensions implemented:

- **label smoothing**: reduce overfitting;
- **scale augment**: improve scale robustness;
- **feature transform**: enable T-Net alignment;
- **weight decay + cosine scheduler**;
- **CSV/TXT metric logging**;
- **MeshLab point cloud export** (with prediction/ground-truth labels).

Output directory: `cls_advanced/` (includes `metrics.csv`, `best_model.pth`, `meshlab_ply/*.ply`).

### 3.4 Advanced on ModelNet10 subset

Script: `colab_final/train_advanced_modelnet10.sh`

- Use the same advanced configuration on ModelNet10 subset;
- Output directory: `cls_cross_advanced/`.

### 3.5 DGCNN comparison

Script: `colab_final/train_dgcnn.sh`

- Train DGCNN on ModelNet40 (20 epochs);
- Output directory: `dgcnn/pytorch/checkpoints/dgcnn_test/`.

## 4. Results and implementation outcomes

### 4.1 Accuracy summary

| Method | Dataset | Final Test Accuracy | Notes |
| --- | --- | ---: | --- |
| PointNet Baseline | ModelNet40 | 76.9% | late-epoch drop at 20 epochs (best=80.5%) |
| PointNet Baseline | ModelNet10 | 81.6% | cross-dataset generalization observed (best=86.6%) |
| PointNet Advanced | ModelNet10 | 88.0% | +6.4pp over baseline final |
| DGCNN | ModelNet40 | 84.7% | better than PointNet due to local structure modeling |
| PointNet Advanced (2.2) | ModelNet40 | 80.8% | +3.9pp over baseline final |

### 4.2 Training curve comparison

`colab_final/plot_compare.py` reads `metrics.csv` and generates three comparison images:

- `curve_compare_loss.png`
- `curve_compare_accuracy.png`
- `curve_compare_lr.png`

ModelNet10 comparison images:

- `curve_compare_modelnet10_loss.png`
- `curve_compare_modelnet10_accuracy.png`
- `curve_compare_modelnet10_lr.png`

### 4.3 Failure cases and limitations

From `meshlab_ply` sample analysis:

- PointNet tends to confuse furniture / similar shapes (`night_stand` ↔ `dresser`, etc.);
- Global aggregation is less sensitive to local structure;
- Overfitting can appear late in training (training acc up, test acc down).

## 5. Packaging and deliverables

### 5.1 ModelNet10 comparison package

Script: `colab_final/package_modelnet10_compare.sh`

- Collect `cls_cross` and `cls_cross_advanced`;
- Auto-generate comparison curves;
- Output `modelnet10_compare/` and `modelnet10_compare.zip`.

### 5.2 Final submission package

Script: `colab_final/package_final.sh`

- Aggregate PointNet and DGCNN artifacts;
- Auto-generate training curves;
- Output `final/` and `final_submission.zip`;
- Include README, comparison plots, model weights, logs, and other evidence files.

## 6. Summary

`colab_final` integrates the full reproducible Colab pipeline for **point cloud classification training, cross-dataset validation, advanced improvements, SOTA comparison, visualization analysis, and final packaging**. The outcomes show:

- Advanced consistently improves over baseline on both ModelNet40 and ModelNet10;
- DGCNN achieves higher overall accuracy than PointNet;
- Training curves and MeshLab samples provide intuitive evidence of model behavior and failure cases;
- One-command scripts improve reproducibility and submission completeness.
