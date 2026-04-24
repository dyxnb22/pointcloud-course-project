# Colab Final Runtime Folder Guide

This folder is the consolidated Colab-ready entry for final submission.

It centralizes scripts and code intended to run directly on Google Colab so that execution, reproduction, and packaging are straightforward.

## File Guide

- `colab_setup.sh`: clones dependency repositories, installs environment, downloads ModelNet40 with mirror fallback, and builds a ModelNet10 subset
- `train_classification_h5.py`: PointNet HDF5 training entry
- `train_baseline.sh`: PointNet baseline training (ModelNet40)
- `train_cross_dataset.sh`: PointNet cross-dataset training (ModelNet10 subset)
- `train_advanced_modelnet10.sh`: PointNet advanced training (ModelNet10 subset)
- `train_dgcnn.sh`: DGCNN comparison training
- `train_advanced.sh`: Advanced requirement 2.2 pipeline (label smoothing + scale augment + feature transform)
- `package_modelnet10_compare.sh`: package ModelNet10 baseline vs advanced outputs
- `package_final.sh`: package final submission outputs

## Recommended Execution Order (Colab)

1. Prepare environment:

```bash
bash colab_final/colab_setup.sh
```

2. Train baseline:

```bash
bash colab_final/train_baseline.sh
```

3. Train cross-dataset (ModelNet10 subset):

```bash
bash colab_final/train_cross_dataset.sh
```

4. Train DGCNN comparison:

```bash
bash colab_final/train_dgcnn.sh
```

5. Run advanced requirement 2.2:

```bash
bash colab_final/train_advanced.sh
```

6. Run ModelNet10 advanced:

```bash
bash colab_final/train_advanced_modelnet10.sh
```

7. Package ModelNet10 comparison:

```bash
bash colab_final/package_modelnet10_compare.sh
```

8. Package final submission:

```bash
bash colab_final/package_final.sh
```

---

## Results

### Accuracy Summary

| Method                    | Dataset                    | Final Test Accuracy | Paper-Reported Accuracy | Gap Analysis |
| ------------------------- | -------------------------- | ------------------- | ----------------------- | ------------ |
| PointNet Baseline         | ModelNet40                 | 74.9%               | 89.2%                   | Fill in: above/below paper and reasons |
| PointNet Baseline         | ModelNet10 (cross-dataset) | 82.7%               | —                       | Fill in: generalization explanation |
| DGCNN                     | ModelNet40                 | 84.7%               | 92.9%                   | Fill in: above/below paper and reasons |
| PointNet Advanced (2.2)   | ModelNet40                 | 77.6%               | —                       | Fill in: comparison vs baseline |

### Training Curve Comparison (Baseline vs Advanced)

![curve_compare_loss](curve_compare_loss.png)
![curve_compare_accuracy](curve_compare_accuracy.png)
![curve_compare_lr](curve_compare_lr.png)

### Paper Comparison Analysis

(Fill in: gap between your results and paper numbers, and possible causes such as epochs, augmentation settings, hardware differences, etc.)

### Failure Cases and Method Limitations

(Fill in: list at least 2–3 classes that are easy to confuse and analyze causes; describe PointNet limitations such as weak local structure modeling and scale sensitivity; propose possible improvements.)

### Advanced 2.2 Improvement Analysis

(Fill in: how much gain comes from label smoothing + scale augment + feature transform? Analyze each change's contribution and whether the outcome matches expectations.)

## Submission Checklist

### 2.1 Basic Requirements

- [ ] **1. Project introduction**: task definition (3D point cloud classification), I/O (point cloud→class), datasets (ModelNet40/10), reference paper (PointNet), motivation, and technical challenges documented in README/report
- [ ] **2. Environment setup**: `colab_setup.sh` runs end-to-end and README provides step-by-step commands (pip install / conda etc.)
- [ ] **3. Demo runnable**: `train_baseline.sh` runs and outputs loss/accuracy; README explains each step
- [ ] **4. Model training**: baseline training (ModelNet40) completed, `cls/cls_model_*.pth` produced and saved
- [ ] **5. Paper comparison**: final test accuracy recorded, compared with paper result (89.2%), with gap analysis and training curves/logs
- [ ] **6. Other dataset validation**: `train_cross_dataset.sh` (ModelNet10) run, cross-dataset accuracy recorded, generalization conclusion provided
- [ ] **7. Weakness analysis and improvements**: PointNet limitations analyzed with failure cases and improvement ideas
- [ ] **8. SOTA method implementation and comparison**: `train_dgcnn.sh` run and compared to PointNet baseline with method-difference analysis

### 2.2 Advanced Requirements

- [ ] **Method extension implemented**: `train_advanced.sh` run with label smoothing + scale augment + feature transform
- [ ] **CSV metrics**: `cls_advanced/metrics.csv` generated with complete per-epoch `epoch,train_loss,train_acc,test_acc`
- [ ] **Motivation explained**: README/report explains why each change is made and what issue it addresses
- [ ] **Result comparison and analysis**: Advanced vs baseline compared and analyzed; include explanation even if gains are limited
- [ ] **Evidence files complete**: `cls_advanced/metrics.csv` and final checkpoints saved as submission evidence

---

## Advanced Requirement 2.2 — Method Extension Details

### Motivation

PointNet baseline on ModelNet40 is mainly constrained by:
1. **Overfitting**: one-hot cross-entropy can lead to over-confident predictions and weaker generalization.
2. **Insufficient scale invariance**: baseline includes rotation and jitter only, but not natural scale variation.

### What Changed

Optional CLI flags added/extended in `train_classification_h5.py` with backward compatibility maintained:

| Flag | Type | Default | Purpose |
|------|------|--------|------|
| `--label_smoothing` | float | `0.0` | Label smoothing factor to reduce overfitting risk |
| `--scale_augment` | switch | off | Randomly scale each point cloud by ×[0.8, 1.25] during training |
| `--log_csv` | str | `""` | Per-epoch CSV output path (`epoch,train_loss,train_acc,test_loss,test_acc,lr`) and synced `loss.txt/accuracy.txt` |
| `--meshlab_dir` | str | `""` | Export MeshLab-readable `.ply` samples each epoch |
| `--meshlab_samples_per_epoch` | int | `0` | Number of exported test samples per epoch |
| `--weight_decay` | float | `0.0` | Adam weight decay |
| `--scheduler` | str | `step` | LR scheduler: `step/cosine/none` |

Implementation notes:
- `label_smoothing_loss()` is equivalent to `F.nll_loss` when `smoothing=0`.
- `ModelNetH5Dataset` adds `scale_augment`, enabled only for training augmentation.
- Full test evaluation runs each epoch, saving train/test metrics and auto-updating `best_model.pth`.
- With `--meshlab_dir`, each epoch exports `.ply` samples with prediction/ground-truth annotations.

### How to Run

```bash
# 1. Prepare environment (run from repo root)
bash colab_final/colab_setup.sh

# 2. Run advanced experiment (run from repo root)
bash colab_final/train_advanced.sh
```

Equivalent full command:

```bash
python colab_final/train_classification_h5.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch 20 \
  --dataset_type modelnet40 \
  --feature_transform \
  --label_smoothing 0.05 \
  --scale_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --min_lr 0.00001 \
  --outf cls_advanced \
  --log_csv cls_advanced/metrics.csv \
  --meshlab_dir cls_advanced/meshlab_ply \
  --meshlab_samples_per_epoch 6
```

### Expected Output Files

| File | Description |
|------|------|
| `cls_advanced/cls_model_<epoch>.pth` | Per-epoch checkpoints |
| `cls_advanced/best_model.pth` | Best checkpoint auto-updated each epoch |
| `cls_advanced/metrics.csv` | Per-epoch `epoch,train_loss,train_acc,test_loss,test_acc,lr` |
| `cls_advanced/loss.txt` | Per-epoch `epoch,train_loss,test_loss` |
| `cls_advanced/accuracy.txt` | Per-epoch `epoch,train_acc,test_acc` |
| `cls_advanced/meshlab_ply/*.ply` | Exported MeshLab point cloud samples |

---

## Notes

- This directory is the consolidated Colab runtime entry for final submission.
- `colab_setup.sh` supports multi-mirror fallback and auto-builds `modelnet10_ply_hdf5_2048` from ModelNet40.
- Save outputs such as `metrics.csv` and model checkpoints from Colab promptly.
