# Point Cloud Course Project (PointNet vs DGCNN)

This repository is used for a **two-stage course project**:

- **Stage 1 (Google Colab)**: environment setup, baseline training, advanced modifications (data augmentation + SE-Attention), and SOTA comparison (DGCNN)
- **Stage 2 (Report & Presentation)**: collect logs, organize comparison tables, MeshLab visualization, and misclassification analysis

---

## Repository Structure

```
pointcloud-course-project/
├── README.md
├── .gitignore
├── requirements.txt
├── colab_final/        # Consolidated Colab final-run code (submission entry)
├── notebooks/          # Google Colab notebooks
├── scripts/            # One-command scripts
│   ├── colab_setup.sh
│   ├── train_baseline.sh
│   ├── train_cross_dataset.sh
│   ├── train_dgcnn.sh
│   └── package_final.sh
├── experiments/        # Stage-by-stage experiment notes
│   ├── baseline/
│   ├── augmentation/
│   ├── attention/
│   └── dgcnn_sota/
├── results/            # Training logs and result comparisons
├── assets/meshlab/     # MeshLab visualization screenshots
└── report/             # Report and PPT materials
```

---

## Colab Final Submission Entry

If you want all final Colab executable code in one place, use [`colab_final/`](colab_final/).

This directory already centralizes executable key scripts and usage documentation:

- `colab_final/README.md`
- `colab_final/colab_setup.sh`
- `colab_final/train_baseline.sh`
- `colab_final/train_cross_dataset.sh`
- `colab_final/train_advanced_modelnet10.sh`
- `colab_final/train_dgcnn.sh`
- `colab_final/package_final.sh`
- `colab_final/package_modelnet10_compare.sh`

---

## 1. Environment Setup (Google Colab T4 GPU)

1. Create a new Colab notebook.
2. Select **Runtime → Change runtime type → Hardware accelerator: T4 GPU**.
3. Run in a Colab cell:

```bash
# Clone PointNet (baseline model)
!git clone https://github.com/fxia22/pointnet.pytorch.git

# Clone DGCNN (SOTA comparison model)
!git clone https://github.com/WangYueFt/dgcnn.git

# Install PointNet dependencies (PyTorch is preinstalled in Colab)
!pip install -e ./pointnet.pytorch
```

Or run the one-command setup script:

```bash
bash scripts/colab_setup.sh
```

---

## 2. Baseline Training (PointNet, ModelNet40 main track)

### 2.1 Download Dataset

```bash
bash scripts/colab_setup.sh
```

> `colab_setup.sh` includes multi-mirror ModelNet40 download with auto-fallback, and automatically builds the second dataset `modelnet10_ply_hdf5_2048` from ModelNet40 (no extra download needed).

### 2.2 Training Command

```bash
!python scripts/train_classification_h5.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch=20 \
  --dataset_type modelnet40
```

Or use script:

```bash
bash scripts/train_baseline.sh
```

> **Note**: This repository uses `scripts/train_classification_h5.py` as the HDF5-compatible PointNet training entry.

---

## 3. Cross-Dataset and Robustness Testing (Basic requirements 6 & 7)

Typical PointNet weaknesses: **limited local feature extraction and sensitivity to noise**.

Testing method:
1. Add Gaussian jitter in test-set data loading.
2. Use a second classification dataset for cross-dataset testing (this repo provides an auto-built ModelNet10 subset from ModelNet40), and record accuracy changes as evidence for weakness analysis.

Cross-dataset training command (ModelNet10 subset):

```bash
bash scripts/train_cross_dataset.sh
```

Cross-dataset Advanced (ModelNet10 subset, robustness validation):

```bash
bash scripts/train_advanced_modelnet10.sh
```

---

## 4. Advanced Modifications (Ablation, 20%)

### A. Data Augmentation

Add in `dataset.py`:

- `random_rotate_point_cloud`: random rotation
- `jitter_point_cloud`: jitter noise

Retrain and record accuracy. See [`experiments/augmentation/`](experiments/augmentation/).

### B. SE-Block Channel Attention

Insert a lightweight SE-Block after the feature extraction layer in `pointnet.py`, retrain, and record accuracy changes. See [`experiments/attention/`](experiments/attention/).

---

## 5. SOTA Comparison: DGCNN (Basic requirement 8)

```bash
!cd dgcnn/pytorch && python main.py \
  --exp_name=dgcnn_test \
  --model=dgcnn \
  --dataset=modelnet40
```

Or use script:

```bash
bash scripts/train_dgcnn.sh
```

See [`experiments/dgcnn_sota/`](experiments/dgcnn_sota/).

---

## 6. Files You Must Save

| File | Description |
|---|---|
| `loss.txt` | Per-epoch training loss |
| `accuracy.txt` | Per-epoch validation accuracy |
| `best_model.pth` | Best checkpoint |

Download these files from the Colab file panel; they must be included in your course submission archive.

To collect and package submission files in one command:

```bash
bash scripts/package_final.sh
```

To export ModelNet10 Baseline vs Advanced comparison results (new folder + zip):

```bash
bash scripts/package_modelnet10_compare.sh
```

> The script attempts to run `colab_final/plot_compare.py` automatically to generate 3 separate images: `curve_compare_loss.png`, `curve_compare_accuracy.png`, `curve_compare_lr.png` (requires `cls/metrics.csv` and `cls_advanced/metrics.csv`) and packages them together.

Generated in repository root:

- `final/` (collected folder)
- `final_submission.zip` (ready to download and upload)

---

## 7. Experiment Comparison Table

| Experiment | Accuracy | Notes |
|---|---:|---|
| Baseline (original PointNet) | | |
| Baseline + data augmentation | | |
| Baseline + data augmentation + SE-Attention | | |
| SOTA (DGCNN) | | |

Full template: [`results/metrics_template.csv`](results/metrics_template.csv).

---

## 8. MeshLab Visualization (Basic requirement 5)

1. Install [MeshLab](https://www.meshlab.net/) locally.
2. Select representative `.off` files from ModelNet40 (for example, `chair`, `airplane`).
3. Adjust viewpoints and save screenshots to [`assets/meshlab/`](assets/meshlab/).
4. Show side-by-side comparisons:
   - Misclassification cases of original PointNet (insufficient local detail)
   - Why attention / DGCNN improves classification

---

## References

- [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)
- [MeshLab](https://www.meshlab.net/)
