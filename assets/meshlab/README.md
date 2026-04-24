# MeshLab Visualization Assets

Store screenshots created by visualizing ModelNet40 point cloud/mesh files in MeshLab.

## Steps

1. Install the open-source 3D software [MeshLab](https://www.meshlab.net/) on your computer.
2. Select representative `.off` files from ModelNet40 (for example: `chair`, `airplane`).
3. Open files in MeshLab, adjust the camera, and take screenshots.
4. Save screenshots in this directory, for example:
   - `chair_correct.png` — correctly classified sample
   - `chair_misclassified.png` — misclassified sample (original PointNet)

## Presentation Logic

Show side-by-side comparisons:
- Misclassification examples from original PointNet and why they fail (insufficient local detail extraction)
- Correct classifications after adding Attention / using DGCNN and why they improve
