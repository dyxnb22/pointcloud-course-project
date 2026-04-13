# MeshLab 可视化素材

存放使用 MeshLab 对 ModelNet40 点云/网格文件进行可视化的截图。

## 操作步骤

1. 在个人电脑上安装开源 3D 软件 [MeshLab](https://www.meshlab.net/)
2. 从 ModelNet40 数据集中挑选典型 `.off` 文件（如 `chair`、`airplane`）
3. 使用 MeshLab 打开文件，调整视角并截图
4. 将截图保存至本目录，命名示例：
   - `chair_correct.png` — 正确分类样本
   - `chair_misclassified.png` — 误分类样本（原始 PointNet）

## 展示逻辑

对比展示：
- 原始 PointNet 将某类目标误分类的截图及原因（局部细节提取不足）
- 加入 Attention / 使用 DGCNN 后正确分类的截图及改进原因
