# SE-Block 通道注意力实验（Advanced Requirements 2.2 — 架构修改）

在数据增强基础上，于 PointNet 全局特征提取（Global Max Pooling）之后插入轻量级
**SE-Block（Squeeze-and-Excitation Block，通道注意力机制）**，进一步提升分类精度。

## 修改动机

原始 PointNet 对全局 1024 维特征向量的每一维度一视同仁，不区分哪些维度对当前形状
类别更具判别性。SE-Block 通过可学习的通道权重对特征进行重新标定：
1. **Squeeze**：对 1024 维向量经一层 FC（1024→64）得到瓶颈表示；
2. **Excitation**：再经 FC（64→1024）+ Sigmoid 生成每个通道的注意力权重；
3. **Scale**：逐元素乘回原始特征向量。

该机制仅增加约 130K 参数（在 PointNet 约 3.5M 参数基础上），计算开销可忽略。

参考文献：Hu, J. et al. "Squeeze-and-Excitation Networks." CVPR 2018.

## 实现位置

`colab_final/train_advanced.py` 中的 `SEBlock1D` 类及 `PointNetFeatWithSE` 网络。

## 关键代码

```python
class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.Linear(channels, mid)
        self.excite  = nn.Linear(mid, channels)

    def forward(self, x):          # x: (B, 1024)
        scale = F.relu(self.squeeze(x))
        scale = torch.sigmoid(self.excite(scale))
        return x * scale            # 通道注意力重标定
```

SE-Block 在 `PointNetFeatWithSE.forward()` 的 GlobalMaxPool 之后、分类 MLP 之前调用。

## 运行命令

```bash
bash colab_final/train_advanced.sh
# 或使用镜像脚本
bash scripts/train_advanced.sh
```

该脚本同时启用 `--enhanced_aug`（全 SO(3) 旋转 + 裁剪抖动），与数据增强实验协同。

## 记录指标

- 最终 Accuracy：（运行后填写 `cls_advanced/final_accuracy.txt`）
- 相比 Baseline 变化：（预期 +1~2 pp on ModelNet40）
- 相比仅数据增强变化：（预期 +0.5~1 pp）
