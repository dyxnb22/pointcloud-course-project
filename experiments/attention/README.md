# Attention 机制实验

在数据增强基础上，于 PointNet 特征提取层后插入轻量级 **SE-Block（通道注意力机制）**，进一步提升分类精度。

## 主要修改

- `pointnet.py` 特征提取层后添加 SE-Block 模块（实现见 [`pointnet.py`](pointnet.py)）
- 继续使用数据增强

## SE-Block 参考结构

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale
```

## 记录指标

- 最终 Accuracy：
- 相比数据增强实验变化：

