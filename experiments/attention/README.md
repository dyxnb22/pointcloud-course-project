# Attention 机制实验

在数据增强基础上，于 PointNet 特征提取层后插入轻量级 **SE-Block（通道注意力机制）**，进一步提升分类精度。

## 主要修改

- [`pointnet_attention.py`](pointnet_attention.py) 中实现 `SEBlock`（通道 Squeeze-and-Excitation）
- [`pointnet_attention.py`](pointnet_attention.py) 中实现 `PointNetFeatSE`（在全局特征 1024-d 后插入 SE-Block）
- [`pointnet_attention.py`](pointnet_attention.py) 中实现 `PointNetClsSE`（完整分类网络，含数据增强）
- 继续使用 `experiments/augmentation/dataset_augmented.py` 中的数据增强

## 运行方式

```bash
bash colab_final/train_attention.sh
```

或直接调用：

```bash
python experiments/attention/pointnet_attention.py \
  --dataset pointnet.pytorch/data/modelnet40_ply_hdf5_2048 \
  --nepoch 20 \
  --outdir results/attention
```

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
