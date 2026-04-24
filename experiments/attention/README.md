# Attention Mechanism Experiment

Based on data augmentation, insert a lightweight **SE-Block (channel attention)** after the PointNet feature extractor to further improve classification accuracy.

## Main Changes

- Add SE-Block module after feature extraction in `pointnet.py`
- Continue using data augmentation

## SE-Block Reference

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

## Metrics to Record

- Final accuracy:
- Change vs data augmentation experiment:
