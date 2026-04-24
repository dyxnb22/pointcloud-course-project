# Results

Stores training logs and final outputs for each experiment.

## Directory Convention

```
results/
├── metrics_template.csv     # Result comparison table template
├── baseline/
│   ├── loss.txt
│   ├── accuracy.txt
│   └── best_model.pth
├── augmentation/
│   ├── loss.txt
│   └── accuracy.txt
├── attention/
│   ├── loss.txt
│   └── accuracy.txt
└── dgcnn_sota/
    ├── loss.txt
    └── accuracy.txt
```

## Notes

`.pth` model weights can be large. For versioning, use Git LFS or store them separately in cloud storage (for example, Google Drive).
