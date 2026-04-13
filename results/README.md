# Results

存放各实验的训练日志与最终结果文件。

## 目录规范

```
results/
├── metrics_template.csv     # 结果对比表模板
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

## 注意

`.pth` 模型权重文件体积较大，如需版本管理请使用 Git LFS 或单独存储于云端（如 Google Drive）。
