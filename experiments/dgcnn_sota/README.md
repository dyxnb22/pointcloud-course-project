# DGCNN SOTA 对比实验

使用 DGCNN（Dynamic Graph CNN）在 ModelNet40 上进行训练，获取 SOTA 级别的分类精度，用于与 PointNet 系列实验对比。

## 训练命令

```bash
bash scripts/train_dgcnn.sh
```

## 记录指标

- 最终 Accuracy：
- 相比 Baseline (PointNet) 变化：

## 参考

- 论文：[Dynamic Graph CNN for Learning on Point Clouds (DGCNN)](https://arxiv.org/abs/1801.07829)
- 代码：[WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
