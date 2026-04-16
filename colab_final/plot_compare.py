#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终版：对比 baseline 与 advanced 的训练曲线（loss / acc / lr）

默认读取：
  - cls/metrics.csv
  - cls_advanced/metrics.csv

兼容列名：
  - epoch
  - train_loss
  - train_acc
  - test_acc 列支持: test_acc 或 val_acc
  - test_loss 列支持: test_loss 或 val_loss
  - lr 列支持: lr 或 learning_rate

用法：
  python plot_compare.py
  python plot_compare.py --baseline cls/metrics.csv --advanced cls_advanced/metrics.csv --out curve_compare.png
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def _find_col(df, candidates):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def load_metrics(csv_path: str, tag: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[{tag}] 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"[{tag}] CSV 为空: {csv_path}")

    col_epoch = _find_col(df, ["epoch"])
    col_train_loss = _find_col(df, ["train_loss"])
    col_train_acc = _find_col(df, ["train_acc"])
    col_test_acc = _find_col(df, ["test_acc", "val_acc"])
    col_test_loss = _find_col(df, ["test_loss", "val_loss"])
    col_lr = _find_col(df, ["lr", "learning_rate"])

    missing = []
    if col_epoch is None: missing.append("epoch")
    if col_train_loss is None: missing.append("train_loss")
    if col_train_acc is None: missing.append("train_acc")
    if col_test_acc is None: missing.append("test_acc/val_acc")
    if col_test_loss is None: missing.append("test_loss/val_loss")
    if col_lr is None: missing.append("lr/learning_rate")
    if missing:
        raise ValueError(f"[{tag}] 缺少必要列: {', '.join(missing)}")

    out = pd.DataFrame({
        "epoch": pd.to_numeric(df[col_epoch], errors="coerce"),
        "train_loss": pd.to_numeric(df[col_train_loss], errors="coerce"),
        "train_acc": pd.to_numeric(df[col_train_acc], errors="coerce"),
        "test_loss": pd.to_numeric(df[col_test_loss], errors="coerce"),
        "test_acc": pd.to_numeric(df[col_test_acc], errors="coerce"),
        "lr": pd.to_numeric(df[col_lr], errors="coerce"),
    })

    out = out.dropna(subset=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"]).copy()
    out = out.sort_values("epoch")
    if out.empty:
        raise ValueError(f"[{tag}] 清洗后无有效数据，请检查 CSV 数值格式: {csv_path}")

    return out


def summarize(df: pd.DataFrame, name: str):
    idx = df["test_acc"].idxmax()
    best_acc = float(df.loc[idx, "test_acc"])
    best_epoch = int(df.loc[idx, "epoch"])
    final_train_loss = float(df["train_loss"].iloc[-1])
    final_train_acc = float(df["train_acc"].iloc[-1])
    final_test_loss = float(df["test_loss"].iloc[-1])
    final_test_acc = float(df["test_acc"].iloc[-1])
    final_lr = float(df["lr"].iloc[-1])

    print(f"{name}:")
    print(f"  best test_acc : {best_acc:.4f} @ epoch {best_epoch}")
    print(f"  final train_loss: {final_train_loss:.4f}")
    print(f"  final test_loss : {final_test_loss:.4f}")
    print(f"  final train_acc : {final_train_acc:.4f}")
    print(f"  final test_acc  : {final_test_acc:.4f}")
    print(f"  final lr        : {final_lr:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="cls/metrics.csv", help="baseline metrics.csv 路径")
    parser.add_argument("--advanced", default="cls_advanced/metrics.csv", help="advanced metrics.csv 路径")
    parser.add_argument("--out", default="curve_compare.png", help="输出图片路径")
    parser.add_argument("--title", default="Training Curves: Baseline vs Advanced", help="总标题")
    parser.add_argument("--dpi", type=int, default=220, help="保存图片 DPI")
    args = parser.parse_args()

    try:
        b = load_metrics(args.baseline, "Baseline")
        a = load_metrics(args.advanced, "Advanced")
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)

    print("=== Metrics Summary ===")
    summarize(b, "Baseline")
    summarize(a, "Advanced")
    print(f"\nΔ best test_acc (Advanced - Baseline): {a['test_acc'].max() - b['test_acc'].max():+.4f}")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(b["epoch"], b["train_loss"], label="Baseline train_loss", linewidth=2)
    ax.plot(a["epoch"], a["train_loss"], label="Advanced train_loss", linewidth=2)
    ax.plot(b["epoch"], b["test_loss"], "--", label="Baseline test_loss", linewidth=1.8)
    ax.plot(a["epoch"], a["test_loss"], "--", label="Advanced test_loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(b["epoch"], b["train_acc"], label="Baseline train_acc", linewidth=2)
    ax.plot(b["epoch"], b["test_acc"], "--", label="Baseline test_acc", linewidth=1.8)
    ax.plot(a["epoch"], a["train_acc"], label="Advanced train_acc", linewidth=2)
    ax.plot(a["epoch"], a["test_acc"], "--", label="Advanced test_acc", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    # LR
    ax = axes[2]
    ax.plot(b["epoch"], b["lr"], label="Baseline lr", linewidth=2)
    ax.plot(a["epoch"], a["lr"], label="Advanced lr", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"\n✅ 已保存对比图: {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
