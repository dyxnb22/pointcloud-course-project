#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终版：对比 baseline 与 advanced 的训练曲线（loss / acc）

默认读取：
  - cls/metrics.csv
  - cls_advanced/metrics.csv

兼容列名：
  - epoch
  - train_loss
  - train_acc
  - eval acc 列支持: test_acc 或 val_acc
  - eval loss 列支持: test_loss 或 val_loss（可选）

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
    col_eval_acc = _find_col(df, ["test_acc", "val_acc"])
    col_eval_loss = _find_col(df, ["test_loss", "val_loss"])  # optional

    missing = []
    if col_epoch is None: missing.append("epoch")
    if col_train_loss is None: missing.append("train_loss")
    if col_train_acc is None: missing.append("train_acc")
    if col_eval_acc is None: missing.append("test_acc/val_acc")
    if missing:
        raise ValueError(f"[{tag}] 缺少必要列: {', '.join(missing)}")

    out = pd.DataFrame({
        "epoch": pd.to_numeric(df[col_epoch], errors="coerce"),
        "train_loss": pd.to_numeric(df[col_train_loss], errors="coerce"),
        "train_acc": pd.to_numeric(df[col_train_acc], errors="coerce"),
        "eval_acc": pd.to_numeric(df[col_eval_acc], errors="coerce"),
    })

    if col_eval_loss is not None:
        out["eval_loss"] = pd.to_numeric(df[col_eval_loss], errors="coerce")

    out = out.dropna(subset=["epoch", "train_loss", "train_acc", "eval_acc"]).copy()
    out = out.sort_values("epoch")
    if out.empty:
        raise ValueError(f"[{tag}] 清洗后无有效数据，请检查 CSV 数值格式: {csv_path}")

    return out


def summarize(df: pd.DataFrame, name: str):
    idx = df["eval_acc"].idxmax()
    best_acc = float(df.loc[idx, "eval_acc"])
    best_epoch = int(df.loc[idx, "epoch"])
    final_train_loss = float(df["train_loss"].iloc[-1])
    final_train_acc = float(df["train_acc"].iloc[-1])
    final_eval_acc = float(df["eval_acc"].iloc[-1])

    print(f"{name}:")
    print(f"  best eval_acc : {best_acc:.4f} @ epoch {best_epoch}")
    print(f"  final train_loss: {final_train_loss:.4f}")
    print(f"  final train_acc : {final_train_acc:.4f}")
    print(f"  final eval_acc  : {final_eval_acc:.4f}")


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
    print(f"\nΔ best eval_acc (Advanced - Baseline): {a['eval_acc'].max() - b['eval_acc'].max():+.4f}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax = axes[0]
    ax.plot(b["epoch"], b["train_loss"], label="Baseline train_loss", linewidth=2)
    ax.plot(a["epoch"], a["train_loss"], label="Advanced train_loss", linewidth=2)
    if "eval_loss" in b.columns:
        ax.plot(b["epoch"], b["eval_loss"], "--", label="Baseline eval_loss", linewidth=1.8)
    if "eval_loss" in a.columns:
        ax.plot(a["epoch"], a["eval_loss"], "--", label="Advanced eval_loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(b["epoch"], b["train_acc"], label="Baseline train_acc", linewidth=2)
    ax.plot(b["epoch"], b["eval_acc"], "--", label="Baseline eval_acc", linewidth=1.8)
    ax.plot(a["epoch"], a["train_acc"], label="Advanced train_acc", linewidth=2)
    ax.plot(a["epoch"], a["eval_acc"], "--", label="Advanced eval_acc", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
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