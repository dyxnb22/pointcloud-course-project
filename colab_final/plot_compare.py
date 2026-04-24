#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final version: compare baseline and advanced training curves (loss / acc / lr)

Default inputs:
  - cls/metrics.csv
  - cls_advanced/metrics.csv

Compatible column names:
  - epoch
  - train_loss
  - train_acc
  - test_acc supports: test_acc or val_acc
  - test_loss supports: test_loss or val_loss
  - lr supports: lr or learning_rate

Usage:
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
        raise FileNotFoundError(f"[{tag}] File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"[{tag}] CSV is empty: {csv_path}")

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
        raise ValueError(f"[{tag}] Missing required columns: {', '.join(missing)}")

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
        raise ValueError(f"[{tag}] No valid rows remain after cleaning, please check CSV numeric format: {csv_path}")

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


def build_output_paths(out_path: str):
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".png"
        root = out_path
    return {
        "loss": f"{root}_loss{ext}",
        "accuracy": f"{root}_accuracy{ext}",
        "lr": f"{root}_lr{ext}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="cls/metrics.csv", help="Path to baseline metrics.csv")
    parser.add_argument("--advanced", default="cls_advanced/metrics.csv", help="Path to advanced metrics.csv")
    parser.add_argument("--out", default="curve_compare.png", help="Output basename (generates *_loss/*_accuracy/*_lr)")
    parser.add_argument("--title", default="Training Curves: Baseline vs Advanced", help="Figure title")
    parser.add_argument("--dpi", type=int, default=220, help="Saved image DPI")
    args = parser.parse_args()

    try:
        b = load_metrics(args.baseline, "Baseline")
        a = load_metrics(args.advanced, "Advanced")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("=== Metrics Summary ===")
    summarize(b, "Baseline")
    summarize(a, "Advanced")
    print(f"\nΔ best test_acc (Advanced - Baseline): {a['test_acc'].max() - b['test_acc'].max():+.4f}")

    outputs = build_output_paths(args.out)

    # Loss plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(b["epoch"], b["train_loss"], label="Baseline train_loss", linewidth=2)
    ax.plot(a["epoch"], a["train_loss"], label="Advanced train_loss", linewidth=2)
    ax.plot(b["epoch"], b["test_loss"], "--", label="Baseline test_loss", linewidth=1.8)
    ax.plot(a["epoch"], a["test_loss"], "--", label="Advanced test_loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs["loss"], dpi=args.dpi)
    plt.close(fig)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(b["epoch"], b["train_acc"], label="Baseline train_acc", linewidth=2)
    ax.plot(b["epoch"], b["test_acc"], "--", label="Baseline test_acc", linewidth=1.8)
    ax.plot(a["epoch"], a["train_acc"], label="Advanced train_acc", linewidth=2)
    ax.plot(a["epoch"], a["test_acc"], "--", label="Advanced test_acc", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs["accuracy"], dpi=args.dpi)
    plt.close(fig)

    # LR plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(b["epoch"], b["lr"], label="Baseline lr", linewidth=2)
    ax.plot(a["epoch"], a["lr"], label="Advanced lr", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs["lr"], dpi=args.dpi)
    plt.close(fig)

    print("\n✅ Saved comparison plots:")
    print(f"  - {outputs['loss']}")
    print(f"  - {outputs['accuracy']}")
    print(f"  - {outputs['lr']}")


if __name__ == "__main__":
    main()
