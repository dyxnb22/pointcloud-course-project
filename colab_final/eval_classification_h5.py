#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from pointnet.model import PointNetCls

from train_classification_h5 import ModelNetH5Dataset


def _resolve_existing_path(path, script_dir):
    """Resolve a potentially relative path from common execution locations.

    Resolution order:
    1) current working directory
    2) script directory
    3) repository root relative to script directory (../)
    """
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(script_dir, path)),
        os.path.abspath(os.path.join(script_dir, "..", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"路径不存在: {path}。已尝试: {', '.join(candidates)}"
    )


def _is_tensor_state_dict(candidate):
    return isinstance(candidate, dict) and bool(candidate) and all(
        torch.is_tensor(v) for v in candidate.values()
    )


def _strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict(model_path):
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print(
            "==> Warning: legacy checkpoint format detected; retrying with "
            "weights_only=False (load trusted checkpoints only)."
        )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if _is_tensor_state_dict(ckpt):
        return _strip_module_prefix(ckpt)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and _is_tensor_state_dict(ckpt[key]):
                return _strip_module_prefix(ckpt[key])
        for key in sorted(ckpt.keys()):
            value = ckpt[key]
            if _is_tensor_state_dict(value):
                return _strip_module_prefix(value)
    raise ValueError(f"无法从权重文件中解析有效 state_dict: {model_path}")


def _infer_num_classes(state_dict):
    for key, value in state_dict.items():
        if key.endswith("fc3.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    for key, value in state_dict.items():
        if key.endswith("fc3.bias") and hasattr(value, "shape") and len(value.shape) == 1:
            return int(value.shape[0])
    raise ValueError("无法从权重中推断类别数（未找到 fc3.weight 或 fc3.bias）。")


def _infer_feature_transform(state_dict):
    return any("fstn" in k for k in state_dict.keys())


def _maybe_remap_labels(dataset, num_classes):
    labels = np.asarray(dataset.label, dtype=np.int64)
    if labels.size == 0:
        return False
    min_label = int(labels.min())
    max_label = int(labels.max())
    if min_label < 0:
        raise ValueError(f"检测到负标签值: min_label={min_label}")
    if max_label <= (num_classes - 1):
        return False

    unique_labels = sorted(np.unique(labels).tolist())
    if len(unique_labels) > num_classes:
        raise ValueError(
            f"标签数量({len(unique_labels)})超过模型类别数({num_classes})，"
            f"无法自动重映射。标签范围=[{min_label}, {max_label}]"
        )

    mapping = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.array([mapping[int(x)] for x in labels], dtype=np.int64)
    dataset.label = remapped
    if hasattr(dataset, "classes") and isinstance(dataset.classes, list) and dataset.classes:
        dataset.classes = [
            dataset.classes[old_idx] if 0 <= old_idx < len(dataset.classes) else str(old_idx)
            for old_idx in unique_labels
        ]
    return True


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for points, target in dataloader:
            points = points.transpose(2, 1).to(device)
            target = target.to(device)
            pred, _, _ = model(points)
            total_loss += F.nll_loss(pred, target, reduction="sum").item()
            total_correct += pred.data.max(1)[1].eq(target.data).sum().item()
            total_samples += points.size(0)
    if total_samples == 0:
        raise RuntimeError("测试集为空，无法评测。")
    return total_loss / total_samples, total_correct / total_samples, total_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained PointNet on HDF5 ModelNet dataset.")
    parser.add_argument("--dataset", required=True, help="dataset path, e.g. pointnet.pytorch/data/modelnet10_ply_hdf5_2048")
    parser.add_argument("--model", required=True, help="pretrained model path")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
    parser.add_argument("--num_points", type=int, default=2500, help="number of input points")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.dataset = _resolve_existing_path(args.dataset, script_dir)
    args.model = _resolve_existing_path(args.model, script_dir)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.abspath(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    test_dataset = ModelNetH5Dataset(
        root=args.dataset,
        split="test",
        npoints=args.num_points,
        data_augmentation=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=int(args.workers),
    )

    state_dict = _load_state_dict(args.model)
    num_classes = _infer_num_classes(state_dict)
    feature_transform = _infer_feature_transform(state_dict)
    remapped = _maybe_remap_labels(test_dataset, num_classes)
    if remapped:
        print("==> 检测到数据集标签非连续，已自动重映射到 [0, num_classes-1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device)
    classifier.load_state_dict(state_dict, strict=True)

    test_loss, test_acc, test_samples = evaluate(classifier, test_dataloader, device)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])
        writer.writerow([0, f"{test_loss:.4f}", f"{test_acc:.4f}", f"{test_loss:.4f}", f"{test_acc:.4f}", "0.00000000"])

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "num_classes_from_checkpoint": num_classes,
        "feature_transform_from_checkpoint": feature_transform,
        "test_samples": test_samples,
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "accuracy.txt"), "w", encoding="utf-8") as f:
        f.write("epoch,train_acc,test_acc\n")
        f.write(f"0,{test_acc:.6f},{test_acc:.6f}\n")
    with open(os.path.join(args.out_dir, "loss.txt"), "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,test_loss\n")
        f.write(f"0,{test_loss:.6f},{test_loss:.6f}\n")

    print("==> 评测完成")
    print(f"    test_loss: {test_loss:.4f}")
    print(f"    test_acc : {test_acc:.4f}")
    print(f"    输出目录 : {args.out_dir}")
    print(f"    指标文件 : {metrics_path}")


if __name__ == "__main__":
    main()
