from __future__ import print_function

import argparse
import csv
import os
import random
import re

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from pointnet.model import PointNetCls, feature_transform_regularizer
from tqdm import tqdm


def label_smoothing_loss(log_prob, target, num_classes, smoothing=0.0):
    if smoothing == 0.0 or num_classes <= 1:
        return F.nll_loss(log_prob, target)
    confidence = 1.0 - smoothing
    smooth_val = smoothing / max(1, num_classes - 1)
    with torch.no_grad():
        smooth_dist = torch.full_like(log_prob, smooth_val)
        smooth_dist.scatter_(1, target.unsqueeze(1), confidence)
    return -(smooth_dist * log_prob).sum(dim=1).mean()


def _safe_name(text):
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(text))[:80]


def write_ascii_ply(path, points_xyz, pred_name=None, gt_name=None):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid points shape for PLY export: {pts.shape}")
    pts = pts[:, :3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if pred_name is not None:
            f.write(f"comment pred_label {pred_name}\n")
        if gt_name is not None:
            f.write(f"comment gt_label {gt_name}\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


class ModelNetH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root, npoints=2500, split="trainval", data_augmentation=True, scale_augment=False):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation
        self.scale_augment = scale_augment

        file_list = self._resolve_h5_list(split)
        data_chunks = []
        label_chunks = []
        for path in file_list:
            with h5py.File(path, "r") as h5f:
                data_chunks.append(h5f["data"][:])
                label_chunks.append(h5f["label"][:].reshape(-1))

        if not data_chunks:
            raise RuntimeError(f"No H5 data loaded from: {root} (split={split})")

        self.data = np.concatenate(data_chunks, axis=0).astype(np.float32)
        self.label = np.concatenate(label_chunks, axis=0).astype(np.int64)
        self.classes = self._load_classes()

    def _resolve_h5_list(self, split):
        if split == "trainval":
            list_candidates = ("train_files.txt", "train_hdf5_file_list.txt")
        elif split == "test":
            list_candidates = ("test_files.txt", "test_hdf5_file_list.txt")
        else:
            raise ValueError(f"Unsupported split: {split}")

        entries = []
        for list_file in list_candidates:
            list_path = os.path.join(self.root, list_file)
            if not os.path.isfile(list_path):
                continue
            with open(list_path, "r", encoding="utf-8") as f:
                for raw in f:
                    item = raw.strip()
                    if item:
                        entries.append(item)
            if entries:
                break

        if not entries:
            prefix = "ply_data_train" if split == "trainval" else "ply_data_test"
            entries = sorted(
                [name for name in os.listdir(self.root) if name.startswith(prefix) and name.endswith(".h5")]
            )

        paths = []
        for entry in entries:
            candidates = []
            if os.path.isabs(entry):
                candidates.append(entry)
            candidates.append(os.path.join(self.root, entry))
            candidates.append(os.path.join(self.root, os.path.basename(entry)))

            resolved = next((p for p in candidates if os.path.isfile(p)), None)
            if resolved is not None:
                paths.append(resolved)

        if not paths:
            raise FileNotFoundError(
                f"Cannot resolve any H5 files for split={split} under {self.root}. "
                f"Expected list files: {list_candidates}."
            )
        return paths

    def _load_classes(self):
        shape_names = os.path.join(self.root, "shape_names.txt")
        if os.path.isfile(shape_names):
            with open(shape_names, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        max_label = int(self.label.max()) if self.label.size > 0 else -1
        return [str(i) for i in range(max_label + 1)]

    @staticmethod
    def _normalize(points):
        points = points - np.expand_dims(np.mean(points, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
        if dist > 0:
            points = points / dist
        return points

    def __getitem__(self, index):
        points = self.data[index]
        target = self.label[index]

        if self.data_augmentation:
            choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        else:
            rng = np.random.default_rng(seed=index)
            replace = points.shape[0] < self.npoints
            choice = rng.choice(points.shape[0], self.npoints, replace=replace)
        points = points[choice, :]
        points = self._normalize(points)

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)
            points += np.random.normal(0, 0.02, size=points.shape)
            if self.scale_augment:
                scale = np.random.uniform(0.8, 1.25)
                points *= scale

        return torch.from_numpy(points.astype(np.float32)), torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.label)


def evaluate_and_export(
    classifier,
    dataloader,
    device,
    num_classes,
    meshlab_dir="",
    epoch_idx=0,
    meshlab_samples_per_epoch=0,
    class_names=None,
):
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    remaining_exports = max(0, meshlab_samples_per_epoch)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            pts_cpu, tgt = data
            pts = pts_cpu.transpose(2, 1).to(device)
            tgt = tgt.to(device)

            pred, _, _ = classifier(pts)
            loss_sum = F.nll_loss(pred, tgt, reduction="sum")
            total_loss += loss_sum.item()

            pred_choice = pred.data.max(1)[1]
            total_correct += pred_choice.eq(tgt.data).sum().item()
            batch_size = pts.size(0)
            total_samples += batch_size

            if meshlab_dir and remaining_exports > 0:
                pred_np = pred_choice.detach().cpu().numpy()
                tgt_np = tgt.detach().cpu().numpy()
                pts_np = pts_cpu.detach().cpu().numpy()
                take = min(remaining_exports, batch_size)
                for j in range(take):
                    pred_id = int(pred_np[j])
                    gt_id = int(tgt_np[j])
                    pred_name = class_names[pred_id] if class_names and pred_id < len(class_names) else str(pred_id)
                    gt_name = class_names[gt_id] if class_names and gt_id < len(class_names) else str(gt_id)
                    file_name = (
                        f"epoch_{epoch_idx:03d}_batch_{batch_idx:04d}_sample_{j:02d}_"
                        f"pred_{_safe_name(pred_name)}_gt_{_safe_name(gt_name)}.ply"
                    )
                    write_ascii_ply(
                        os.path.join(meshlab_dir, file_name),
                        pts_np[j],
                        pred_name=pred_name,
                        gt_name=gt_name,
                    )
                remaining_exports -= take

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
    parser.add_argument("--num_points", type=int, default=2500, help="number of input points")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--nepoch", type=int, default=250, help="number of epochs to train for")
    parser.add_argument("--outf", type=str, default="cls", help="output folder")
    parser.add_argument("--model", type=str, default="", help="model path")
    parser.add_argument("--dataset", type=str, required=True, help="dataset path")
    parser.add_argument("--dataset_type", type=str, default="modelnet40", help="dataset type")
    parser.add_argument("--feature_transform", action="store_true", help="use feature transform")

    parser.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="label smoothing factor for cross-entropy loss (0=disabled, e.g. 0.05)",
    )
    parser.add_argument(
        "--scale_augment", action="store_true", default=False,
        help="apply random scale augmentation [0.8, 1.25] during training",
    )
    parser.add_argument(
        "--log_csv", type=str, default="",
        help="if set, write per-epoch metrics to this CSV file",
    )
    parser.add_argument(
        "--meshlab_dir", type=str, default="",
        help="if set, export sample test point clouds (.ply) for MeshLab per epoch",
    )
    parser.add_argument(
        "--meshlab_samples_per_epoch", type=int, default=0,
        help="number of test samples to export as .ply each epoch (requires --meshlab_dir)",
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for Adam optimizer")
    parser.add_argument(
        "--scheduler", type=str, default="step", choices=["step", "cosine", "none"],
        help="lr scheduler type",
    )
    parser.add_argument("--lr_step_size", type=int, default=20, help="step scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="step scheduler gamma")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum lr for cosine scheduler")

    opt = parser.parse_args()
    print(opt)

    if opt.dataset_type != "modelnet40":
        raise ValueError("This wrapper supports only dataset_type=modelnet40 with HDF5 ModelNet data.")

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ModelNetH5Dataset(
        root=opt.dataset, npoints=opt.num_points, split="trainval",
        scale_augment=opt.scale_augment,
    )
    test_dataset = ModelNetH5Dataset(
        root=opt.dataset,
        split="test",
        npoints=opt.num_points,
        data_augmentation=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)
    )
    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers)
    )

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print("classes", num_classes)

    os.makedirs(opt.outf, exist_ok=True)
    if opt.meshlab_dir:
        os.makedirs(opt.meshlab_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform).to(device)
    if opt.model:
        classifier.load_state_dict(torch.load(opt.model, map_location=device))

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=opt.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=opt.weight_decay,
    )

    if opt.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
    elif opt.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, opt.nepoch),
            eta_min=opt.min_lr,
        )
    else:
        scheduler = None

    num_batch = max(1, len(dataset) // opt.batchSize)
    blue = lambda x: "\033[94m" + x + "\033[0m"

    csv_fh = None
    csv_writer = None
    loss_fh = None
    acc_fh = None
    if opt.log_csv:
        os.makedirs(os.path.dirname(opt.log_csv) or ".", exist_ok=True)
        csv_fh = open(opt.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])

        log_dir = os.path.dirname(opt.log_csv) or "."
        loss_fh = open(os.path.join(log_dir, "loss.txt"), "w", encoding="utf-8")
        acc_fh = open(os.path.join(log_dir, "accuracy.txt"), "w", encoding="utf-8")
        loss_fh.write("epoch,train_loss,test_loss\n")
        acc_fh.write("epoch,train_acc,test_acc\n")

    best_test_acc = -1.0

    for epoch in range(opt.nepoch):
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        test_iter = iter(testdataloader)

        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = label_smoothing_loss(pred, target, num_classes, smoothing=opt.label_smoothing)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum().item()
            epoch_loss_sum += loss.item()
            epoch_correct += correct
            epoch_total += points.size(0)
            print(
                "[%d: %d/%d] train loss: %f accuracy: %f"
                % (epoch, i, num_batch, loss.item(), correct / float(points.size(0)))
            )

            if i % 10 == 0:
                try:
                    points_t, target_t = next(test_iter)
                except StopIteration:
                    test_iter = iter(testdataloader)
                    points_t, target_t = next(test_iter)
                points_t = points_t.transpose(2, 1).to(device)
                target_t = target_t.to(device)
                classifier = classifier.eval()
                pred_t, _, _ = classifier(points_t)
                loss_t = F.nll_loss(pred_t, target_t)
                pred_choice_t = pred_t.data.max(1)[1]
                correct_t = pred_choice_t.eq(target_t.data).sum().item()
                print(
                    "[%d: %d/%d] %s loss: %f accuracy: %f"
                    % (
                        epoch,
                        i,
                        num_batch,
                        blue("test"),
                        loss_t.item(),
                        correct_t / float(points_t.size(0)),
                    )
                )

        torch.save(classifier.state_dict(), "%s/cls_model_%d.pth" % (opt.outf, epoch))

        train_loss_avg = epoch_loss_sum / num_batch
        train_acc = epoch_correct / max(1, epoch_total)
        test_loss_avg, test_acc = evaluate_and_export(
            classifier,
            testdataloader,
            device,
            num_classes,
            meshlab_dir=opt.meshlab_dir,
            epoch_idx=epoch,
            meshlab_samples_per_epoch=opt.meshlab_samples_per_epoch,
            class_names=test_dataset.classes,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if csv_writer is not None:
            csv_writer.writerow(
                [
                    epoch,
                    f"{train_loss_avg:.4f}",
                    f"{train_acc:.4f}",
                    f"{test_loss_avg:.4f}",
                    f"{test_acc:.4f}",
                    f"{current_lr:.8f}",
                ]
            )
            csv_fh.flush()
            loss_fh.write(f"{epoch},{train_loss_avg:.6f},{test_loss_avg:.6f}\n")
            acc_fh.write(f"{epoch},{train_acc:.6f},{test_acc:.6f}\n")
            loss_fh.flush()
            acc_fh.flush()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(classifier.state_dict(), os.path.join(opt.outf, "best_model.pth"))

        print(
            "Epoch %d summary — train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f lr: %.8f"
            % (epoch, train_loss_avg, train_acc, test_loss_avg, test_acc, current_lr)
        )

        if scheduler is not None:
            scheduler.step()

    if csv_fh is not None:
        csv_fh.close()
    if loss_fh is not None:
        loss_fh.close()
    if acc_fh is not None:
        acc_fh.close()

    total_correct = 0
    total_testset = 0
    classifier = classifier.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            points = points.transpose(2, 1).to(device)
            target = target.to(device)
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            total_correct += pred_choice.eq(target.data).sum().item()
            total_testset += points.size(0)

    print("final accuracy {}".format(total_correct / float(max(1, total_testset))))


if __name__ == "__main__":
    main()
