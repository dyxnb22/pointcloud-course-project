"""
Advanced Extension (2.2): PointNet + SE-Attention + Enhanced Data Augmentation
===============================================================================

Rationale:

  Modification 1 — Architecture (SE-Block channel attention)
  -----------------------------------------------------------
  Original PointNet obtains a 1024-dim global feature vector via max-pooling
  over all point-wise features, then feeds it directly into a 3-layer MLP for
  classification.  Every feature dimension is treated equally, regardless of
  how discriminative it actually is for the current shape category.

  We insert a lightweight Squeeze-and-Excitation (SE) Block [Hu et al. 2018]
  immediately after the global max-pooling.  The SE-Block
    1. "squeezes" the 1024-dim vector through a bottleneck FC (1024 → 64),
    2. learns a per-channel weight (64 → 1024) via Sigmoid, and
    3. re-scales the global feature element-wise.

  This recalibration lets the network focus on the most informative feature
  channels for each input shape, improving discrimination at negligible extra
  cost (~130 k parameters on top of ~3.5 M baseline).

  Reference: Hu, J. et al. "Squeeze-and-Excitation Networks." CVPR 2018.

  Modification 2 — Data (enhanced augmentation)
  ----------------------------------------------
  The baseline `train_classification_h5.py` applies only a random rotation
  around the Y-axis and small Gaussian jitter (σ=0.02).  This biases the
  model toward upright orientations and provides limited regularisation.

  Two improvements are applied when `--enhanced_aug` is set:
    a. Random SO(3) rotation — a uniformly sampled 3-D rotation matrix is
       generated via QR decomposition of a random Gaussian matrix.  This
       forces the model to learn truly rotation-invariant features.
    b. Scaled jitter with ±0.05 clipping — jitter noise is drawn from
       N(0, 0.04) and clamped to [−0.05, 0.05].  This is more aggressive
       than the baseline, improving generalisation to noisy point clouds
       without corrupting local geometry too severely.

Final Results:
  Run the script and record the printed "final accuracy" line.
  See results/metrics_template.csv for the comparison table.
  Expected gain over baseline (no feature_transform): ~1–2 pp on ModelNet40.
"""

from __future__ import print_function

import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm


# ===========================================================================
# Advanced Modification 1 – SE-Block (channel-wise attention)
# ===========================================================================

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1-D feature vectors (B, C).

    Args:
        channels (int): Number of input/output channels.
        reduction (int): Bottleneck reduction ratio (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock1D, self).__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.Linear(channels, mid)
        self.excite = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        scale = F.relu(self.squeeze(x))
        scale = torch.sigmoid(self.excite(scale))
        return x * scale


# ===========================================================================
# Advanced Modification 1 – PointNet + SE classification network
# ===========================================================================

class STN3d(nn.Module):
    """Mini T-Net for 3×3 input transform (same as original PointNet)."""

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = (
            torch.eye(3, device=x.device)
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        x = x + iden
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    """Mini T-Net for k×k feature transform (same as original PointNet)."""

    def __init__(self, k: int = 64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = (
            torch.eye(self.k, device=x.device)
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        x = x + iden
        return x.view(-1, self.k, self.k)


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularisation loss for the feature transform matrix."""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


class PointNetFeatWithSE(nn.Module):
    """PointNet feature extraction with optional T-Nets and SE-Block.

    Architecture (with SE-Block, no feature transform):
        Conv(3→64) → BN → ReLU
        Conv(64→128) → BN → ReLU
        Conv(128→1024) → BN → ReLU
        GlobalMaxPool  →  (B, 1024)
        SEBlock1D(1024, reduction=16)   ← Advanced Modification 1
    """

    def __init__(self, feature_transform: bool = False):
        super(PointNetFeatWithSE, self).__init__()
        self.feature_transform = feature_transform

        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if feature_transform:
            self.fstn = STNkd(k=64)

        # SE-Block inserted after global max pooling (Advanced Modification 1)
        self.se = SEBlock1D(channels=1024, reduction=16)

    def forward(self, x):
        # x: (B, 3, N)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # (B, 1024)

        # SE-Block: channel-wise attention (Advanced Modification 1)
        x = self.se(x)

        return x, trans, trans_feat


class PointNetClsWithSE(nn.Module):
    """PointNet + SE classification head."""

    def __init__(self, k: int = 40, feature_transform: bool = False):
        super(PointNetClsWithSE, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeatWithSE(feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


# ===========================================================================
# Advanced Modification 2 – Enhanced Data Augmentation
# ===========================================================================

def random_so3_rotation() -> np.ndarray:
    """Uniformly sample a random 3-D rotation matrix from SO(3).

    Uses QR decomposition of a random Gaussian matrix, which yields a
    uniformly distributed rotation matrix (Haar measure on SO(3)).
    This is more thorough than rotating around a single axis because it
    exposes the model to all possible orientations during training.
    """
    H = np.random.randn(3, 3).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))  # make diagonal of R positive (canonical form)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # ensure det(Q) = +1 (proper rotation)
    return Q


def jitter_with_clip(
    points: np.ndarray,
    sigma: float = 0.04,
    clip: float = 0.05,
) -> np.ndarray:
    """Add Gaussian noise to points with hard clipping.

    More aggressive than the baseline (σ=0.02, no clip) to improve
    generalisation to moderately noisy real-scan point clouds.
    """
    noise = np.clip(
        np.random.normal(0, sigma, size=points.shape),
        -clip,
        clip,
    ).astype(np.float32)
    return points + noise


class ModelNetH5Dataset(torch.utils.data.Dataset):
    """ModelNet HDF5 dataset with optional enhanced augmentation.

    Enhanced augmentation (Advanced Modification 2):
      - Random SO(3) rotation (uniform over all 3-D orientations)
      - Jitter N(0, 0.04) clipped to ±0.05
    """

    def __init__(
        self,
        root: str,
        npoints: int = 2500,
        split: str = "trainval",
        data_augmentation: bool = True,
        enhanced_aug: bool = False,
    ):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation
        self.enhanced_aug = enhanced_aug

        file_list = self._resolve_h5_list(split)
        data_chunks, label_chunks = [], []
        for path in file_list:
            with h5py.File(path, "r") as h5f:
                data_chunks.append(h5f["data"][:])
                label_chunks.append(h5f["label"][:].reshape(-1))

        if not data_chunks:
            raise RuntimeError(f"No H5 data loaded from: {root} (split={split})")

        self.data = np.concatenate(data_chunks, axis=0).astype(np.float32)
        self.label = np.concatenate(label_chunks, axis=0).astype(np.int64)
        self.classes = self._load_classes()

    # ------------------------------------------------------------------
    # Internal helpers (same as baseline train_classification_h5.py)
    # ------------------------------------------------------------------

    def _resolve_h5_list(self, split: str):
        candidates = (
            ("train_files.txt", "train_hdf5_file_list.txt")
            if split == "trainval"
            else ("test_files.txt", "test_hdf5_file_list.txt")
        )
        entries = []
        for list_file in candidates:
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
                n for n in os.listdir(self.root)
                if n.startswith(prefix) and n.endswith(".h5")
            )

        paths = []
        for entry in entries:
            for candidate in (
                entry if os.path.isabs(entry) else None,
                os.path.join(self.root, entry),
                os.path.join(self.root, os.path.basename(entry)),
            ):
                if candidate and os.path.isfile(candidate):
                    paths.append(candidate)
                    break

        if not paths:
            raise FileNotFoundError(
                f"Cannot resolve H5 files for split={split} under {self.root}."
            )
        return paths

    def _load_classes(self):
        shape_names = os.path.join(self.root, "shape_names.txt")
        if os.path.isfile(shape_names):
            with open(shape_names, "r", encoding="utf-8") as f:
                return [l.strip() for l in f if l.strip()]
        max_label = int(self.label.max()) if self.label.size > 0 else -1
        return [str(i) for i in range(max_label + 1)]

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        points = points - np.mean(points, axis=0, keepdims=True)
        dist = np.max(np.linalg.norm(points, axis=1))
        if dist > 0:
            points /= dist
        return points

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        points = self.data[index].copy()
        target = self.label[index]

        # Sub-sample
        rng = np.random.default_rng() if self.data_augmentation else np.random.default_rng(seed=index)
        replace = points.shape[0] < self.npoints
        choice = rng.choice(points.shape[0], self.npoints, replace=replace)
        points = points[choice, :]
        points = self._normalize(points)

        if self.data_augmentation:
            if self.enhanced_aug:
                # Advanced Modification 2a: full SO(3) random rotation
                R = random_so3_rotation()
                points = points @ R.T
                # Advanced Modification 2b: scaled jitter with clipping
                points = jitter_with_clip(points, sigma=0.04, clip=0.05)
            else:
                # Baseline augmentation: Y-axis rotation + small jitter
                theta = np.random.uniform(0, np.pi * 2)
                R_y = np.array(
                    [[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]],
                    dtype=np.float32,
                )
                points[:, [0, 2]] = points[:, [0, 2]] @ R_y
                points += np.random.normal(0, 0.02, size=points.shape).astype(np.float32)

        return (
            torch.from_numpy(points.astype(np.float32)),
            torch.tensor(target, dtype=torch.long),
        )


# ===========================================================================
# Training loop
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PointNet + SE-Block + Enhanced Augmentation (Advanced Req. 2.2)"
    )
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nepoch", type=int, default=250)
    parser.add_argument("--outf", type=str, default="cls_advanced")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--feature_transform",
        action="store_true",
        help="use feature transform T-Net (optional)",
    )
    parser.add_argument(
        "--enhanced_aug",
        action="store_true",
        help="use enhanced data augmentation (SO(3) rotation + clipped jitter)",
    )
    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ModelNetH5Dataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split="trainval",
        data_augmentation=True,
        enhanced_aug=opt.enhanced_aug,
    )
    test_dataset = ModelNetH5Dataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split="test",
        data_augmentation=False,
        enhanced_aug=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
    )
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers),
    )

    num_classes = len(dataset.classes)
    print(f"Train: {len(dataset)}  Test: {len(test_dataset)}  Classes: {num_classes}")

    os.makedirs(opt.outf, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classifier = PointNetClsWithSE(k=num_classes, feature_transform=opt.feature_transform).to(device)
    if opt.model:
        classifier.load_state_dict(torch.load(opt.model, map_location=device))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    num_batch = max(1, len(dataset) // opt.batchSize)
    blue = lambda x: "\033[94m" + x + "\033[0m"

    for epoch in range(opt.nepoch):
        test_iter = iter(testdataloader)
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform and trans_feat is not None:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum().item()
            print(
                "[%d: %d/%d] train loss: %.4f  acc: %.4f"
                % (epoch, i, num_batch, loss.item(), correct / float(points.size(0)))
            )

            if i % 10 == 0:
                try:
                    pts_t, tgt_t = next(test_iter)
                except StopIteration:
                    test_iter = iter(testdataloader)
                    pts_t, tgt_t = next(test_iter)
                pts_t = pts_t.transpose(2, 1).to(device)
                tgt_t = tgt_t.to(device)
                classifier.eval()
                with torch.no_grad():
                    pred_t, _, _ = classifier(pts_t)
                loss_t = F.nll_loss(pred_t, tgt_t)
                pred_choice_t = pred_t.data.max(1)[1]
                correct_t = pred_choice_t.eq(tgt_t.data).cpu().sum().item()
                print(
                    "[%d: %d/%d] %s  loss: %.4f  acc: %.4f"
                    % (
                        epoch, i, num_batch,
                        blue("test"), loss_t.item(),
                        correct_t / float(pts_t.size(0)),
                    )
                )

        torch.save(
            classifier.state_dict(),
            os.path.join(opt.outf, "cls_advanced_%d.pth" % epoch),
        )
        scheduler.step()

    # Final evaluation
    total_correct = 0
    total_testset = 0
    classifier.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(testdataloader, 0), desc="Final eval"):
            points, target = data
            points = points.transpose(2, 1).to(device)
            target = target.to(device)
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            total_correct += pred_choice.eq(target.data).cpu().sum().item()
            total_testset += points.size(0)

    final_acc = total_correct / float(max(1, total_testset))
    print(f"final accuracy {final_acc:.4f}")

    # Save final accuracy to file for reproducibility
    acc_file = os.path.join(opt.outf, "final_accuracy.txt")
    with open(acc_file, "w", encoding="utf-8") as f:
        f.write(
            f"model: PointNetClsWithSE\n"
            f"dataset: {opt.dataset}\n"
            f"feature_transform: {opt.feature_transform}\n"
            f"enhanced_aug: {opt.enhanced_aug}\n"
            f"epochs: {opt.nepoch}\n"
            f"final_accuracy: {final_acc:.4f}\n"
        )
    print(f"Accuracy saved to {acc_file}")


if __name__ == "__main__":
    main()
