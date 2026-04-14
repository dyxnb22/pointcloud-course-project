from __future__ import print_function

import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from pointnet.model import PointNetCls, feature_transform_regularizer
from tqdm import tqdm


class ModelNetH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root, npoints=2500, split="trainval", data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation

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
        elif points.shape[0] >= self.npoints:
            choice = np.arange(self.npoints)
        else:
            repeats = int(np.ceil(self.npoints / points.shape[0]))
            choice = np.tile(np.arange(points.shape[0]), repeats)[: self.npoints]
        points = points[choice, :]
        points = self._normalize(points)

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)
            points += np.random.normal(0, 0.02, size=points.shape)

        return torch.from_numpy(points.astype(np.float32)), torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.label)


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

    opt = parser.parse_args()
    print(opt)

    if opt.dataset_type != "modelnet40":
        raise ValueError("This wrapper supports only dataset_type=modelnet40 with HDF5 ModelNet data.")

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ModelNetH5Dataset(root=opt.dataset, npoints=opt.num_points, split="trainval")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform).to(device)
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
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).sum().item()
            print(
                "[%d: %d/%d] train loss: %f accuracy: %f"
                % (epoch, i, num_batch, loss.item(), correct / float(opt.batchSize))
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
                        correct_t / float(opt.batchSize),
                    )
                )

        torch.save(classifier.state_dict(), "%s/cls_model_%d.pth" % (opt.outf, epoch))
        scheduler.step()

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
