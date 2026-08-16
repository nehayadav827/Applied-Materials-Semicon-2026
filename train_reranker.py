"""
Trains the Siamese reranker. Splits underlying pattern WORLDS (not raw
samples) three ways -- train / val / test -- so:

  - train: what the model actually learns from.
  - val:   watched every epoch for early-stopping / sanity, so it is a
           TUNING set, not a clean measure of generalization.
  - test:  never touched during training or model selection. Its
           sample_ids are written to test_split.csv so evaluate.py can
           automatically score ONLY this held-out set by default -- that's
           what should go in the PPT / README numbers.

Splitting is done by world_id (the underlying rendered pattern a sample was
cropped from), not by sample_id, so no near-duplicate crop of the same
world ends up split across train and test -- that would leak information
and inflate the reported accuracy.

Usage:
    python train_reranker.py --dataset ./synthetic_sem_dataset --epochs 30
"""

import argparse, os, time, random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(4)
PATCH_SIZE = 96
REF_TO_SEARCH_SCALE = 10.0   # matches the generator's fixed 10:1 scale


class TinyEmbedNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=1)


class PairDataset(Dataset):
    def __init__(self, dataset_dir, df, hard_neg_df, augment=True):
        self.dataset_dir = dataset_dir
        self.df = df.reset_index(drop=True)
        self.hard_neg = hard_neg_df.set_index("sample_id")
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_gray(self, rel_path):
        return cv2.imread(os.path.join(self.dataset_dir, rel_path), cv2.IMREAD_GRAYSCALE)

    def _augment(self, patch):
        if not self.augment:
            return patch
        if random.random() < 0.5:
            patch = cv2.flip(patch, 1)
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, patch.shape).astype(np.float32)
            patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return patch

    def _to_tensor(self, patch):
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
        return torch.from_numpy(patch).float().unsqueeze(0) / 255.0

    def _to_ref_tensor(self, patch):
        """The reference is a 1000x1000 100x-magnification crop; a candidate
        search-image patch covering the same physical area is only ~100x100
        px (since search is ~10x zoomed out). Resizing the reference
        straight to PATCH_SIZE is a ~10.4x downsample that erases the fine
        periodic structure (pitch is only a few px at search resolution);
        resizing a candidate patch to PATCH_SIZE is only ~1.04x. That
        mismatch means the network was comparing a near-featureless blur of
        the reference against sharp candidate crops -- fix: pre-downsample
        the reference by the known nominal scale (REF_TO_SEARCH_SCALE) so
        both sides go through a comparable final resize."""
        h, w = patch.shape[:2]
        pre_w = max(8, int(round(w / REF_TO_SEARCH_SCALE)))
        pre_h = max(8, int(round(h / REF_TO_SEARCH_SCALE)))
        patch = cv2.resize(patch, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
        return torch.from_numpy(patch).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        search = self._load_gray(row["search_file"])
        reference = self._load_gray(row["reference_file"])

        gt_xmin, gt_ymin = row["GT_X_min"], row["GT_Y_min"]
        gt_xmax, gt_ymax = row["GT_X_max"], row["GT_Y_max"]
        pos_w = max(int(gt_xmax - gt_xmin), 8)
        pos_h = max(int(gt_ymax - gt_ymin), 8)

        px0 = max(0, int(gt_xmin))
        py0 = max(0, int(gt_ymin))
        positive = search[py0:py0 + pos_h, px0:px0 + pos_w]

        h, w = search.shape[:2]
        neg_row = self.hard_neg.loc[row["sample_id"]]
        nx = int(np.clip(neg_row["neg_x"] - pos_w / 2, 0, max(0, w - pos_w)))
        ny = int(np.clip(neg_row["neg_y"] - pos_h / 2, 0, max(0, h - pos_h)))
        negative = search[ny:ny + pos_h, nx:nx + pos_w]

        if positive.size == 0:
            positive = search[:pos_h, :pos_w]
        if negative.size == 0:
            negative = search[-pos_h:, -pos_w:]

        return (self._to_ref_tensor(self._augment(reference)),
                self._to_tensor(self._augment(positive)),
                self._to_tensor(self._augment(negative)))


def split_worlds(df, val_frac, test_frac, seed=42):
    worlds = list(df["world_id"].unique())
    random.Random(seed).shuffle(worlds)
    n = len(worlds)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_worlds = set(worlds[:n_test])
    val_worlds = set(worlds[n_test:n_test + n_val])
    train_worlds = set(worlds[n_test + n_val:])
    return train_worlds, val_worlds, test_worlds


def train(args):
    df = pd.read_csv(os.path.join(args.dataset, "ground_truth.csv"))
    hard_neg_df = pd.read_csv(args.hard_neg)

    train_worlds, val_worlds, test_worlds = split_worlds(df, args.val_frac, args.test_frac)
    train_df = df[df["world_id"].isin(train_worlds)]
    val_df = df[df["world_id"].isin(val_worlds)]
    test_df = df[df["world_id"].isin(test_worlds)]
    print(f"Worlds -> train:{len(train_worlds)} val:{len(val_worlds)} test:{len(test_worlds)}")
    print(f"Samples -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # test split is written out and NEVER read again during this script --
    # evaluate.py reads it to know which sample_ids it's allowed to score by default
    test_split_path = os.path.join(args.dataset, "test_split.csv")
    test_df[["sample_id"]].to_csv(test_split_path, index=False)
    print(f"Held-out test sample_ids written -> {test_split_path} "
          f"(never used for training or model selection)")

    train_loader = DataLoader(PairDataset(args.dataset, train_df, hard_neg_df, True),
                               batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(PairDataset(args.dataset, val_df, hard_neg_df, False),
                             batch_size=16, shuffle=False, num_workers=2)

    model = TinyEmbedNet(128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=0.3)

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for ref, pos, neg in train_loader:
            optimizer.zero_grad()
            loss = triplet_loss(model(ref), model(pos), model(neg))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ref, pos, neg in val_loader:
                e_ref, e_pos, e_neg = model(ref), model(pos), model(neg)
                d_pos = (e_ref - e_pos).pow(2).sum(1)
                d_neg = (e_ref - e_neg).pow(2).sum(1)
                correct += (d_pos < d_neg).sum().item()
                total += ref.size(0)
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc:.3f}  time={time.time()-t0:.1f}s")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)

    print(f"Saved best model (val_acc={best_val_acc:.3f}) -> {args.out}")
    print(f"Run evaluate.py now -- it will score ONLY the {len(test_df)}-sample "
          f"held-out test split by default, which is what should go in the report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./synthetic_sem_dataset")
    parser.add_argument("--hard_neg", default="hard_negatives.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--out", default="reranker.pt")
    args = parser.parse_args()
    train(args)
