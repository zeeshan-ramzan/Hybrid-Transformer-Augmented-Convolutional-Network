"""Near-duplicate screening and group-aware split (Methods, 'Near-Duplicate Screening and Leakage Control').

(i) exact duplicates by MD5; (ii) candidate near-duplicates by 64-bit pHash, Hamming distance <= 6;
(iii) confirmation by cosine similarity >= 0.95 of ImageNet-pre-trained (not fine-tuned) EfficientNet-B4
embeddings. Confirmed pairs are linked into connected components (duplicate groups). Outputs:
  splits/duplicate_groups.csv, splits/group_aware_split.csv, splits/dedup_report.json

Usage: python scripts/01_near_duplicate_screening.py --data-root data/PlantVillage/color
"""
import argparse
import hashlib
import os
import sys
from collections import defaultdict

import imagehash
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import ImageListDataset, eval_transform, load_split, pv_classes, scan_imagefolder
from common.utils import get_device, save_json


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@torch.no_grad()
def embeddings(paths, device, bs=128, workers=8, pretrained=True):
    m = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier = torch.nn.Identity()
    m.eval().to(device)
    dl = DataLoader(ImageListDataset(paths, [0] * len(paths), eval_transform(224)), bs, num_workers=workers)
    out = []
    for x, _ in dl:
        f = m(x.to(device)).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--cos", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    a = ap.parse_args()
    df = scan_imagefolder(a.data_root, pv_classes(a.data_root))
    n = len(df)
    print(f"[dedup] {n} images")

    md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df.path]
    ph = np.array([int(str(imagehash.phash(Image.open(p).convert("RGB"), hash_size=8)), 16) for p in df.path],
                  dtype=np.uint64)
    uf = UnionFind(n)
    exact = 0
    by_md5 = defaultdict(list)
    for i, h in enumerate(md5):
        by_md5[h].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            exact += 1

    # candidate pairs: pHash Hamming <= t (checked within the same class, blockwise for memory)
    emb = embeddings(df.path.tolist(), get_device(), workers=a.workers, pretrained=not a.no_pretrained)
    confirmed = 0
    labels = df.label.values
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        h = ph[idx]
        for s in range(0, len(idx), 512):
            blk = h[s:s + 512]
            x = np.bitwise_xor(blk[:, None], h[None, :])
            dist = np.unpackbits(x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8), axis=2).sum(2)
            ii, jj = np.where(dist <= a.hamming)
            for u, v in zip(ii + s, jj):
                if u < v:
                    i, j = idx[u], idx[v]
                    if float(emb[i] @ emb[j]) >= a.cos:
                        uf.union(i, j)
                        confirmed += 1
    roots = np.array([uf.find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    df["group"] = group
    sizes = df.group.map(df.group.value_counts())
    multi = df[sizes > 1]
    os.makedirs(a.out_dir, exist_ok=True)
    df[["path", "label", "class_name", "group"]].to_csv(os.path.join(a.out_dir, "duplicate_groups.csv"), index=False)

    # leakage in the standard split
    std = load_split(a.data_root, a.split_csv, a.seed).merge(df[["path", "group"]], on="path")
    train_groups = set(std[std.split == "train"].group)
    test = std[std.split == "test"]
    leaked = test[test.group.isin(train_groups) & (test.group.map(std.group.value_counts()) > 1)]

    # group-aware stratified 70/15/15 split (every duplicate group in exactly one subset)
    n_splits = min(20, int(df.label.value_counts().min()))  # 20 folds -> 14/3/3 = 70/15/15
    if n_splits < 20:
        print(f"[dedup] warning: smallest class has {n_splits} images; using {n_splits} folds")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=a.seed)
    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(sgkf.split(df.path, df.label, df.group)):
        fold[te] = k
    df["split"] = np.where(fold < round(0.70 * n_splits), "train",
                           np.where(fold < round(0.85 * n_splits), "val", "test"))
    df[["path", "label", "class_name", "split", "group"]].to_csv(os.path.join(a.out_dir, "group_aware_split.csv"),
                                                                 index=False)
    rep = dict(images=n, exact_duplicate_links=exact, confirmed_near_duplicate_pairs=confirmed,
               near_duplicate_groups=int(multi.group.nunique()), images_in_groups=int(len(multi)),
               standard_split_test_images_with_train_duplicate=int(len(leaked)),
               standard_split_leak_percent=100 * len(leaked) / max(len(test), 1),
               group_aware_split_sizes=df.split.value_counts().to_dict())
    save_json(rep, os.path.join(a.out_dir, "dedup_report.json"))
    print(rep)


if __name__ == "__main__":
    main()
