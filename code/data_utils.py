"""
data_utils.py
=============
Shared data / evaluation infrastructure for the HTA-ConvNet reproduction.

Why a shared module?
--------------------
The paper states that "the same split protocol and evaluation metrics are used
throughout" for every method in Table 10.  Guaranteeing that is only possible if
one single, seed-driven splitter and one metric implementation are used
everywhere, so this file is imported by `hta_convnet.py` and by every baseline /
experiment script.  It contains **no model code** -- every architecture lives in
its own file.

Public API
----------
    load_index(...)                 -> DatasetIndex   (classes + file lists)
    make_splits(index, ...)         -> {"train": Subset, "val": ..., "test": ...}
    build_loaders(...)              -> torch DataLoaders with the paper's transforms
    load_plantdoc(...)              -> DatasetIndex for the cross-dataset test
    classification_metrics(y, yhat, probs) -> acc / macro P,R,F1 / AUC / ECE
    per_class_table(...)            -> Table 14 style rows
    figures: plot_curves, plot_per_class_f1, plot_confusion, plot_roc,
             plot_reliability, plot_bars, plot_grouped_bars, plot_lines,
             plot_image_grid, plot_scatter_labeled
    saving:  save_table, save_json, save_predictions, load_predictions
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config as C

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG")
_SPLIT_ALIASES = {
    "train": "train", "training": "train", "trn": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "eval": "test",
}


# ============================================================================
#  0. GENERIC HELPERS
# ============================================================================


def banner(title: str, char: str = "=") -> None:
    line = char * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(force_cpu: bool = False):
    import torch

    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context(device):
    """torch.amp autocast that is a guaranteed no-op on CPU.

    `torch.amp.autocast("cpu")` exists but bfloat16 CPU autocast slows this
    workload down and changes numerics, so mixed precision is enabled only on
    CUDA -- exactly the "AMP was used" claim of the paper, which ran on an A100.
    """
    import contextlib

    import torch

    if C.USE_AMP and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(device):
    import torch

    enabled = bool(C.USE_AMP and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:                                    # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0

    @property
    def seconds(self) -> float:
        return getattr(self, "elapsed", time.perf_counter() - self.t0)


def cli_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments understood by every script in this package."""
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the number of training epochs")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke-test schedule (1 epoch, 64 px, few images)")
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    parser.add_argument("--img-size", type=int, default=None,
                        help=f"network input size (paper: {C.IMG_SIZE})")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", default=C.DEFAULT_SPLIT,
                        choices=sorted(C.SPLIT_RATIOS),
                        help="train/val/test ratio (the paper gives both)")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--pretrained", action="store_true",
                        help="try to download ImageNet-1k weights (paper protocol; "
                             "needs internet, falls back to random init)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="cap the number of images used per class")
    parser.add_argument("--max-classes", type=int, default=None,
                        help="use only the first N classes (debugging)")
    parser.add_argument("--workers", type=int, default=None)
    return parser


@dataclass
class RunConfig:
    """Resolved runtime schedule shared by every script."""
    epochs: int
    img_size: int
    batch_size: int
    patience: int
    max_per_class: int | None
    max_classes: int | None
    device: object
    workers: int
    quick: bool
    split: str
    seed: int
    pretrained: bool


def resolve(args, default_epochs: int | None = None,
            default_patience: int | None = None) -> RunConfig:
    """Turn parsed CLI arguments into the concrete schedule this run will use."""
    q = C.QUICK
    quick = bool(getattr(args, "quick", False))
    epochs = args.epochs or (q["epochs"] if quick else (default_epochs or C.HTA["epochs"]))
    img = args.img_size or (q["img_size"] if quick else C.IMG_SIZE)
    bs = args.batch_size or (q["batch_size"] if quick else C.HTA["batch_size"])
    pat = (q["patience"] if quick
           else (default_patience if default_patience is not None
                 else C.HTA["early_stopping_patience"]))
    mpc = args.max_per_class if args.max_per_class is not None else (
        q["max_per_class"] if quick else None)
    mc = args.max_classes if args.max_classes is not None else (
        q["max_classes"] if quick else None)
    workers = args.workers if args.workers is not None else C.NUM_WORKERS
    return RunConfig(epochs=epochs, img_size=img, batch_size=bs, patience=pat,
                     max_per_class=mpc, max_classes=mc,
                     device=get_device(getattr(args, "cpu", False)),
                     workers=workers, quick=quick,
                     split=getattr(args, "split", C.DEFAULT_SPLIT),
                     seed=getattr(args, "seed", C.SEED),
                     pretrained=bool(getattr(args, "pretrained", False)))


def describe_run(rc: RunConfig, name: str) -> None:
    print(f"  run configuration [{name}]")
    print(f"    device {rc.device} | input {rc.img_size}x{rc.img_size} | "
          f"batch {rc.batch_size} | epochs {rc.epochs} | split {rc.split} | "
          f"seed {rc.seed}" + ("  (QUICK SMOKE TEST)" if rc.quick else ""))


# ============================================================================
#  1. DATASET DISCOVERY
# ============================================================================


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _images_in(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir()
                   if p.is_file() and p.suffix in IMAGE_EXTS and not p.name.startswith(".")])


def _subdirs(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _class_map(root: Path) -> dict[str, list[Path]]:
    """{class_name: [image paths]} for an ImageFolder-style directory."""
    out: dict[str, list[Path]] = {}
    for sub in _subdirs(root):
        files = _images_in(sub)
        if not files:                       # allow one extra nesting level
            nested: list[Path] = []
            for s2 in _subdirs(sub):
                nested += _images_in(s2)
            files = sorted(nested)
        if files:
            out[sub.name] = files
    return out


def _looks_presplit(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for sub in _subdirs(root):
        key = _SPLIT_ALIASES.get(_norm(sub.name))
        if key and _class_map(sub):
            found.setdefault(key, sub)
    return found if {"train", "test"} <= set(found) else {}


def _read_user_config() -> dict:
    if C.DATASET_CONFIG_FILE.exists():
        try:
            cfg = json.loads(C.DATASET_CONFIG_FILE.read_text())
            print(f"  [config] using {C.DATASET_CONFIG_FILE.name}")
            return {k: v for k, v in cfg.items() if not k.startswith("_")}
        except Exception as e:                            # pragma: no cover
            print(f"  [!] could not parse {C.DATASET_CONFIG_FILE}: {e}")
    return {}


def _resolve_in_root(value) -> Path:
    """Interpret a user-supplied path, forbidding escapes from the project root."""
    p = Path(value)
    p = p if p.is_absolute() else (C.DATASET_DIR / p)
    p = p.resolve()
    if C.PROJECT_ROOT.resolve() not in p.parents and p != C.PROJECT_ROOT.resolve():
        raise SystemExit(f"[!] dataset path {p} lies outside the project root; "
                         "paths in dataset_config.json must stay inside the package.")
    return p


NO_DATA_MSG = """
No PlantVillage-style image dataset was found in
    {ds}

Put the data there in ANY of these layouts (the loader auto-detects all three):

  A. plain ImageFolder
        dataset/Apple___Apple_scab/*.jpg
        dataset/Apple___Black_rot/*.jpg
        ...
  B. one wrapping folder
        dataset/PlantVillage/Apple___Apple_scab/*.jpg
        ...
  C. already split
        dataset/train/Apple___Apple_scab/*.jpg
        dataset/val/Apple___Apple_scab/*.jpg
        dataset/test/Apple___Apple_scab/*.jpg

PlantVillage itself: https://github.com/spMohanty/PlantVillage-Dataset
(38 classes, 54,305 colour images).  Nothing is downloaded automatically.

If auto-detection fails, write dataset/dataset_config.json -- see
dataset/dataset_config.example.json and README section 3.

No data yet?  Generate a small SIMULATED stand-in so every script runs:

    cd code
    python make_synthetic_dataset.py
"""


@dataclass
class DatasetIndex:
    """Every image path in the dataset plus its integer label."""
    root: Path
    name: str
    classes: list[str]
    files: list[Path]
    labels: np.ndarray
    presplit: dict[str, np.ndarray] = field(default_factory=dict)
    layout: str = "imagefolder"
    synthetic: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_classes)


def _build_index(name: str, root: Path, per_split: dict[str, dict[str, list[Path]]],
                 max_per_class: int | None, max_classes: int | None,
                 seed: int = C.SPLIT_SEED) -> DatasetIndex:
    """Merge {split: {class: files}} into one flat, optionally sub-sampled index."""
    classes = sorted({c for m in per_split.values() for c in m})
    if max_classes is not None and max_classes < len(classes):
        classes = classes[:max_classes]
    cls_idx = {c: i for i, c in enumerate(classes)}

    rng = np.random.default_rng(seed)
    files: list[Path] = []
    labels: list[int] = []
    origin: list[str] = []
    for split, m in per_split.items():
        for c in classes:
            fs = list(m.get(c, []))
            if max_per_class is not None and len(fs) > max_per_class:
                # deterministic sub-sample, class-balanced
                keep = rng.choice(len(fs), size=max_per_class, replace=False)
                fs = [fs[i] for i in sorted(keep)]
            files += fs
            labels += [cls_idx[c]] * len(fs)
            origin += [split] * len(fs)

    labels_a = np.asarray(labels, dtype=np.int64)
    presplit: dict[str, np.ndarray] = {}
    if set(per_split) != {"__all__"}:
        origin_a = np.asarray(origin)
        for s in ("train", "val", "test"):
            idx = np.flatnonzero(origin_a == s)
            if idx.size:
                presplit[s] = idx
    layout = "presplit" if presplit else "imagefolder"
    synth = any("SYNTHETIC" in p.parts[-3:-1] or p.name.startswith("synth_")
                for p in files[:1]) or (root / ".synthetic").exists()
    return DatasetIndex(root=root, name=name, classes=classes, files=files,
                        labels=labels_a, presplit=presplit, layout=layout,
                        synthetic=synth)


def load_index(max_per_class: int | None = None, max_classes: int | None = None,
               verbose: bool = True, require: bool = True) -> DatasetIndex:
    """Locate and index the training dataset (PlantVillage).

    Raises SystemExit with a full explanation when nothing usable is found.
    """
    cfg = _read_user_config()
    roots: list[Path] = []
    if cfg.get("root"):
        roots.append(_resolve_in_root(cfg["root"]))
    if any(cfg.get(k) for k in ("train_dir", "test_dir")):
        per_split = {}
        for k, s in (("train_dir", "train"), ("val_dir", "val"), ("test_dir", "test")):
            if cfg.get(k):
                m = _class_map(_resolve_in_root(cfg[k]))
                if m:
                    per_split[s] = m
        if per_split:
            idx = _build_index("PlantVillage", C.DATASET_DIR, per_split,
                               max_per_class, max_classes)
            if verbose:
                _describe(idx)
            return idx

    excluded = {_norm(n) for n in C.PLANTDOC["dir_names"]}
    roots.append(C.DATASET_DIR)
    for sub in _subdirs(C.DATASET_DIR):
        if _norm(sub.name) not in excluded:
            roots.append(sub)

    for root in roots:
        if not root.exists():
            continue
        ps = _looks_presplit(root)
        if ps:
            per_split = {s: _class_map(d) for s, d in ps.items()}
            idx = _build_index("PlantVillage", root, per_split, max_per_class, max_classes)
            if verbose:
                _describe(idx)
            return idx
        m = _class_map(root)
        m = {k: v for k, v in m.items() if _norm(k) not in excluded}
        if len(m) >= 2:
            idx = _build_index("PlantVillage", root, {"__all__": m},
                               max_per_class, max_classes)
            if verbose:
                _describe(idx)
            return idx

    if not require:
        return None                                        # type: ignore[return-value]
    raise SystemExit(NO_DATA_MSG.format(ds=C.DATASET_DIR))


def load_plantdoc(max_per_class: int | None = None,
                  classes: list[str] | None = None,
                  verbose: bool = True) -> DatasetIndex | None:
    """Locate the PlantDoc field-image set used for the Table 12 cross-dataset test."""
    cfg = _read_user_config()
    cands: list[Path] = []
    if cfg.get("plantdoc_dir"):
        cands.append(_resolve_in_root(cfg["plantdoc_dir"]))
    for n in C.PLANTDOC["dir_names"]:
        cands.append(C.DATASET_DIR / n)
    for c in cands:
        if not c.exists():
            continue
        ps = _looks_presplit(c)
        root = ps.get("test", c) if ps else c
        m = _class_map(root)
        if len(m) >= 2:
            if classes is not None:
                lut = {_norm(k): k for k in m}
                m = {c2: m[lut[_norm(c2)]] for c2 in classes if _norm(c2) in lut}
                if not m:
                    print("  [!] PlantDoc found but no class name matches the "
                          "training label space -- skipping.")
                    return None
            idx = _build_index("PlantDoc", root, {"__all__": m}, max_per_class, None)
            if classes is not None:                       # keep the training label ids
                lut2 = {c2: i for i, c2 in enumerate(classes)}
                idx.labels = np.asarray([lut2[idx.classes[l]] for l in idx.labels],
                                        dtype=np.int64)
                idx.classes = list(classes)
            if verbose:
                _describe(idx)
            return idx
    return None


def _describe(idx: DatasetIndex) -> None:
    cnt = np.bincount(idx.labels, minlength=idx.n_classes)
    print(f"  [{idx.name}] {len(idx.files):,} images | {idx.n_classes} classes | "
          f"layout={idx.layout} | root={idx.root}")
    if cnt.size:
        lo, hi = int(cnt.min()), int(cnt.max())
        j = int(np.argmin(cnt)); k = int(np.argmax(cnt))
        print(f"    class sizes: min {lo} ({idx.classes[j]}), max {hi} "
              f"({idx.classes[k]}), imbalance ratio {hi / max(lo, 1):.1f}:1")
    if idx.presplit:
        print("    pre-split sizes: " +
              ", ".join(f"{s} {len(v):,}" for s, v in idx.presplit.items()))
    if idx.name == "PlantVillage" and idx.n_classes != C.PAPER_N_CLASSES:
        print(f"    [note] the paper uses {C.PAPER_N_CLASSES} classes; this run has "
              f"{idx.n_classes}. Absolute numbers are not comparable to the paper.")


# ============================================================================
#  2. SPLITS  (paper "Data Partitioning" / Table 5)
# ============================================================================


def stratified_split(labels: np.ndarray, ratios: tuple[float, float, float],
                     seed: int = C.SPLIT_SEED) -> dict[str, np.ndarray]:
    """Stratified train/val/test partition with a fixed seed (paper: seed 42).

    Every class is shuffled independently and cut at the ratio boundaries, so the
    class proportions of each subset track the full dataset (paper Table 5).
    Classes with fewer than three images still contribute at least one training
    image; val/test then simply get fewer.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(ratios[0] * n)))
        n_va = int(round(ratios[1] * n))
        if n_tr + n_va >= n:                       # guarantee a non-empty test slice
            n_va = max(0, n - n_tr - 1)
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    out = {"train": np.concatenate(tr), "val": np.concatenate(va),
           "test": np.concatenate(te)}
    for k in out:
        out[k] = np.sort(out[k])
    if out["val"].size == 0:                       # degenerate tiny datasets
        out["val"] = out["test"]
    return out


def stratified_folds(labels: np.ndarray, n_folds: int = C.N_FOLDS,
                     seed: int = C.SPLIT_SEED) -> list[np.ndarray]:
    """Stratified k-fold index lists (paper "Cross-Validation Protocol", Table 4)."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_folds].append(int(j))
    return [np.sort(np.asarray(f, dtype=np.int64)) for f in folds]


def make_splits(idx: DatasetIndex, split: str = C.DEFAULT_SPLIT,
                seed: int = C.SPLIT_SEED, verbose: bool = True) -> dict[str, np.ndarray]:
    """Honour a pre-split folder layout, otherwise apply the paper's ratios."""
    if idx.presplit:
        s = dict(idx.presplit)
        s.setdefault("val", s["test"])
        if verbose:
            print(f"  using the dataset's own train/val/test folders "
                  f"({', '.join(f'{k} {len(v):,}' for k, v in s.items())}); "
                  f"the --split ratio is ignored")
        return s
    s = stratified_split(idx.labels, C.SPLIT_RATIOS[split], seed)
    if verbose:
        pct = C.SPLIT_RATIOS[split]
        print(f"  stratified split {split} (seed {seed}): " +
              ", ".join(f"{k} {len(v):,} ({100 * p:.0f}%)"
                        for (k, v), p in zip(s.items(), pct)))
    return s


def split_table(idx: DatasetIndex, splits: dict[str, np.ndarray]) -> list[dict]:
    """Paper Table 5 -- dataset split statistics."""
    total = sum(len(v) for v in splits.values())
    rows = []
    for name in ("train", "val", "test"):
        v = splits[name]
        rows.append({"Split": name.capitalize(), "Images": int(len(v)),
                     "% of Total": round(100 * len(v) / max(total, 1), 1),
                     "Classes": int(len(np.unique(idx.labels[v])))})
    rows.append({"Split": "Total", "Images": int(total), "% of Total": 100.0,
                 "Classes": idx.n_classes})
    return rows


# ============================================================================
#  3. TRANSFORMS AND DATALOADERS  (paper Table 3)
# ============================================================================


def _tv():
    try:
        import torchvision  # noqa: F401
    except Exception as e:                                  # pragma: no cover
        raise SystemExit(
            "[!] torchvision is required (backbones and image transforms).\n"
            f"    import failed with: {e}\n"
            "    pip install torchvision")
    from torchvision import transforms

    return transforms


def build_transforms(img_size: int = C.IMG_SIZE, train: bool = True):
    """Paper Table 3 for training; resize + centre crop + normalise for eval."""
    T = _tv()
    A = C.AUGMENT
    norm = T.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD)
    if not train:
        resize = max(img_size + 1, int(round(img_size * C.RESIZE_SIZE / C.IMG_SIZE)))
        return T.Compose([T.Resize(resize), T.CenterCrop(img_size), T.ToTensor(), norm])

    k = A["blur_kernel"] | 1
    ops = [
        T.RandomResizedCrop(img_size, scale=A["rrc_scale"], antialias=True),
        T.RandomHorizontalFlip(p=A["hflip_p"]),
        T.RandomVerticalFlip(p=A["vflip_p"]),
        T.RandomRotation(A["rotation_deg"]),
        T.RandomApply([T.ColorJitter(A["jitter_brightness"], A["jitter_contrast"],
                                     A["jitter_saturation"], A["jitter_hue"])],
                      p=A["jitter_p"]),
        T.RandomApply([T.GaussianBlur(k, sigma=A["blur_sigma"])], p=A["blur_p"]),
        T.ToTensor(),
        norm,
        T.RandomErasing(p=A["erasing_p"], scale=A["erasing_scale"]),
    ]
    return T.Compose(ops)


class LeafDataset:
    """Minimal torch Dataset over a list of image paths."""

    def __init__(self, files, labels, transform, cache: dict | None = None):
        self.files = list(files)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        p = self.files[i]
        if self.cache is not None and p in self.cache:
            img = self.cache[p]
        else:
            with Image.open(p) as im:
                img = im.convert("RGB")
            if self.cache is not None:
                self.cache[p] = img
        return self.transform(img), int(self.labels[i])


def class_weights(labels: np.ndarray, n_classes: int, scheme: str = "inv_sqrt") -> np.ndarray:
    """Eq. (7) class weights w_c = 1 / sqrt(N_c), normalised to mean 1."""
    cnt = np.bincount(labels, minlength=n_classes).astype(np.float64)
    cnt[cnt == 0] = 1.0
    if scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(cnt)
    elif scheme == "inv":
        w = 1.0 / cnt
    else:
        w = np.ones_like(cnt)
    return w / w.mean()


def build_loaders(idx: DatasetIndex, splits: dict[str, np.ndarray], rc: RunConfig,
                  min_size: int | None = None, train_indices: np.ndarray | None = None,
                  cache_images: bool = True):
    """DataLoaders for the three splits, at the resolution this run uses."""
    import torch
    from torch.utils.data import DataLoader

    size = rc.img_size if min_size is None else max(rc.img_size, min_size)
    cache: dict | None = {} if (cache_images and len(idx.files) <= 20000) else None
    tr_idx = splits["train"] if train_indices is None else train_indices

    def make(indices, train):
        ds = LeafDataset([idx.files[i] for i in indices], idx.labels[indices],
                         build_transforms(size, train=train), cache=cache)
        g = torch.Generator()
        g.manual_seed(rc.seed)
        return DataLoader(ds, batch_size=rc.batch_size, shuffle=train,
                          num_workers=rc.workers, drop_last=False, generator=g,
                          pin_memory=False)

    return {"train": make(tr_idx, True),
            "val": make(splits["val"], False),
            "test": make(splits["test"], False)}, size


def build_eval_loader(idx: DatasetIndex, indices, rc: RunConfig, size: int):
    from torch.utils.data import DataLoader

    ds = LeafDataset([idx.files[i] for i in indices], idx.labels[indices],
                     build_transforms(size, train=False))
    return DataLoader(ds, batch_size=rc.batch_size, shuffle=False,
                      num_workers=rc.workers, pin_memory=False)


# ============================================================================
#  4. METRICS  ("Evaluation Metrics", "Calibration Analysis")
# ============================================================================


def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = C.ECE_BINS) -> float:
    """ECE with `n_bins` equal-width confidence intervals (paper: 15 bins)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = C.ECE_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, confs, ns = [], [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        accs.append(float(acc[m].mean()) if m.any() else np.nan)
        confs.append(float(conf[m].mean()) if m.any() else np.nan)
        ns.append(int(m.sum()))
    return np.array(centres), np.array(accs), np.array(confs), np.array(ns)


def macro_auc(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """One-vs-rest AUC per class plus macro and micro averages."""
    from sklearn.metrics import roc_auc_score

    per: dict[int, float] = {}
    for c in range(n_classes):
        pos = (y == c)
        if pos.any() and not pos.all():
            try:
                per[c] = float(roc_auc_score(pos.astype(int), probs[:, c]))
            except Exception:
                pass
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    try:
        micro = float(roc_auc_score(onehot.ravel(), probs.ravel()))
    except Exception:
        micro = float("nan")
    return {"per_class": per,
            "macro": float(np.mean(list(per.values()))) if per else float("nan"),
            "micro": micro}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           probs: np.ndarray | None = None,
                           n_classes: int | None = None) -> dict:
    """Accuracy + macro precision / recall / F1 (+ AUC and ECE when probs given).

    Macro averaging treats all classes equally, exactly as the paper specifies,
    so a strong result on the dominant healthy classes cannot mask weak
    performance on rare disease categories.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = n_classes or int(max(y_true.max(), y_pred.max()) + 1)
    labels = list(range(n_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    out = {"accuracy": float((y_true == y_pred).mean() * 100),
           "precision_macro": float(p * 100), "recall_macro": float(r * 100),
           "f1_macro": float(f * 100), "n": int(len(y_true))}
    if probs is not None:
        auc = macro_auc(probs, y_true, n_classes)
        out["auc_macro"] = auc["macro"]
        out["auc_micro"] = auc["micro"]
        out["ece"] = expected_calibration_error(probs, y_true)
        out["nll"] = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true],
                                           1e-12, None)).mean())
    return out


def per_class_table(y_true, y_pred, classes: list[str]) -> list[dict]:
    """Paper Table 14 -- per-class precision, recall and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    rows = [{"Idx": i + 1, "Class Name": classes[i], "Support": int(s[i]),
             "Prec (%)": round(float(p[i] * 100), 2),
             "Rec (%)": round(float(r[i] * 100), 2),
             "F1 (%)": round(float(f[i] * 100), 2)} for i in range(len(classes))]
    rows.append({"Idx": "Macro Average", "Class Name": "---",
                 "Support": int(s.sum()),
                 "Prec (%)": round(float(p.mean() * 100), 2),
                 "Rec (%)": round(float(r.mean() * 100), 2),
                 "F1 (%)": round(float(f.mean() * 100), 2)})
    return rows


def confusion(y_true, y_pred, n_classes: int, normalise: bool = True) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(np.float64)
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return cm


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 10) -> list[dict]:
    m = cm.copy()
    np.fill_diagonal(m, 0.0)
    flat = np.argsort(m, axis=None)[::-1][:k]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, m.shape)
        if m[i, j] <= 0:
            continue
        rows.append({"True class": classes[i], "Predicted as": classes[j],
                     "Rate (%)": round(float(m[i, j] * 100), 2)})
    return rows


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test with continuity correction ("Statistical Significance")."""
    from scipy import stats

    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "chi2": float(chi2),
            "p_value": float(stats.chi2.sf(chi2, 1))}


def aggregate(runs: list[dict], keys=("accuracy", "precision_macro",
                                      "recall_macro", "f1_macro")) -> dict:
    out: dict[str, float] = {}
    for k in keys:
        v = [r[k] for r in runs if k in r]
        if v:
            out[k] = float(np.mean(v))
            out[k + "_std"] = float(np.std(v))
    out["n_runs"] = len(runs)
    return out


# ============================================================================
#  5. SAVING
# ============================================================================


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2))
    print(f"  [saved] {path.relative_to(C.PROJECT_ROOT)}")


def save_table(rows: list[dict], path: Path, title: str = "",
               print_it: bool = True, max_print_rows: int = 45) -> None:
    """Write a table as .csv AND .md, and echo it to the terminal."""
    import pandas as pd

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    path.with_suffix(".md").write_text((f"### {title}\n\n" if title else "") + md + "\n")
    if print_it:
        if title:
            print(f"\n{title}")
        with pd.option_context("display.width", 220, "display.max_columns", 60):
            if len(df) > max_print_rows:
                print(df.head(max_print_rows).to_string(index=False))
                print(f"  ... {len(df) - max_print_rows} more rows in the csv")
            else:
                print(df.to_string(index=False))
    print(f"  [saved] {path.with_suffix('.csv').relative_to(C.PROJECT_ROOT)}")


def slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(name))).strip("_")


def save_predictions(name: str, y_true, y_pred, probs=None, tag: str = "") -> None:
    """Per-sample test predictions so compare_all.py can pair methods sample-wise."""
    d = C.RESULTS_DIR / "_predictions"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{slug(name)}{('_' + tag) if tag else ''}.npz",
                        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
                        probs=np.asarray(probs) if probs is not None else np.array([]))


def load_predictions(name: str, tag: str = ""):
    f = C.RESULTS_DIR / "_predictions" / f"{slug(name)}{('_' + tag) if tag else ''}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["y_true"], z["y_pred"], (z["probs"] if z["probs"].size else None)


def read_result(result_name: str) -> dict | None:
    f = C.RESULTS_DIR / result_name / f"{result_name}_results.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


# ============================================================================
#  6. FIGURES
# ============================================================================


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": C.FIG_DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _saved(path: Path) -> None:
    print(f"  [saved] {path.relative_to(C.PROJECT_ROOT)}")


def plot_curves(history: dict, path: Path, title: str, ylabel: str,
                keys=("train", "val"), labels=("Training", "Validation")) -> None:
    """Paper Figs. 4 and 6 -- training / validation accuracy and loss per epoch."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    styles = [("-", "s", C.PALETTE["primary"]), ("--", "^", C.PALETTE["secondary"])]
    for (k, lab, (ls, mk, col)) in zip(keys, labels, styles):
        y = np.asarray(history.get(k, []), dtype=float)
        if y.size == 0:
            continue
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, ls, marker=mk, ms=3.5, color=col, label=lab, lw=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_per_class_f1(f1: np.ndarray, path: Path, title: str,
                      threshold: float = 99.5, classes: list[str] | None = None) -> None:
    """Paper Fig. 5 -- per-class F1 bars, split at a reference threshold."""
    plt = _plt()
    f1 = np.asarray(f1, dtype=float)
    n = len(f1)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.19 * n + 2.5), 3.6))
    colors = [C.PALETTE["primary"] if v >= threshold else C.PALETTE["warn"] for v in f1]
    ax.bar(np.arange(1, n + 1), f1, color=colors, width=0.75)
    ax.axhline(threshold, color=C.PALETTE["grey"], ls=":", lw=1.0)
    lo = float(np.nanmin(f1)) if n else 0.0
    ax.set_ylim(max(0.0, lo - 2.0), 100.5)
    ax.set_xlabel("Class index")
    ax.set_ylabel("F1-score (%)")
    ax.set_title(title, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.PALETTE["primary"]),
               plt.Rectangle((0, 0), 1, 1, color=C.PALETTE["warn"])]
    ax.legend(handles, [f"F1 >= {threshold}%", f"F1 < {threshold}%"], fontsize=7)
    if classes is not None and n <= 20:
        ax.set_xticks(np.arange(1, n + 1))
        ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=6)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_confusion(cm: np.ndarray, path: Path, title: str,
                   classes: list[str] | None = None) -> None:
    plt = _plt()
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(4.6, 0.26 * n + 2.2),
                                    max(4.0, 0.26 * n + 1.8)))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=max(1e-9, float(cm.max())))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontweight="bold")
    ax.grid(False)
    if classes is not None and n <= 20:
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=6)
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="w" if cm[i, j] < 0.6 else "k")
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_roc(probs: np.ndarray, y: np.ndarray, classes: list[str], path: Path,
             title: str, max_curves: int = 6) -> list[dict]:
    """Paper Fig. 7 -- one-vs-rest ROC curves for representative classes."""
    from sklearn.metrics import roc_curve, roc_auc_score

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    rows = []
    present = [c for c in range(len(classes))
               if (y == c).any() and not (y == c).all()]
    step = max(1, len(present) // max_curves)
    chosen = present[::step][:max_curves]
    for c in chosen:
        pos = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(pos, probs[:, c])
        a = float(roc_auc_score(pos, probs[:, c]))
        ax.plot(fpr, tpr, lw=1.3, label=f"{classes[c][:26]} (AUC={a:.4f})")
        rows.append({"Class": classes[c], "AUC": round(a, 4)})
    ax.plot([0, 1], [0, 1], "--", color=C.PALETTE["grey"], lw=1.0, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    _saved(path)
    return rows


def plot_reliability(probs: np.ndarray, y: np.ndarray, path: Path, title: str,
                     n_bins: int = C.ECE_BINS) -> None:
    plt = _plt()
    centres, accs, confs, ns = reliability_bins(probs, y, n_bins)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot([0, 1], [0, 1], "--", color=C.PALETTE["grey"], lw=1.0, label="Perfect calibration")
    m = ~np.isnan(accs)
    ax.bar(centres[m], accs[m], width=0.9 / n_bins, color=C.PALETTE["primary"],
           edgecolor="white", label="Empirical accuracy")
    ax.plot(centres[m], confs[m], "o-", ms=3, color=C.PALETTE["secondary"],
            label="Mean confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_bars(labels, values, path: Path, title: str, ylabel: str,
              highlight: str | None = None, rotate: int = 30,
              annotate: bool = True, color=None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(5.0, 0.62 * len(labels) + 2.0), 3.8))
    cols = [(C.PALETTE["secondary"] if (highlight and h in str(l)) else
             (color or C.PALETTE["primary"])) for l in labels
            for h in [highlight or "\0"]]
    ax.bar(range(len(labels)), values, color=cols, width=0.68)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center",
                       fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    v = np.asarray(values, dtype=float)
    if v.size:
        pad = max(0.4, 0.08 * float(np.ptp(v)))
        ax.set_ylim(max(0.0, float(v.min()) - pad), float(v.max()) + pad * 1.6)
        if annotate:
            for i, val in enumerate(v):
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_grouped_bars(categories, series: dict, path: Path, title: str,
                      ylabel: str, rotate: int = 30) -> None:
    plt = _plt()
    n, k = len(categories), max(1, len(series))
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(5.5, 0.95 * n + 2.0), 3.9))
    cols = list(C.PALETTE.values())
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(np.arange(n) + i * w - 0.4 + w / 2, vals, width=w, label=name,
               color=cols[i % len(cols)])
    ax.set_xticks(range(n))
    ax.set_xticklabels(categories, rotation=rotate, ha="right" if rotate else "center",
                       fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=min(4, k))
    allv = np.concatenate([np.asarray(v, dtype=float) for v in series.values()]) \
        if series else np.array([0.0])
    pad = max(0.5, 0.10 * float(np.ptp(allv)))
    ax.set_ylim(max(0.0, float(allv.min()) - pad), float(allv.max()) + pad)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_lines(x, series: dict, path: Path, title: str, xlabel: str, ylabel: str,
               marker: str = "o", highlight_x=None, logx: bool = False) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = list(C.PALETTE.values())
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, marker=marker, ms=4, lw=1.4, label=name, color=cols[i % len(cols)])
    if highlight_x is not None and highlight_x in list(x):
        j = list(x).index(highlight_x)
        first = list(series.values())[0]
        ax.plot([x[j]], [first[j]], "o", ms=8, color=C.PALETTE["secondary"],
                label=f"Optimal ({highlight_x})")
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7.5)
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_scatter_labeled(x, y, labels, path: Path, title: str, xlabel: str,
                         ylabel: str, highlight: str | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for xi, yi, li in zip(x, y, labels):
        hl = highlight is not None and highlight in str(li)
        ax.scatter([xi], [yi], s=70 if hl else 38,
                   color=C.PALETTE["secondary"] if hl else C.PALETTE["primary"],
                   zorder=3, edgecolor="white")
        ax.annotate(str(li), (xi, yi), textcoords="offset points", xytext=(5, 4),
                    fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_image_grid(paths, captions, path: Path, title: str, ncols: int = 3) -> None:
    """Paper Fig. 1 -- representative leaf samples, one per class."""
    from PIL import Image

    plt = _plt()
    n = len(paths)
    if n == 0:
        return
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.45 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, cap in zip(axes, paths, captions):
        try:
            with Image.open(p) as im:
                ax.imshow(np.asarray(im.convert("RGB")))
        except Exception:
            continue
        ax.set_title(str(cap), fontsize=6.5)
    fig.suptitle(title, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _saved(path)


def plot_heatmap_panels(images, titles, path: Path, suptitle: str, ncols: int = 4) -> None:
    """Grad-CAM panels: (rgb, heat) pairs already blended into RGB arrays."""
    plt = _plt()
    n = len(images)
    if n == 0:
        return
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.35 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(t, fontsize=6.5)
    fig.suptitle(suptitle, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _saved(path)
