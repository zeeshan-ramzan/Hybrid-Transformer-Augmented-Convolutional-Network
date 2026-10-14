"""General helpers: seeding, argument parsing, I/O."""
import argparse
import json
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by all training scripts (defaults = manuscript protocol)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", default="data/PlantVillage/color",
                   help="PlantVillage root with one sub-folder per class")
    p.add_argument("--split-csv", default="splits/standard_split.csv",
                   help="CSV with columns path,label,split (created automatically if missing)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-grid", type=str, default="",
                   help="comma-separated LRs, e.g. 5e-5,1e-4,3e-4: selects the LR on validation loss")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-warmup", action="store_true", help="disable LR warm-up (stability test)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (debug only)")
    p.add_argument("--max-train-batches", type=int, default=0, help="debug: limit batches/epoch")
    return p
