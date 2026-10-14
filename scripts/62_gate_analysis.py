"""Gate activation patterns (Discussion, 'Analysis of Gate Activation Patterns'): mean +- SD of the gate values
G (weight of the DWSC branch) per stage, averaged over channels, using up to 500 random images per class.

Usage: python scripts/62_gate_analysis.py --run runs/hta_convnet --data-root data/PlantVillage/color
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, load_split, simple_loader
from common.loading import load_run
from common.utils import get_device, save_json


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--subset", default="test", choices=["test", "train", "val"])
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    dev = get_device()
    model, _ = load_run(a.run, len(PV_CLASSES), dev)
    df = load_split(a.data_root, a.split_csv)
    df = df[df.split == a.subset].sample(frac=1.0, random_state=0).groupby("label").head(a.per_class)
    model.set_record(True)
    per_stage, per_class = {k: [] for k in model.dbsa_blocks()}, {k: {} for k in model.dbsa_blocks()}
    for x, y in simple_loader(df, batch_size=64, img_size=a.img_size):
        model(x.to(dev))
        for k, b in model.dbsa_blocks().items():
            if b.last_gate is None:
                continue
            g = b.last_gate.mean(1).cpu().numpy()  # channel-averaged gate per image
            per_stage[k].append(g)
            for yy, gg in zip(y.numpy(), g):
                per_class[k].setdefault(PV_CLASSES[yy], []).append(float(gg))
    res = {}
    for k, v in per_stage.items():
        if v:
            v = np.concatenate(v)
            res[f"stage{k}"] = dict(mean=float(v.mean()), sd=float(v.std()),
                                    per_class_mean={c: float(np.mean(g)) for c, g in per_class[k].items()})
            print(f"Stage {k}: G = {v.mean():.2f} +- {v.std():.2f}  (G -> DWSC weight, 1-G -> MHSA weight)")
    save_json(res, os.path.join(a.run, "gate_analysis.json"))


if __name__ == "__main__":
    main()
