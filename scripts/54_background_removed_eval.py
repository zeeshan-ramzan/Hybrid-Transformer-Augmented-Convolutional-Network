"""Background-bias test (Table 'dedup', row 3): models trained on the standard split are evaluated on the
background-removed ('segmented') version of the same PlantVillage test images.

The public release stores them as segmented/<class>/<stem>_final_masked.jpg next to color/<class>/<stem>.JPG.

Usage: python scripts/54_background_removed_eval.py --runs runs/hta_convnet runs/efficientnet_b4 \
          --color-root data/PlantVillage/color --segmented-root data/PlantVillage/segmented
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, load_split, simple_loader
from common.loading import load_run
from common.metrics import classification_metrics
from common.trainer import predict
from common.utils import get_device, save_json


def to_segmented(path, color_root, seg_root):
    rel = os.path.relpath(path, color_root)
    cls, fname = os.path.split(rel)
    stem = os.path.splitext(fname)[0]
    for cand in (f"{stem}_final_masked.jpg", fname, f"{stem}.jpg", f"{stem}.JPG"):
        p = os.path.join(seg_root, cls, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--color-root", default="data/PlantVillage/color")
    ap.add_argument("--segmented-root", default="data/PlantVillage/segmented")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    dev = get_device()
    test = load_split(a.color_root, a.split_csv)
    test = test[test.split == "test"].copy()
    test["seg"] = [to_segmented(p, a.color_root, a.segmented_root) for p in test.path]
    missing = int(test.seg.isna().sum())
    seg = test.dropna(subset=["seg"]).assign(path=lambda d: d.seg)
    print(f"segmented test images found: {len(seg)} (missing {missing})")
    res = {}
    for run in a.runs:
        model, info = load_run(run, len(PV_CLASSES), dev)
        probs, y = predict(model, simple_loader(seg, workers=a.workers, img_size=a.img_size), dev)
        res[info["model"]] = dict(missing=missing, **classification_metrics(y, probs))
        print(info["model"], res[info["model"]]["acc"])
    save_json(res, "runs/background_removed.json")


if __name__ == "__main__":
    main()
