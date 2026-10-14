"""Zero-shot cross-dataset evaluation (Tables 'cross' and 'crossdetail'; lenient PP2020 protocol).

For every trained run directory: unrestricted 38-way prediction on
  * PlantDoc test images mapped to PlantVillage classes (Table 'mapping'): accuracy, macro-F1, ECE, per-crop acc
  * Plant Pathology 2020 single-label images (healthy/scab/rust): accuracy, macro-F1, ECE
  * [NEW, R3.3] all 1,821 PP2020 images with the lenient rule for 'multiple_diseases' images (correct if the
    prediction is apple scab OR cedar apple rust) and the share of those images predicted healthy.
Macro-F1 is computed over the mapped classes present in the target set.

Usage: python scripts/53_cross_dataset_zero_shot.py --runs runs/hta_convnet runs/efficientnet_b4 runs/swin_t \
          --plantdoc-root data/PlantDoc-Dataset --pp2020-root data/plant-pathology-2020
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, load_plantdoc, load_pp2020, simple_loader
from common.loading import load_run
from common.metrics import ece
from common.trainer import predict
from common.utils import get_device, save_json


def summarise(y, probs):
    pred = probs.argmax(1)
    present = np.unique(y)
    return dict(n=int(len(y)), acc=100 * float((pred == y).mean()),
                f1=100 * float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
                ece=ece(y, probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--plantdoc-root", default="")
    ap.add_argument("--pp2020-root", default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/cross_dataset_results.json")
    a = ap.parse_args()
    dev = get_device()
    classes = PV_CLASSES
    pd_df = load_plantdoc(a.plantdoc_root, "test", classes) if a.plantdoc_root else None
    pp_all = load_pp2020(a.pp2020_root, classes, include_multiple=True) if a.pp2020_root else None
    scab, rust, healthy = (classes.index(c) for c in ("Apple___Apple_scab", "Apple___Cedar_apple_rust",
                                                      "Apple___healthy"))
    all_res = {}
    for run in a.runs:
        model, info = load_run(run, len(classes), dev)
        name = info["model"]
        r = {}
        if pd_df is not None:
            probs, y = predict(model, simple_loader(pd_df, a.img_size, workers=a.workers), dev)
            r["plantdoc"] = summarise(y, probs)
            r["plantdoc_per_crop"] = {c: dict(n=int(m.sum()), acc=100 * float((probs[m].argmax(1) == y[m]).mean()))
                                      for c in sorted(pd_df.crop.unique()) for m in [pd_df.crop.values == c]}
            np.savez_compressed(os.path.join(run, "plantdoc_predictions.npz"), probs=probs, labels=y,
                                paths=pd_df.path.values)
        if pp_all is not None:
            loader_df = pp_all.assign(label=pp_all.label.clip(lower=0))
            probs, _ = predict(model, simple_loader(loader_df, a.img_size, workers=a.workers), dev)
            y = pp_all.label.values
            single = y >= 0
            r["pp2020"] = summarise(y[single], probs[single])
            pred = probs.argmax(1)
            multi = ~single
            ok = np.where(single, pred == y, np.isin(pred, [scab, rust]))
            r["pp2020_lenient_all"] = dict(n=int(len(y)), acc=100 * float(ok.mean()),
                                           multi_n=int(multi.sum()),
                                           multi_correct_lenient=100 * float(ok[multi].mean()) if multi.any() else None,
                                           multi_pred_healthy=100 * float((pred[multi] == healthy).mean())
                                           if multi.any() else None)
            np.savez_compressed(os.path.join(run, "pp2020_predictions.npz"), probs=probs, labels=y,
                                paths=pp_all.path.values)
        all_res[name] = r
        print(name, {k: (v.get("acc") if isinstance(v, dict) and "acc" in v else "...") for k, v in r.items()})
    save_json(all_res, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
