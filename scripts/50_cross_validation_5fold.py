"""Stratified 5-fold cross-validation of HTA-ConvNet (Table 'cv') + out-of-fold probabilities for the label
audit (02_label_audit.py).

Each fold: 80% training (of which 10% is held out for early stopping) and 20% evaluation.

Usage: python scripts/50_cross_validation_5fold.py --data-root data/PlantVillage/color [-- extra train flags]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import pv_classes, scan_imagefolder
from common.metrics import auc_ovr, t_ci
from common.utils import save_json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-script", default="10_hta_convnet.py")
    a, extra = ap.parse_known_args()
    df = scan_imagefolder(a.data_root, pv_classes(a.data_root))
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    rows, oof = [], []
    for k, (tr, te) in enumerate(skf.split(df.path, df.label), start=1):
        d = df.copy()
        d["split"] = "test"
        tr_i, va_i = train_test_split(tr, test_size=0.1, stratify=df.label.values[tr], random_state=a.seed)
        d.loc[tr_i, "split"] = "train"
        d.loc[va_i, "split"] = "val"
        csv = f"splits/cv_fold{k}.csv"
        os.makedirs("splits", exist_ok=True)
        d.to_csv(csv, index=False)
        name = f"cv_fold{k}"
        subprocess.run([sys.executable, os.path.join(HERE, a.model_script), "--name", name, "--split-csv", csv,
                        "--data-root", a.data_root, "--out-dir", "runs/cv"] + extra, check=True)
        with open(f"runs/cv/{name}/results.json") as f:
            r = json.load(f)
        z = np.load(f"runs/cv/{name}/test_predictions.npz", allow_pickle=True)
        auc = auc_ovr(z["labels"], z["probs"])["micro"]
        rows.append(dict(fold=k, acc=r["test"]["acc"], f1=r["test"]["f1"], auc=auc))
        oof.append(pd.DataFrame(dict(path=z["paths"], label=z["labels"],
                                     p_true=z["probs"][np.arange(len(z["labels"])), z["labels"]],
                                     pred=z["probs"].argmax(1))))
    t = pd.DataFrame(rows)
    summ = {m: dict(zip(["mean", "std", "ci_low", "ci_high"], t_ci(t[m]))) for m in ("acc", "f1", "auc")}
    save_json(dict(folds=rows, summary=summ), "runs/cv/cv_summary.json")
    pd.concat(oof).to_csv("runs/cv/out_of_fold_predictions.csv", index=False)
    print(t.to_string(index=False))
    print(summ)


if __name__ == "__main__":
    main()
