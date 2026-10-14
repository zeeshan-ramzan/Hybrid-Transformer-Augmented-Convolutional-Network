"""Calibration (ECE, 15 bins), one-vs-rest ROC/AUC (Fig. 'roc', Table 'auc'), per-class metrics (Table
'perclass', Fig. 'f1hist'), confusion matrix and error-type analysis (diseased->healthy, healthy->diseased).

Works on any run directory that contains test_predictions.npz.

Usage: python scripts/64_calibration_roc_confusion.py --run runs/hta_convnet
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES
from common.metrics import auc_ovr, classification_metrics, confusion, per_class_metrics, roc_points
from common.utils import save_json

ROC_CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Corn_(maize)___Common_rust_", "Tomato___Late_blight",
               "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    z = np.load(os.path.join(a.run, "test_predictions.npz"), allow_pickle=True)
    y, p = z["labels"], z["probs"]
    pred = p.argmax(1)
    n = p.shape[1]
    healthy = np.array(["healthy" in c.lower() for c in PV_CLASSES[:n]])
    cm = confusion(y, pred, n)
    off = [(int(cm[i, j]), PV_CLASSES[i], PV_CLASSES[j]) for i in range(n) for j in range(n) if i != j and cm[i, j]]
    off.sort(reverse=True)
    roc_idx = [PV_CLASSES.index(c) for c in ROC_CLASSES]
    res = dict(overall=classification_metrics(y, p), auc=auc_ovr(y, p, roc_idx),
               per_class=per_class_metrics(y, p, n), top_confusions=off[:20],
               diseased_pred_healthy=int((~healthy[y] & healthy[pred]).sum()), diseased_total=int((~healthy[y]).sum()),
               healthy_pred_diseased=int((healthy[y] & ~healthy[pred]).sum()), healthy_total=int(healthy[y].sum()))
    save_json(res, os.path.join(a.run, "analysis.json"))
    np.savetxt(os.path.join(a.run, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    plt.figure(figsize=(5, 4.5))
    for c in roc_idx:
        fpr, tpr = roc_points(y, p, c)
        plt.plot(fpr, tpr, label=f"{PV_CLASSES[c].split('___')[1][:18]} (AUC {res['auc'][c]:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate"), plt.legend(fontsize=7)
    plt.tight_layout(), plt.savefig(os.path.join(a.run, "roc.png"), dpi=200)
    f1 = [r["f1"] for r in res["per_class"]]
    plt.figure(figsize=(12, 3.5))
    plt.bar(range(1, n + 1), f1, color=["tab:orange" if v < 99.5 else "tab:blue" for v in f1])
    plt.axhline(99.5, ls=":", c="k")
    plt.ylim(min(f1) - 1, 100.3), plt.xlabel("Class index"), plt.ylabel("F1 (%)")
    plt.tight_layout(), plt.savefig(os.path.join(a.run, "perclass_f1.png"), dpi=200)
    print(res["overall"], "\nAUC", res["auc"])


if __name__ == "__main__":
    main()
