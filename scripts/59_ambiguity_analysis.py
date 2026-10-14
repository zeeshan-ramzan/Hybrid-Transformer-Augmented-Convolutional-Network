"""[NEW in revision, R3.6] Quantitative multi-lesion ambiguity analysis (Table 'ambig').

Step 1 (--make-template): writes an annotation sheet with every test image of the seven most-confused classes
        (5 tomato + 2 corn diseases; 1,281 images with the standard split). Two annotators fill columns
        'annotator1' and 'annotator2' with 0 = single-symptom, 1 = ambiguous (multi-lesion / lesion + senescence)
        and column 'consensus' after resolving disagreements.
Step 2: computes Cohen's kappa, and per subset the error rate, top-2 accuracy, mean top-1/top-2 margin and
        mean predictive entropy, plus the share of all errors that fall on ambiguous images.

Usage: python scripts/59_ambiguity_analysis.py --run runs/hta_convnet --make-template
       python scripts/59_ambiguity_analysis.py --run runs/hta_convnet --annotations ambiguity_annotations.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES
from common.metrics import entropy
from common.utils import save_json

CONFUSED = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Septoria_leaf_spot", "Tomato___Leaf_Mold",
            "Tomato___Target_Spot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Northern_Leaf_Blight"]


def stats(sub):
    p = np.stack(sub.probs.values)
    y = sub.label.values
    srt = np.sort(p, 1)
    top2 = np.argsort(-p, 1)[:, :2]
    return dict(images=int(len(sub)), error_pct=100 * float((p.argmax(1) != y).mean()),
                top2_acc_pct=100 * float((top2 == y[:, None]).any(1).mean()),
                mean_margin=float((srt[:, -1] - srt[:, -2]).mean()), mean_entropy=float(entropy(p).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing test_predictions.npz")
    ap.add_argument("--annotations", default="ambiguity_annotations.csv")
    ap.add_argument("--make-template", action="store_true")
    a = ap.parse_args()
    z = np.load(os.path.join(a.run, "test_predictions.npz"), allow_pickle=True)
    idx = [PV_CLASSES.index(c) for c in CONFUSED]
    df = pd.DataFrame(dict(path=z["paths"], label=z["labels"], probs=list(z["probs"])))
    df = df[df.label.isin(idx)].reset_index(drop=True)
    if a.make_template:
        t = df[["path", "label"]].assign(class_name=[PV_CLASSES[i] for i in df.label], annotator1="",
                                         annotator2="", consensus="")
        t.to_csv(a.annotations, index=False)
        print(f"annotation sheet with {len(t)} images -> {a.annotations}")
        return
    ann = pd.read_csv(a.annotations)
    df = df.merge(ann[["path", "annotator1", "annotator2", "consensus"]], on="path")
    kappa = cohen_kappa_score(df.annotator1.astype(int), df.annotator2.astype(int))
    amb = df.consensus.astype(int) == 1
    err = np.stack(df.probs.values).argmax(1) != df.label.values
    res = dict(cohen_kappa=float(kappa), single_symptom=stats(df[~amb]), ambiguous=stats(df[amb]),
               all_seven_classes=stats(df),
               share_of_errors_on_ambiguous_pct=100 * float(err[amb].sum() / max(err.sum(), 1)))
    save_json(res, os.path.join(a.run, "ambiguity_analysis.json"))
    print(res)


if __name__ == "__main__":
    main()
