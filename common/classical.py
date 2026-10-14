"""Runner for the classical feature-based baselines (fit on train, tune on val, report on test)."""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from common.data import PV_CLASSES, load_plantdoc, load_pp2020, load_split, pv_classes
from common.handcrafted import read
from common.metrics import classification_metrics
from common.utils import save_json


def classical_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-root", default="data/PlantVillage/color")
    p.add_argument("--split-csv", default="splits/standard_split.csv")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--plantdoc-root", default="", help="optional: also evaluate zero-shot on PlantDoc test")
    p.add_argument("--pp2020-root", default="", help="optional: also evaluate zero-shot on Plant Pathology 2020")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-split", type=int, default=0, help="debug: subsample each split")
    return p


def extract(paths, fn, jobs=-1):
    return np.stack(Parallel(n_jobs=jobs, batch_size=64)(delayed(lambda p: fn(read(p)))(p) for p in paths))


def one_hot_probs(pred, n):
    p = np.zeros((len(pred), n))
    p[np.arange(len(pred)), pred] = 1.0
    return p


def grid_search(make_clf, grid, Xtr, ytr, Xva, yva):
    best, best_acc = None, -1
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        clf = make_clf(**params).fit(Xtr, ytr)
        acc = (clf.predict(Xva) == yva).mean()
        print(f"   {params} val acc {100 * acc:.2f}")
        if acc > best_acc:
            best, best_acc = params, acc
    return best, best_acc


def run_classical(args, name, feature_fn, make_clf, grid):
    t0 = time.time()
    classes = pv_classes(args.data_root)
    df = load_split(args.data_root, args.split_csv, args.seed)
    if args.max_per_split:
        df = df.sample(frac=1.0, random_state=0).groupby("split").head(args.max_per_split)
    sets = {s: df[df.split == s] for s in ("train", "val", "test")}
    X = {s: extract(d.path.tolist(), feature_fn, args.jobs) for s, d in sets.items()}
    y = {s: d.label.values for s, d in sets.items()}
    sc = StandardScaler().fit(X["train"])
    X = {s: sc.transform(v) for s, v in X.items()}
    params, vacc = grid_search(make_clf, grid, X["train"], y["train"], X["val"], y["val"])
    clf = make_clf(**params).fit(np.vstack([X["train"]]), y["train"])
    pred = clf.predict(X["test"])
    res = dict(model=name, best_params=params, val_acc=100 * vacc,
               test=classification_metrics(y["test"], one_hot_probs(pred, len(classes))))
    for key, root in (("plantdoc", args.plantdoc_root), ("pp2020", args.pp2020_root)):
        if root:
            fd = load_plantdoc(root, "test", classes) if key == "plantdoc" else load_pp2020(root, classes)
            Xf = sc.transform(extract(fd.path.tolist(), feature_fn, args.jobs))
            res[key] = classification_metrics(fd.label.values, one_hot_probs(clf.predict(Xf), len(classes)))
    res["minutes"] = (time.time() - t0) / 60
    out = os.path.join(args.out_dir, name)
    save_json(res, os.path.join(out, "results.json"))
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), probs=one_hot_probs(pred, len(classes)),
                        labels=y["test"])
    print(f"[{name}] TEST acc {res['test']['acc']:.2f}  F1 {res['test']['f1']:.2f}  params {params}")
    return res
