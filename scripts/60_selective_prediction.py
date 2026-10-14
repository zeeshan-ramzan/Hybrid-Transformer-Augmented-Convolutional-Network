"""[NEW in revision, R3.6] Uncertainty-based rejection / selective prediction (Table 'selective').

1. Temperature scaling (Guo et al. 2017) fitted on the PlantVillage validation set.
2. Confidence = maximum softmax probability (Hendrycks & Gimpel 2017) of the temperature-scaled model, or
   predictive entropy of 20 MC-dropout passes (Gal & Ghahramani 2016).
3. Threshold tau chosen on the validation set as the least strict value whose selective risk <= target risk
   (default 0.1%). The same tau is applied to the PlantVillage test set and to PlantDoc (zero-shot).
4. Reports coverage, selective risk, AURC (x1e3) and the fraction of test errors rejected.

Usage: python scripts/60_selective_prediction.py --runs runs/hta_convnet runs/efficientnet_b4 \
          --data-root data/PlantVillage/color --plantdoc-root data/PlantDoc-Dataset
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, load_plantdoc, load_split, simple_loader
from common.loading import load_run
from common.metrics import entropy, risk_coverage
from common.trainer import predict
from common.utils import get_device, save_json, seed_everything


def fit_temperature(logits, labels):
    t = torch.zeros(1, requires_grad=True)  # log T
    lo, y = torch.tensor(logits), torch.tensor(labels)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lo / t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp())


@torch.no_grad()
def mc_dropout_probs(model, loader, dev, passes=20):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    out = None
    for _ in range(passes):
        p, _ = predict(model, loader, dev)
        out = p if out is None else out + p
    model.eval()
    return out / passes


def choose_tau(conf, correct, target_risk):
    """Largest coverage such that the error among accepted (conf >= tau) <= target risk."""
    order = np.argsort(-conf)
    err = np.cumsum(~correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(err <= target_risk)[0]
    if len(ok) == 0:
        return float("inf")
    return float(conf[order][ok.max()])


def evaluate(conf, pred, y, tau):
    acc = conf >= tau
    wrong = pred != y
    _, _, aurc = risk_coverage(y, conf, pred)
    return dict(coverage=100 * float(acc.mean()),
                risk=100 * float(wrong[acc].mean()) if acc.any() else 0.0,
                aurc_x1e3=1e3 * aurc, errors=int(wrong.sum()),
                errors_rejected_pct=100 * float((wrong & ~acc).sum() / max(wrong.sum(), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--plantdoc-root", default="")
    ap.add_argument("--target-risk", type=float, default=0.001)
    ap.add_argument("--mc-passes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    seed_everything(42)
    dev = get_device()
    df = load_split(a.data_root, a.split_csv)
    sets = {"val": df[df.split == "val"], "test": df[df.split == "test"]}
    if a.plantdoc_root:
        sets["plantdoc"] = load_plantdoc(a.plantdoc_root, "test", PV_CLASSES)
    loaders = {k: simple_loader(v, workers=a.workers, img_size=a.img_size) for k, v in sets.items()}
    res = {}
    for run in a.runs:
        model, info = load_run(run, len(PV_CLASSES), dev)
        logits, labels = {}, {}
        for k, dl in loaders.items():
            _, labels[k], logits[k] = predict(model, dl, dev, return_logits=True)
        T = fit_temperature(logits["val"], labels["val"])
        r = dict(temperature=T)
        # (a) temperature-scaled max softmax
        sm = {k: F.softmax(torch.tensor(v) / T, 1).numpy() for k, v in logits.items()}
        conf = {k: v.max(1) for k, v in sm.items()}
        tau = choose_tau(conf["val"], sm["val"].argmax(1) == labels["val"], a.target_risk)
        r["msp"] = dict(tau=tau, **{k: evaluate(conf[k], sm[k].argmax(1), labels[k], tau)
                                    for k in sm if k != "val"})
        # (b) MC-dropout entropy (confidence = -entropy)
        mc = {k: mc_dropout_probs(model, dl, dev, a.mc_passes) for k, dl in loaders.items()}
        cmc = {k: -entropy(v) for k, v in mc.items()}
        tau2 = choose_tau(cmc["val"], mc["val"].argmax(1) == labels["val"], a.target_risk)
        r["mc_dropout_entropy"] = dict(tau=tau2, **{k: evaluate(cmc[k], mc[k].argmax(1), labels[k], tau2)
                                                    for k in mc if k != "val"})
        res[info["model"]] = r
        print(info["model"], {m: {k: v for k, v in r[m].items() if k != "tau"} for m in ("msp", "mc_dropout_entropy")})
    save_json(res, "runs/selective_prediction.json")


if __name__ == "__main__":
    main()
