"""[NEW in revision, R3.3] AdaBN unsupervised domain adaptation (Li et al., Pattern Recognit. 2018).

Re-estimates the batch-normalisation statistics of a PlantVillage-trained model on UNLABELLED target images
(adaptation pool), then evaluates on the held-out target test images (Table 'adapt'). Models without
BatchNorm (e.g. Swin/ConvNeXt use LayerNorm) are unchanged by AdaBN.

Usage: python scripts/55_adapt_adabn.py --runs runs/hta_convnet runs/efficientnet_b4 \
          --dataset plantdoc --target-root data/PlantDoc-Dataset
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptation import target_splits
from common.data import PV_CLASSES, simple_loader
from common.loading import load_run
from common.metrics import classification_metrics
from common.trainer import predict
from common.utils import get_device, save_json, seed_everything


@torch.no_grad()
def adabn(model, loader, device):
    m = copy.deepcopy(model).to(device)
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average over all target batches
    m.eval()
    for bn in bns:
        bn.train()
    for x, _ in loader:
        m(x.to(device))
    m.eval()
    return m, len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", choices=["plantdoc", "pp2020"], required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seed_everything(a.seed)
    dev = get_device()
    pool, test = target_splits(a.dataset, a.target_root, PV_CLASSES, a.seed)
    res = {}
    for run in a.runs:
        model, info = load_run(run, len(PV_CLASSES), dev)
        p0, y = predict(model, simple_loader(test, workers=a.workers, img_size=a.img_size), dev)
        adapted, nbn = adabn(model, simple_loader(pool, batch_size=a.batch_size, workers=a.workers, img_size=a.img_size), dev)
        p1, _ = predict(adapted, simple_loader(test, workers=a.workers, img_size=a.img_size), dev)
        res[info["model"]] = dict(zero_shot=classification_metrics(y, p0), adabn=classification_metrics(y, p1),
                                  bn_layers=nbn, n_pool=len(pool), n_test=len(test))
        print(f"{info['model']}: zero-shot {res[info['model']]['zero_shot']['acc']:.2f} -> "
              f"AdaBN {res[info['model']]['adabn']['acc']:.2f} ({nbn} BN layers)")
    save_json(res, a.out or f"runs/adapt_adabn_{a.dataset}.json")


if __name__ == "__main__":
    main()
