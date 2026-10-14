"""[NEW in revision, R3.3] DANN unsupervised domain adaptation (Ganin et al., JMLR 2016).

Starting from a PlantVillage-trained model, training continues on labelled PlantVillage images (source,
manuscript loss) plus UNLABELLED target images, with a domain classifier on the pooled features behind a
gradient-reversal layer (lambda schedule 2/(1+exp(-10p))-1). Evaluation on the held-out target test images.
Mean over seeds (default 3) is reported (Table 'adapt').

Usage: python scripts/56_adapt_dann.py --run runs/hta_convnet --dataset plantdoc \
          --target-root data/PlantDoc-Dataset --data-root data/PlantVillage/color
"""
import argparse
import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptation import target_splits
from common.data import PV_CLASSES, class_counts, load_split, make_loaders, simple_loader
from common.loading import load_run
from common.losses import SmoothWeightedCE, class_weights
from common.metrics import classification_metrics
from common.trainer import param_groups, predict
from common.utils import get_device, save_json, seed_everything


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class FeatureHook:
    """Captures the input of the final linear classifier (= pooled features)."""

    def __init__(self, model):
        last = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        self.dim = last.in_features
        self.h = last.register_forward_hook(self._hook)
        self.feat = None

    def _hook(self, mod, inp, out):
        self.feat = inp[0]


def cycle(loader):
    while True:
        for b in loader:
            yield b


def run_dann(base_model, src_loader, tgt_loader, counts, dev, epochs, steps, lr, amp=True):
    model = copy.deepcopy(base_model).to(dev).train()
    hook = FeatureHook(model)
    disc = nn.Sequential(nn.Linear(hook.dim, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1024), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(1024, 1)).to(dev)
    crit = SmoothWeightedCE(class_weights(counts).to(dev), 0.1)
    opt = torch.optim.Adam(param_groups(model, 1e-5) + [{"params": disc.parameters(), "weight_decay": 0.0,
                                                          "lr": lr * 10}], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    src, tgt = cycle(src_loader), cycle(tgt_loader)
    total = epochs * steps
    for it in range(total):
        p = it / total
        lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
        xs, ys = next(src)
        xt, _ = next(tgt)
        xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            logits = model(torch.cat([xs, xt]))
            feats = hook.feat
            cls_loss = crit(logits[:len(xs)], ys)
            d = disc(GradReverse.apply(feats.float(), lam)).squeeze(1)
            dom = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt))]).to(dev)
            dom_loss = F.binary_cross_entropy_with_logits(d.float(), dom)
            loss = cls_loss + dom_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if it % steps == 0:
            print(f"   step {it}/{total} cls {cls_loss.item():.3f} dom {dom_loss.item():.3f} lambda {lam:.2f}",
                  flush=True)
    hook.h.remove()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", choices=["plantdoc", "pp2020"], required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    dev = get_device()
    base, info = load_run(a.run, len(PV_CLASSES), dev)
    df = load_split(a.data_root, a.split_csv)
    dl_src, _, _, tr = make_loaders(df, len(PV_CLASSES), img_size=a.img_size, batch_size=a.batch_size, workers=a.workers)
    counts = class_counts(tr, len(PV_CLASSES))
    res = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        seed_everything(seed)
        pool, test = target_splits(a.dataset, a.target_root, PV_CLASSES, 42)
        dl_tgt = simple_loader(pool, batch_size=a.batch_size, workers=a.workers, train=True, img_size=a.img_size)
        model = run_dann(base, dl_src, dl_tgt, counts, dev, a.epochs, a.steps_per_epoch, a.lr)
        probs, y = predict(model, simple_loader(test, workers=a.workers, img_size=a.img_size), dev)
        m = classification_metrics(y, probs)
        res.append(dict(seed=seed, **m))
        print(f"[DANN {info['model']} seed {seed}] {a.dataset} acc {m['acc']:.2f} F1 {m['f1']:.2f}")
    accs = [r["acc"] for r in res]
    save_json(dict(model=info["model"], dataset=a.dataset, runs=res, mean_acc=float(np.mean(accs)),
                   std_acc=float(np.std(accs))), os.path.join(a.run, f"adapt_dann_{a.dataset}.json"))
    print(f"mean acc {np.mean(accs):.2f} +- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
