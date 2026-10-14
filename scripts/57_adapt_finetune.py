"""[NEW in revision, R3.3] Few-shot (k labelled images per class) and full fine-tuning on field data.

Starts from a PlantVillage-trained model; fine-tunes all layers with the manuscript loss, initial LR 1e-5,
at most 20 epochs, early stopping on a validation subset of the target training data; evaluation on the
held-out target test images; mean over seeds (Table 'adapt').

Usage: python scripts/57_adapt_finetune.py --run runs/hta_convnet --dataset plantdoc \
          --target-root data/PlantDoc-Dataset --shots 10      # few-shot
       python scripts/57_adapt_finetune.py --run runs/hta_convnet --dataset plantdoc \
          --target-root data/PlantDoc-Dataset --shots 0       # full fine-tuning
"""
import argparse
import copy
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptation import few_shot, inner_val, target_splits
from common.data import PV_CLASSES, class_counts, simple_loader
from common.loading import load_run
from common.metrics import classification_metrics
from common.trainer import predict, train_one
from common.utils import get_device, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", choices=["plantdoc", "pp2020"], required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--shots", type=int, default=10, help="labelled images per class; 0 = all (full FT)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    dev = get_device()
    base, info = load_run(a.run, len(PV_CLASSES), dev)
    pool, test = target_splits(a.dataset, a.target_root, PV_CLASSES, 42)
    res = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        seed_everything(seed)
        data = few_shot(pool, a.shots, seed) if a.shots > 0 else pool
        tr, va = inner_val(data, 0.15 if a.shots == 0 else 0.2, seed)
        args = SimpleNamespace(lr=a.lr, epochs=a.epochs, weight_decay=1e-5, warmup_epochs=1, min_lr=1e-7,
                               patience=5, label_smoothing=0.1, mixup_alpha=0.2, no_amp=False, no_warmup=False,
                               max_train_batches=0)
        model = copy.deepcopy(base)
        state, hist, _ = train_one(model, simple_loader(tr, batch_size=a.batch_size, workers=a.workers, train=True, img_size=a.img_size),
                                   simple_loader(va, workers=a.workers, img_size=a.img_size), class_counts(tr, len(PV_CLASSES)), args,
                                   dev, f"[FT {info['model']} k={a.shots or 'all'} seed {seed}]")
        if state is not None:
            model.load_state_dict(state)
        probs, y = predict(model.to(dev), simple_loader(test, workers=a.workers, img_size=a.img_size), dev)
        m = classification_metrics(y, probs)
        res.append(dict(seed=seed, n_train=len(tr), **m))
        print(f"   -> test acc {m['acc']:.2f}  F1 {m['f1']:.2f}")
    accs = [r["acc"] for r in res]
    tag = f"fewshot{a.shots}" if a.shots else "full"
    save_json(dict(model=info["model"], dataset=a.dataset, setting=tag, runs=res, mean_acc=float(np.mean(accs)),
                   std_acc=float(np.std(accs))), os.path.join(a.run, f"adapt_finetune_{tag}_{a.dataset}.json"))
    print(f"{tag}: mean acc {np.mean(accs):.2f} +- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
