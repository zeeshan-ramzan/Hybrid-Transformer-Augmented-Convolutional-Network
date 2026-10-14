"""[NEW in revision, R3.5] Empirical stability of the fusion rules (Table 'stability').

For each trained fusion variant (proposed gate, G=0.5 average, plain addition, concat+1x1, gate w/o residual):
  1. median relative update ||Y-X||/||X|| of the Stage-2 block           (Proposition 1, bounded update)
  2. empirical local Lipschitz estimate of the network on the logits:      (Proposition 2)
       max over 10 Gaussian perturbations with ||delta|| = 0.01 ||x|| of ||f(x+delta)-f(x)|| / ||delta||
     -> median and 95th percentile over test images
  3. accuracy under additive Gaussian noise (sigma = 0.1 on normalised inputs)
  4. optional (--divergence-seeds): number of seeds whose training produces NaN loss WITHOUT warm-up
     (trains each variant for --divergence-epochs epochs per seed through 10_hta_convnet.py --no-warmup)

Usage: python scripts/58_fusion_stability.py --runs runs/hta_convnet runs/fusion_avg runs/fusion_add \
          runs/fusion_concat runs/fusion_gate_nores --data-root data/PlantVillage/color [--divergence-seeds 10]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, load_split, simple_loader
from common.loading import load_run
from common.utils import get_device, save_json, seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def analyse(model, loader, dev, n_images=1000, n_pert=10, rel=0.01, sigma=0.1):
    model.eval()
    blk2 = model.blocks["2"] if "2" in model.blocks else None
    ratios, lips, correct, noisy_correct, seen = [], [], 0, 0, 0
    for x, y in loader:
        if seen >= n_images:
            break
        x, y = x.to(dev).float(), y.to(dev)
        x, y = x[: n_images - seen], y[: n_images - seen]
        model.set_record(True)
        f0 = model(x)
        model.set_record(False)
        if blk2 is not None and blk2.last_update_ratio is not None:
            ratios.append(blk2.last_update_ratio.cpu())
        best = torch.zeros(len(x), device=dev)
        xn = x.flatten(1).norm(dim=1)
        for _ in range(n_pert):
            d = torch.randn_like(x)
            d = d / d.flatten(1).norm(dim=1)[:, None, None, None] * (rel * xn)[:, None, None, None]
            r = (model(x + d) - f0).flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1)
            best = torch.maximum(best, r)
        lips.append(best.cpu())
        correct += (f0.argmax(1) == y).sum().item()
        noisy_correct += (model(x + sigma * torch.randn_like(x)).argmax(1) == y).sum().item()
        seen += len(x)
    lips = torch.cat(lips).numpy()
    out = dict(images=seen, clean_acc=100 * correct / seen, noisy_acc=100 * noisy_correct / seen,
               local_lipschitz_median=float(np.median(lips)), local_lipschitz_p95=float(np.percentile(lips, 95)))
    if ratios:
        out["stage2_update_ratio_median"] = float(torch.cat(ratios).median())
    return out


def divergence_test(cfg, seeds, epochs, passthrough):
    diverged = 0
    for s in range(42, 42 + seeds):
        name = f"divtest_{cfg['fusion']}_s{s}"
        cmd = [sys.executable, os.path.join(HERE, "10_hta_convnet.py"), "--name", name, "--fusion", cfg["fusion"],
               "--no-warmup", "--epochs", str(epochs), "--patience", str(epochs), "--seed", str(s),
               "--out-dir", "runs/divergence"] + passthrough
        subprocess.run(cmd, check=True)
        d = f"runs/divergence/{name}" + (f"_seed{s}" if s != 42 else "")
        with open(os.path.join(d, "results.json")) as f:
            diverged += int(json.load(f)["diverged"])
    return diverged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--divergence-seeds", type=int, default=0)
    ap.add_argument("--divergence-epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    a, extra = ap.parse_known_args()   # unknown flags are forwarded to 10_hta_convnet.py
    seed_everything(42)
    dev = get_device()
    df = load_split(a.data_root, a.split_csv)
    test = df[df.split == "test"].sample(frac=1.0, random_state=0)
    loader = simple_loader(test, batch_size=50, workers=a.workers, img_size=a.img_size)
    res = {}
    for run in a.runs:
        model, info = load_run(run, len(PV_CLASSES), dev)
        r = analyse(model, loader, dev, a.n_images)
        cfg = info.get("hta_config", {})
        if a.divergence_seeds:
            r["diverged_without_warmup"] = divergence_test(cfg, a.divergence_seeds, a.divergence_epochs,
                                                           ["--data-root", a.data_root, "--split-csv", a.split_csv,
                                                            "--img-size", str(a.img_size)] + extra)
            r["divergence_seeds"] = a.divergence_seeds
        res[info["model"]] = dict(fusion=cfg.get("fusion"), **r)
        print(info["model"], r)
    save_json(res, "runs/fusion_stability.json")


if __name__ == "__main__":
    main()
