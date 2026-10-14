"""Shared training / evaluation loop (identical protocol for HTA-ConvNet and every re-trained baseline).

Adam (b1=0.9, b2=0.999), weight decay 1e-5 (not on biases / norm parameters), 5-epoch linear warm-up from
1e-6, cosine annealing to 5e-6 at epoch 50, batch 32, label-smoothed class-weighted CE, mixup (alpha=0.2),
weighted sampling, AMP, early stopping (patience 10) on validation loss, best-val-loss checkpoint.
"""
import copy
import math
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import PV_CLASSES, class_counts, load_split, make_loaders, pv_classes
from common.losses import SmoothWeightedCE, class_weights, mixup
from common.metrics import classification_metrics, per_class_metrics
from common.utils import get_device, save_json, seed_everything


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total_steps, warm_steps, base_lr, min_lr, warm_start=1e-6):
    if warm_steps > 0 and step < warm_steps:
        return warm_start + (base_lr - warm_start) * step / warm_steps
    t = (step - warm_steps) / max(1, total_steps - warm_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def predict(model, loader, device, amp=True, return_logits=False):
    model.eval()
    probs, labels, logits_all = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            lo = model(x)
        lo = lo.float()
        probs.append(F.softmax(lo, 1).cpu())
        logits_all.append(lo.cpu())
        labels.append(torch.as_tensor(y))
    p, y = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    if return_logits:
        return p, y, torch.cat(logits_all).numpy()
    return p, y


@torch.no_grad()
def val_loss(model, loader, device, amp=True):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            lo = model(x)
        tot += F.cross_entropy(lo.float(), y, reduction="sum").item()
        n += len(y)
    return tot / max(n, 1)


def train_one(model, dl_tr, dl_va, counts, args, device, log_prefix="", nan_check=True):
    """Trains with the manuscript protocol, returns (best_state_dict, history, diverged_flag)."""
    n_classes = len(counts)
    model.to(device)
    crit = SmoothWeightedCE(class_weights(counts).to(device), args.label_smoothing)
    opt = torch.optim.Adam(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps_per_epoch = len(dl_tr) if not args.max_train_batches else min(len(dl_tr), args.max_train_batches)
    total = steps_per_epoch * args.epochs
    warm = 0 if args.no_warmup else steps_per_epoch * args.warmup_epochs
    best, best_state, bad, hist, step = float("inf"), None, 0, [], 0
    for ep in range(args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(dl_tr):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, args.lr, args.min_lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, t = mixup(x, y, n_classes, args.mixup_alpha)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), t)
            if nan_check and not torch.isfinite(loss):
                print(f"{log_prefix} non-finite loss at epoch {ep + 1}, step {bi} -> diverged")
                return best_state, hist, True
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
            step += 1
        vl = val_loss(model, dl_va, device, amp)
        hist.append(dict(epoch=ep + 1, train_loss=run_loss / max(seen, 1), val_loss=vl,
                         lr=opt.param_groups[0]["lr"], sec=time.time() - t0))
        print(f"{log_prefix} epoch {ep + 1:3d}/{args.epochs} train {hist[-1]['train_loss']:.4f} "
              f"val {vl:.4f} ({hist[-1]['sec']:.0f}s)", flush=True)
        if vl < best:
            best, bad, best_state = vl, 0, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            hist[-1]["best"] = True
        else:
            bad += 1
            if bad >= args.patience:
                print(f"{log_prefix} early stopping")
                break
    return best_state, hist, False


def run_experiment(model_fn: Callable[[], nn.Module], args, name: str, extra: Optional[Dict] = None):
    """Full pipeline: split -> (optional LR selection on validation loss) -> train -> test metrics -> save."""
    seed_everything(args.seed)
    device = get_device()
    classes = pv_classes(args.data_root)
    df = load_split(args.data_root, args.split_csv, args.seed)
    dl_tr, dl_va, dl_te, tr = make_loaders(df, len(classes), args.img_size, args.batch_size, args.workers,
                                           seed=args.seed)
    counts = class_counts(tr, len(classes))
    out_dir = os.path.join(args.out_dir, name + (f"_seed{args.seed}" if args.seed != 42 else ""))
    os.makedirs(out_dir, exist_ok=True)

    grid = [float(v) for v in args.lr_grid.split(",")] if args.lr_grid else [args.lr]
    results = []
    for lr in grid:
        seed_everything(args.seed)
        args.lr = lr
        model = model_fn()
        state, hist, diverged = train_one(model, dl_tr, dl_va, counts, args, device, f"[{name} lr={lr:g}]")
        best_vl = min([h["val_loss"] for h in hist], default=float("inf"))
        results.append(dict(lr=lr, best_val_loss=best_vl, hist=hist, state=state, diverged=diverged))
    best = min(results, key=lambda r: r["best_val_loss"])
    model = model_fn()
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.to(device)
    torch.save(best["state"], os.path.join(out_dir, "best.pt"))
    probs, y = predict(model, dl_te, device, not args.no_amp)
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"), probs=probs, labels=y,
                        paths=df[df.split == "test"].path.values)
    m = classification_metrics(y, probs)
    best_epoch = next((h["epoch"] for h in reversed(best["hist"]) if h.get("best")), None)
    summary = dict(model=name, selected_lr=best["lr"], best_epoch=best_epoch, diverged=best["diverged"],
                   params_m=sum(p.numel() for p in model.parameters()) / 1e6, test=m,
                   lr_search=[dict(lr=r["lr"], best_val_loss=r["best_val_loss"], diverged=r["diverged"])
                              for r in results],
                   per_class=per_class_metrics(y, probs, len(classes)), history=best["hist"], **(extra or {}))
    save_json(summary, os.path.join(out_dir, "results.json"))
    print(f"[{name}] TEST acc {m['acc']:.2f}  prec {m['prec']:.2f}  rec {m['rec']:.2f}  F1 {m['f1']:.2f}  "
          f"ECE {m['ece']:.4f}  (lr {best['lr']:g}, best epoch {best_epoch}) -> {out_dir}")
    return model, summary, out_dir
