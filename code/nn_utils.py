"""
nn_utils.py
===========
Shared training / evaluation machinery for every *image classifier* in this
package -- the nine deep baselines of Table 10 and the proposed HTA-ConvNet.

Everything the paper specifies about optimisation lives here, in one place, so
that every method really is trained under "the same split protocol and
evaluation metrics":

    Adam, lr 1e-4                       "Classification Head and Training Protocol"
    linear warm-up 1e-6 -> 1e-4 (5 ep)  "Warm-Up and Learning Rate Scheduling"
    cosine annealing to 5e-6 by ep 50   idem
    weight decay 1e-5, no decay on
      biases and BatchNorm parameters   "Regularization and Generalization Strategy"
    label smoothing eps = 0.1           Eq. (7)
    class weights w_c = 1/sqrt(N_c)     Eq. (7)
    mixup alpha = 0.2 (HTA-ConvNet)     "Regularization and Generalization Strategy"
    batch size 32, 50 epochs, early
      stopping patience 10 on val loss  "Classification Head and Training Protocol"
    AMP                                 "Experimental Environment" (CUDA only here)

No architecture that the paper proposes is defined in this file -- HTA-ConvNet
and its DBSA block live entirely in `hta_convnet.py`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
import data_utils as D


# ============================================================================
#  1. BACKBONE CONSTRUCTION
# ============================================================================


def _weights_enum(tv_name: str):
    from torchvision import models

    try:
        return models.get_model_weights(tv_name).DEFAULT
    except Exception:                                       # pragma: no cover
        return None


def build_torchvision_model(key: str, n_classes: int, img_size: int,
                            pretrained: bool = False, verbose: bool = True):
    """Instantiate one of the Table 10 / Table 13 baseline backbones.

    `pretrained=True` follows the paper (ImageNet-1k initialisation) but needs
    network access to torchvision's weight server; when that fails the model is
    randomly initialised and a loud warning is printed, because ImageNet
    pre-training is worth several accuracy points on PlantVillage.
    """
    from torchvision import models

    spec = C.DEEP_BASELINES[key]
    tv = spec["tv"]
    kwargs: dict = {}
    weights = None
    if pretrained:
        try:
            weights = _weights_enum(tv)
        except Exception:
            weights = None

    if tv == "inception_v3":
        kwargs.update(aux_logits=False, init_weights=weights is None,
                      transform_input=False)
    if tv == "vit_b_16":
        # ViT needs a fixed input size; pretrained weights only exist for 224.
        kwargs.update(image_size=int(img_size))
        if weights is not None and int(img_size) != 224:
            if verbose:
                print(f"  [!] ViT-B/16 ImageNet weights only exist for 224 px; "
                      f"running at {img_size} px with random initialisation.")
            weights = None

    def _construct(w):
        if tv == "inception_v3" and w is not None:
            kwargs["init_weights"] = False
        return getattr(models, tv)(weights=w, **kwargs)

    try:
        model = _construct(weights)
        if verbose and weights is not None:
            print(f"  loaded ImageNet-1k weights for {spec['display']}")
    except Exception as e:
        if pretrained and verbose:
            print(f"  [!] could not fetch ImageNet weights for {spec['display']} "
                  f"({type(e).__name__}: {e}).\n"
                  "      Falling back to RANDOM initialisation -- the paper "
                  "fine-tunes from ImageNet, so accuracy will be lower.")
        kwargs.pop("init_weights", None)
        if tv == "inception_v3":
            kwargs["init_weights"] = True
        model = _construct(None)

    _replace_head(model, tv, n_classes)
    return model, int(spec["min_size"])


def _replace_head(model, tv: str, n_classes: int) -> None:
    if tv in ("alexnet", "vgg16"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    elif tv in ("resnet50", "inception_v3"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif tv == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif tv in ("mobilenet_v2", "efficientnet_b4"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    elif tv == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
    else:                                                   # pragma: no cover
        raise ValueError(f"unknown torchvision model {tv}")


def unwrap_logits(out):
    """InceptionV3 returns a namedtuple when aux heads are on."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ============================================================================
#  2. COMPLEXITY  (paper Table 13)
# ============================================================================


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def count_macs(model, img_size: int, device=None, in_ch: int = 3) -> int:
    """Multiply-accumulate operations for one forward pass, by layer hooks.

    Conv2d, Linear and MultiheadAttention are counted; everything else (BN,
    activations, pooling) is negligible and ignored, which is the convention the
    "GFLOPs" columns of papers such as this one follow.
    """
    device = device or torch.device("cpu")
    total = [0]
    hooks = []

    def conv_hook(m, i, o):
        total[0] += int(o.numel() * m.in_channels // m.groups *
                        m.kernel_size[0] * m.kernel_size[1])

    def lin_hook(m, i, o):
        total[0] += int(np.prod(o.shape) * m.in_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    was = model.training
    model.eval().to(device)
    try:
        model(torch.zeros(1, in_ch, img_size, img_size, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return int(total[0])


@torch.no_grad()
def measure_fps(model, img_size: int, device, batch: int = 8, iters: int = 4) -> float:
    """Images per second at inference, measured on THIS machine (paper: A100)."""
    model.eval().to(device)
    x = torch.zeros(batch, 3, img_size, img_size, device=device)
    for _ in range(2):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return float(batch * iters / max(dt, 1e-9))


# ============================================================================
#  3. LOSS, OPTIMISER, SCHEDULE
# ============================================================================


class WeightedLabelSmoothingCE(nn.Module):
    r"""Eq. (7):

        L = - sum_c  w_c [ (1-eps) y_c log p_c  +  (eps/C) log p_c ]

    i.e. cross-entropy against the smoothed target distribution, with a
    per-class weight w_c = 1/sqrt(N_c) that amplifies the gradient contribution
    of minority disease classes.

    Note on the paper's notation: as written, the eps/C term is *added* to the
    (1-eps) y_c term rather than distributed over the C-1 incorrect classes, so
    the target masses sum to (1-eps) + eps = 1 only for the true class row.
    Standard label smoothing -- and what is implemented here, via PyTorch's
    `label_smoothing` -- puts (1-eps) + eps/C on the true class and eps/C on
    every other class.  The two differ by a constant factor only.
    """

    def __init__(self, weight: torch.Tensor | None = None, eps: float = 0.1):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.eps = float(eps)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight,
                               label_smoothing=self.eps)


def build_criterion(train_labels: np.ndarray, n_classes: int, device,
                    scheme: str = "inv_sqrt", eps: float = 0.1):
    w = None
    if scheme and scheme != "none":
        w = torch.tensor(D.class_weights(train_labels, n_classes, scheme),
                         dtype=torch.float32, device=device)
    return WeightedLabelSmoothingCE(w, eps).to(device)


def build_optimizer(model, lr: float, weight_decay: float, kind: str = "adam"):
    """Adam with weight decay applied to weights only (not biases / BN)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.Adam(groups, lr=lr)


def lr_at_epoch(epoch: int, cfg: dict) -> float:
    """Two-phase schedule: linear warm-up then cosine annealing.

    `epoch` is zero-based.  Warm-up runs over the first `warmup_epochs` epochs
    from `warmup_start_lr` to `lr`; cosine annealing then decays to `eta_min`
    at the final epoch.
    """
    import math

    wu = int(cfg.get("warmup_epochs", 0))
    total = int(cfg["epochs"])
    base, start, floor = cfg["lr"], cfg.get("warmup_start_lr", cfg["lr"]), cfg.get("eta_min", 0.0)
    if wu > 0 and epoch < wu:
        return start + (base - start) * (epoch + 1) / wu
    span = max(1, total - wu)
    t = min(1.0, max(0.0, (epoch - wu) / span))
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * t))


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float, generator=None):
    """Mixup (alpha = 0.2) -- returns mixed inputs and the two label sets."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


# ============================================================================
#  4. TRAIN / EVALUATE
# ============================================================================


def train_classifier(model, loaders, device, cfg: dict, n_classes: int,
                     train_labels: np.ndarray, epochs: int | None = None,
                     patience: int | None = None, verbose: bool = True,
                     forward=None, tag: str = "") -> dict:
    """The paper's training protocol.  Returns a history dict; the model is left
    holding the checkpoint with the lowest validation loss.
    """
    cfg = dict(cfg)
    cfg["epochs"] = int(epochs or cfg["epochs"])
    patience = int(patience if patience is not None else cfg.get("early_stopping_patience", 10))
    criterion = build_criterion(train_labels, n_classes, device,
                                cfg.get("class_weighting", "inv_sqrt"),
                                cfg.get("label_smoothing", 0.1))
    opt = build_optimizer(model.to(device), cfg["lr"], cfg["weight_decay"],
                          cfg.get("optimizer", "adam"))
    scaler = D.make_scaler(device)
    fwd = forward or (lambda m, xb: unwrap_logits(m(xb)))
    alpha = float(cfg.get("mixup_alpha", 0.0))
    clip = float(cfg.get("grad_clip", 0.0))

    hist: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                  "lr": [], "epoch_time_s": []}
    best_loss, best_state, bad = np.inf, None, 0
    t_all = time.perf_counter()

    for ep in range(cfg["epochs"]):
        lr = lr_at_epoch(ep, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        tl, tn, tc, t0 = 0.0, 0, 0, time.perf_counter()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            xm, ya, yb2, lam = mixup(xb, yb, alpha)
            opt.zero_grad(set_to_none=True)
            with D.amp_context(device):
                logits = fwd(model, xm)
                loss = (lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
                        if alpha > 0 else criterion(logits, yb))
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            bs = yb.size(0)
            tl += float(loss.detach()) * bs
            tn += bs
            tc += int((logits.detach().float().argmax(1) == yb).sum())
        va = evaluate(model, loaders["val"], device, n_classes, criterion, forward=fwd)
        hist["train_loss"].append(tl / max(tn, 1))
        hist["train_acc"].append(100.0 * tc / max(tn, 1))
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["accuracy"])
        hist["lr"].append(lr)
        hist["epoch_time_s"].append(time.perf_counter() - t0)
        if verbose:
            print(f"    epoch {ep + 1:3d}/{cfg['epochs']}  lr {lr:.2e}  "
                  f"train loss {hist['train_loss'][-1]:.4f} acc {hist['train_acc'][-1]:6.2f}%  "
                  f"| val loss {va['loss']:.4f} acc {va['accuracy']:6.2f}%  "
                  f"[{hist['epoch_time_s'][-1]:.1f}s]", flush=True)
        if va["loss"] < best_loss - 1e-6:
            best_loss, bad = va["loss"], 0
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in model.state_dict().items()})
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stopping at epoch {ep + 1} "
                          f"(no val-loss improvement for {patience} epochs)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    hist["best_val_loss"] = float(best_loss)
    hist["train_time_s"] = time.perf_counter() - t_all
    hist["tag"] = tag
    return hist


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int, criterion=None,
             forward=None, return_arrays: bool = False) -> dict:
    """Forward the whole loader; returns metrics (and optionally raw arrays)."""
    fwd = forward or (lambda m, xb: unwrap_logits(m(xb)))
    model.eval().to(device)
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with D.amp_context(device):
            logits = fwd(model, xb)
        logits = logits.float()
        if criterion is not None:
            loss_sum += float(criterion(logits, yb)) * yb.size(0)
        n += yb.size(0)
        logits_all.append(logits.cpu())
        y_all.append(yb.cpu())
    logits_t = torch.cat(logits_all) if logits_all else torch.zeros(0, n_classes)
    y = torch.cat(y_all).numpy() if y_all else np.zeros(0, dtype=np.int64)
    probs = torch.softmax(logits_t, dim=1).numpy() if len(logits_t) else np.zeros((0, n_classes))
    pred = probs.argmax(1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = D.classification_metrics(y, pred, probs if len(probs) else None, n_classes)
    m["loss"] = loss_sum / max(n, 1)
    if return_arrays:
        m["y_true"], m["y_pred"], m["probs"] = y, pred, probs
    return m


# ============================================================================
#  5. STANDARD RESULT BUNDLE
# ============================================================================


def report_classification(display: str, result_name: str, classes: list[str],
                          test: dict, history: dict | None = None,
                          extra: dict | None = None, val: dict | None = None,
                          figure_prefix: str = "", paper_row=None) -> dict:
    """Write the standard bundle every algorithm in this package produces:

        tables/overall_performance.{csv,md}   Acc / Prec / Rec / F1 / AUC / ECE
        tables/per_class_metrics.{csv,md}     Table 14 style per-class breakdown
        tables/top_confusions.{csv,md}        largest off-diagonal rates
        tables/training_history.{csv,md}      per-epoch loss / accuracy / lr
        figures/<p>training_accuracy.png      Fig. 4
        figures/<p>training_loss.png          Fig. 6
        figures/<p>per_class_f1.png           Fig. 5
        figures/<p>confusion_matrix.png       "Confusion Matrix Analysis"
        figures/<p>roc_curves.png             Fig. 7
        figures/<p>reliability_diagram.png    "Calibration Analysis"
        <result_name>_results.json
    """
    out = C.result_dir(result_name)
    p = figure_prefix
    y, yhat, probs = test["y_true"], test["y_pred"], test["probs"]
    n_classes = len(classes)

    D.banner(f"RESULTS -- {display}")
    row = {"Method": display,
           "Acc (%)": round(test["accuracy"], 2),
           "Prec (%)": round(test["precision_macro"], 2),
           "Rec (%)": round(test["recall_macro"], 2),
           "F1 (%)": round(test["f1_macro"], 2),
           "AUC (macro)": round(test.get("auc_macro", float("nan")), 4),
           "ECE": round(test.get("ece", float("nan")), 4),
           "Test images": int(test["n"])}
    if paper_row is not None:
        row["Paper Acc (%)"] = paper_row[0]
        row["Paper F1 (%)"] = paper_row[3] if len(paper_row) > 3 else paper_row[1]
    D.save_table([row], out / "tables" / "overall_performance",
                 f"Test-set performance -- {display}")

    D.save_table(D.per_class_table(y, yhat, classes),
                 out / "tables" / "per_class_metrics",
                 f"Per-class precision / recall / F1 -- {display}", print_it=False)

    cm = D.confusion(y, yhat, n_classes, normalise=True)
    conf_rows = D.top_confusions(cm, classes, k=10)
    if conf_rows:
        D.save_table(conf_rows, out / "tables" / "top_confusions",
                     "Largest off-diagonal confusion rates", print_it=False)

    if history:
        n_ep = len(history.get("train_loss", []))
        if n_ep:
            D.save_table([{"Epoch": i + 1,
                           "Train loss": round(history["train_loss"][i], 5),
                           "Val loss": round(history["val_loss"][i], 5),
                           "Train acc (%)": round(history["train_acc"][i], 3),
                           "Val acc (%)": round(history["val_acc"][i], 3),
                           "LR": history["lr"][i],
                           "Time (s)": round(history["epoch_time_s"][i], 2)}
                          for i in range(n_ep)],
                        out / "tables" / "training_history", print_it=False)
            D.plot_curves({"train": history["train_acc"], "val": history["val_acc"]},
                          out / "figures" / f"{p}training_accuracy.png",
                          f"{display} training & validation accuracy", "Accuracy (%)")
            D.plot_curves({"train": history["train_loss"], "val": history["val_loss"]},
                          out / "figures" / f"{p}training_loss.png",
                          f"{display} training & validation loss",
                          "Cross-entropy loss",
                          labels=("Training loss", "Validation loss"))

    from sklearn.metrics import precision_recall_fscore_support

    _, _, f1c, _ = precision_recall_fscore_support(y, yhat, labels=list(range(n_classes)),
                                                   zero_division=0)
    D.plot_per_class_f1(f1c * 100, out / "figures" / f"{p}per_class_f1.png",
                        f"Per-class F1-scores ({n_classes} classes) -- {display}",
                        threshold=99.5 if float(np.mean(f1c)) > 0.98 else
                        round(float(np.mean(f1c)) * 100, 1),
                        classes=classes)
    D.plot_confusion(cm, out / "figures" / f"{p}confusion_matrix.png",
                     f"Normalised confusion matrix -- {display}", classes)
    auc_rows = D.plot_roc(probs, y, classes, out / "figures" / f"{p}roc_curves.png",
                          f"ROC curves -- representative classes ({display})")
    if auc_rows:
        D.save_table(auc_rows, out / "tables" / "roc_auc", print_it=False)
    D.plot_reliability(probs, y, out / "figures" / f"{p}reliability_diagram.png",
                       f"Reliability diagram -- {display} "
                       f"(ECE = {test.get('ece', float('nan')):.4f})")

    payload = {"method": display, "result_name": result_name, "classes": classes,
               "test": {k: v for k, v in test.items()
                        if k not in ("y_true", "y_pred", "probs")},
               "validation": ({k: v for k, v in val.items()
                               if k not in ("y_true", "y_pred", "probs")}
                              if val else None),
               "per_class_f1": (f1c * 100).tolist(),
               "history": history or {}}
    if extra:
        payload.update(extra)
    D.save_json(payload, out / f"{result_name}_results.json")
    D.save_predictions(display, y, yhat, probs)

    print(f"\n{display}:  Acc {test['accuracy']:.2f}%   macro-F1 {test['f1_macro']:.2f}%   "
          f"macro-P {test['precision_macro']:.2f}%   macro-R {test['recall_macro']:.2f}%")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")
    return payload


def run_baseline(key: str, args, result_name: str | None = None,
                 display: str | None = None) -> dict:
    """One complete standalone baseline run: load, split, train, test, report.

    Used by every `baseline_*.py` deep-network script so that they differ only in
    the backbone they name.
    """
    spec = C.DEEP_BASELINES[key]
    display = display or spec["display"]
    result_name = result_name or key
    rc = D.resolve(args, default_epochs=C.BASELINE_TRAIN["epochs"],
                   default_patience=C.BASELINE_TRAIN["early_stopping_patience"])
    D.banner(f"{display} -- PlantVillage classification baseline (paper Table 10)")
    D.describe_run(rc, display)
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc, min_size=spec["min_size"])
    if key == "vit_b16" and size % 16:
        size = int(round(size / 16)) * 16
        loaders, size = D.build_loaders(idx, splits, rc, min_size=size)
        print(f"  [note] ViT-B/16 needs a multiple of 16; using {size} px")
    if size != rc.img_size:
        print(f"  [note] {display} needs at least {spec['min_size']} px; "
              f"running at {size} px")

    cfg = dict(C.BASELINE_TRAIN)
    model, _ = build_torchvision_model(key, idx.n_classes, size, rc.pretrained)
    if key == "plantdnet":
        model = PlantDNetHead(model, idx.n_classes)
    n_par = count_parameters(model)
    print(f"  parameters: {n_par / 1e6:.2f} M")

    print(f"\n  training {display} for up to {rc.epochs} epochs ...")
    hist = train_classifier(model, loaders, rc.device, cfg, idx.n_classes,
                            idx.labels[splits["train"]], epochs=rc.epochs,
                            patience=rc.patience, tag=display)
    val = evaluate(model, loaders["val"], rc.device, idx.n_classes, return_arrays=True)
    test = evaluate(model, loaders["test"], rc.device, idx.n_classes, return_arrays=True)

    macs = count_macs(model, size, torch.device("cpu"))
    fps = measure_fps(model, size, rc.device)
    D.save_table([{"Method": display, "Params (M)": round(n_par / 1e6, 2),
                   "GFLOPs (MACs)": round(macs / 1e9, 3),
                   "Input": f"{size}x{size}",
                   "FPS (this machine)": round(fps, 1),
                   "Train time (s)": round(hist["train_time_s"], 1),
                   "Paper Params (M)": C.PAPER_TABLE13.get(display, ("-",) * 3)[0],
                   "Paper GFLOPs": C.PAPER_TABLE13.get(display, ("-",) * 3)[1],
                   "Paper FPS (A100)": C.PAPER_TABLE13.get(display, ("-",) * 3)[2]}],
                 C.result_dir(result_name) / "tables" / "complexity",
                 f"Model complexity -- {display} (paper Table 13)")

    return report_classification(
        display, result_name, idx.classes, test, hist, val=val,
        paper_row=C.PAPER_TABLE10.get(display),
        extra={"complexity": {"params": n_par, "macs": macs, "gflops": macs / 1e9,
                              "fps_measured": fps, "input_size": size},
               "run": {"epochs": rc.epochs, "img_size": size, "batch_size": rc.batch_size,
                       "split": rc.split, "seed": rc.seed, "quick": rc.quick,
                       "pretrained": rc.pretrained, "device": str(rc.device)},
               "dataset": {"root": str(idx.root), "n_images": len(idx.files),
                           "n_classes": idx.n_classes,
                           "split_sizes": {k: int(len(v)) for k, v in splits.items()}},
               "paper_reference": C.PAPER_TABLE10.get(display)})


class PlantDNetHead(nn.Module):
    """Reconstruction of the "PlantDNet" row of Table 10.

    The paper cites Thapa et al. [18] for PlantDNet but never defines the
    architecture; [18] itself uses a ResNet-50 trunk.  This reconstruction keeps
    that trunk and adds the multi-level feature fusion that the PlantDiseaseNet
    family is characterised by: global-average-pooled descriptors are taken from
    three depths of the trunk, concatenated, and classified by a two-layer head.
    Treat the resulting numbers as "a ResNet-50-based multi-scale plant disease
    network", not as a faithful re-implementation of an unspecified model.
    """

    def __init__(self, resnet, n_classes: int, hidden: int = 512, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        d = 512 + 1024 + 2048
        self.head = nn.Sequential(nn.BatchNorm1d(d), nn.Dropout(dropout),
                                  nn.Linear(d, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        z = torch.cat([self.pool(f).flatten(1) for f in (f2, f3, f4)], dim=1)
        return self.head(z)
