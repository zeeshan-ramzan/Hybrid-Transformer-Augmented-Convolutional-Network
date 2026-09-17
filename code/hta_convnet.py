"""
hta_convnet.py -- PROPOSED ARCHITECTURE (complete, single file)
===============================================================
Hybrid Transformer-Augmented Convolutional Network (HTA-ConvNet) for plant
disease recognition from leaf images, built around the Dual-Branch Spatial
Attention (DBSA) block.

This one file contains the *entire* proposed method:

    Eq. (1)   F_conv = sigma(BN(DWSConv_3x3(X)))            -> DBSA.conv_branch
    Eq. (2)   F_attn = MHSA(LN(Flatten(X)))                  -> DBSA.attn_branch
    Eq. (3)   Y = X + G o F_conv + (1-G) o F_attn,
              G = sigma(FC([F_conv; F_attn]))                -> DBSA.gate / DBSA.forward
    Eq. (4)   MHSA with k = 8 heads, d_k = C/k               -> DBSA.attn
    Eqs.(5-6) DWSC / MHSA FLOP model                         -> dbsa_flops()
    Eq. (7)   weighted label-smoothed cross-entropy          -> nn_utils.WeightedLabelSmoothingCE
    Table 6   EfficientNet-B4 stages + DBSA insertion points -> HTAConvNet.stage_table()
    Table 8   layer-by-layer architecture summary            -> HTAConvNet.architecture_table()
    "Classification Head and Training Protocol"              -> nn_utils.train_classifier
    "Analysis of Gate Activation Patterns"                   -> gate_activation_analysis()
    "Gradient-Weighted Class Activation Mapping"             -> grad_cam()

Running this file trains HTA-ConvNet on the train split, selects the checkpoint
with the lowest validation loss, evaluates it on the held-out test split, and
writes EVERY HTA-ConvNet figure and table of the paper into
<PROJECT_ROOT>/results/hta_convnet/ :

    figures/fig01_representative_samples.png   Fig. 1  one sample per class
    figures/fig02_architecture_pipeline.png    Fig. 2  end-to-end pipeline
    figures/fig03_dbsa_block.png               Fig. 3  DBSA schematic
    figures/fig04_training_accuracy.png        Fig. 4  train/val accuracy
    figures/fig05_per_class_f1.png             Fig. 5  per-class F1
    figures/fig06_training_loss.png            Fig. 6  train/val loss
    figures/fig07_roc_curves.png               Fig. 7  one-vs-rest ROC
    figures/fig09_confusion_matrix.png         "Confusion Matrix Analysis"
    figures/fig10_reliability_diagram.png      "Calibration Analysis"
    figures/fig11_gradcam.png                  "Gradient-Weighted CAM"
    figures/fig12_gate_activation.png          "Analysis of Gate Activation Patterns"
    tables/table02_class_distribution.csv      Table 2
    tables/table03_augmentation.csv            Table 3
    tables/table05_split_statistics.csv        Table 5
    tables/table06_stage_configuration.csv     Table 6
    tables/table07_dbsa_cost.csv               Table 7  (+ the paper's own numbers)
    tables/table08_architecture_summary.csv    Table 8
    tables/table10_hta_row.csv                 Table 10 row for HTA-ConvNet
    tables/table13_complexity.csv              Table 13 row for HTA-ConvNet
    tables/table14_per_class_metrics.csv       Table 14
    tables/gate_activation.csv                 mean gate G per DBSA stage
    tables/training_history.csv                per-epoch loss / accuracy / lr
    hta_convnet_results.json                   every number, machine readable

Dataset location
----------------
    <PROJECT_ROOT>/dataset/     <- put PlantVillage here (README section 3)

Usage
-----
    python hta_convnet.py                    # paper schedule: 50 epochs, 224 px
    python hta_convnet.py --quick            # 1 epoch, 64 px (smoke test)
    python hta_convnet.py --epochs 20 --cpu
    python hta_convnet.py --pretrained       # ImageNet-1k init (needs internet)
    python hta_convnet.py --attn-tokens 1024 # cap MHSA tokens (see README 9.4)
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
import data_utils as D
import nn_utils as N

RESULT_NAME = "hta_convnet"
DISPLAY = "HTA-ConvNet (Ours)"

# =============================================================================
#  PART 1 -- THE DUAL-BRANCH SPATIAL ATTENTION (DBSA) BLOCK
# =============================================================================


def pick_heads(channels: int, requested: int) -> int:
    """Largest divisor of `channels` that does not exceed `requested`.

    Eq. (4) sets d_k = C/k, which requires k | C.  The paper's own channel
    counts (Table 6: 88, 120, 208) are all divisible by 8, but the real
    EfficientNet-B4 widths are not divisible by every k in the Fig. 8 sweep
    {1,2,4,8,16}, so the head count is clipped to the nearest feasible divisor
    and the choice is reported.
    """
    k = max(1, min(int(requested), int(channels)))
    while channels % k:
        k -= 1
    return max(1, k)


class DBSA(nn.Module):
    r"""Dual-Branch Spatial Attention -- the paper's principal contribution.

    For an input feature map X in R^{H x W x C}:

    Branch 1 (DWSC), Eq. (1)
        F_conv = sigma( BN( DWSConv_{3x3}(X) ) )
        A 3x3 depth-wise convolution (one kernel per channel) followed by a 1x1
        point-wise convolution that mixes channels: 9C + C^2 parameters instead
        of the 9C^2 an ordinary 3x3 convolution would need.

    Branch 2 (MHSA), Eq. (2)
        F_attn = MHSA( LN( Flatten(X) ) )
        Each of the n = H*W spatial locations is one token of dimension d = C.
        Layer normalisation precedes attention to stabilise gradients; the
        output is reshaped back to H x W x C.  k = 8 heads, d_k = C/k (Eq. 4).

    Gating fusion and residual connection, Eq. (3)
        G = sigma( FC( [F_conv ; F_attn] ) )
        Y = X + G o F_conv + (1 - G) o F_attn
        The gate is a two-layer bottleneck MLP on the channel-averaged
        concatenation: 2C -> C/2 -> C, i.e. 1.5 C^2 parameters, producing one
        sigmoid weight per channel that lets the network shift the balance
        between local texture and global context per stage and per class.

    `branch` selects the ablation configuration (paper Table 11):
        "both" -- full DBSA block          "conv" -- DWSC branch only
        "attn" -- MHSA branch only         "none" -- identity
    Single-branch modes have no gate, since Eq. (3) needs two branches to fuse.
    """

    def __init__(self, channels: int, n_heads: int = 8, branch: str = "both",
                 attn_max_tokens: int = 0, attn_dropout: float = 0.0):
        super().__init__()
        self.channels = int(channels)
        self.branch = branch
        self.attn_max_tokens = int(attn_max_tokens)
        self.n_heads = pick_heads(channels, n_heads) if branch in ("both", "attn") else 0
        self.requested_heads = int(n_heads)

        if branch in ("both", "conv"):
            self.conv_branch = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_branch = None

        if branch in ("both", "attn"):
            self.norm = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, self.n_heads, batch_first=True,
                                              dropout=attn_dropout)
        else:
            self.norm = self.attn = None

        if branch == "both":
            hidden = max(4, channels // 2)
            self.gate = nn.Sequential(
                nn.Linear(2 * channels, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, channels), nn.Sigmoid())
        else:
            self.gate = None

        self.last_gate: torch.Tensor | None = None   # populated on every forward

    # -- Eq. (2) ------------------------------------------------------------
    def attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        B, Ch, H, W = x.shape
        h, w = H, W
        z = x
        if self.attn_max_tokens and H * W > self.attn_max_tokens:
            # Documented deviation: sub-sample the token grid, attend, upsample
            # back.  Keeps Eq. (2) exact in form while making the quadratic term
            # of Eq. (6) tractable at 224 px.  See README section 9.4.
            r = math.sqrt(self.attn_max_tokens / (H * W))
            h, w = max(1, int(H * r)), max(1, int(W * r))
            z = F.adaptive_avg_pool2d(x, (h, w))
        tokens = z.flatten(2).transpose(1, 2)                   # B, n, C
        tokens = self.norm(tokens)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = out.transpose(1, 2).reshape(B, Ch, h, w)
        if (h, w) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    # -- Eqs. (1)-(3) -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch == "none":
            return x
        if self.branch == "conv":
            self.last_gate = None
            return x + self.conv_branch(x)
        if self.branch == "attn":
            self.last_gate = None
            return x + self.attn_branch(x)

        f_conv = self.conv_branch(x)                            # Eq. (1)
        f_attn = self.attn_branch(x)                            # Eq. (2)
        pooled = torch.cat([f_conv.mean(dim=(2, 3)), f_attn.mean(dim=(2, 3))], dim=1)
        g = self.gate(pooled)                                   # B, C   in (0,1)
        self.last_gate = g.detach()
        g = g.unsqueeze(-1).unsqueeze(-1)
        return x + g * f_conv + (1.0 - g) * f_attn              # Eq. (3)

    # -- Eqs. (5)-(6) -------------------------------------------------------
    def flops(self, h: int, w: int) -> dict:
        """Analytic cost of this block at spatial size h x w, per Eqs. (5)-(6)."""
        n, c = h * w, self.channels
        dwsc = n * c * (9 + c) if self.branch in ("both", "conv") else 0
        if self.branch in ("both", "attn"):
            n_eff = n
            if self.attn_max_tokens and n > self.attn_max_tokens:
                n_eff = self.attn_max_tokens
            mhsa_paper = 4 * n_eff * c * c + 2 * n_eff * n_eff * c
        else:
            mhsa_paper = 0
        gate = (2 * c * (c // 2) + (c // 2) * c) if self.branch == "both" else 0
        return {"dwsc": int(dwsc), "mhsa": int(mhsa_paper), "gate": int(gate),
                "total": int(dwsc + mhsa_paper + gate)}


def dbsa_flops(n_tokens: int, channels: int, heads: int = 8) -> dict:
    """Eqs. (5) and (6) evaluated directly, for the Table 7 comparison.

        FLOPs_DWSC = H*W*C*(9 + C)
        FLOPs_MHSA = 4*n*C^2 + 2*n^2*C
    """
    return {"dwsc": n_tokens * channels * (9 + channels),
            "mhsa": 4 * n_tokens * channels ** 2 + 2 * n_tokens ** 2 * channels}


# =============================================================================
#  PART 2 -- THE MULTI-SCALE BACKBONE WITH DBSA BLOCKS
# =============================================================================

# EfficientNet-B4 (torchvision) `features` indices grouped into the five stages
# the paper's Table 6 names.  Spatial sizes at 224 px input match Table 6
# exactly; the channel widths do not (see README section 9.1).
STAGE_SLICES: dict[int, tuple[int, int]] = {
    1: (0, 2),     # stem conv + MBConv1     -> 112 x 112
    2: (2, 3),     # MBConv stage            ->  56 x  56
    3: (3, 4),     #                         ->  28 x  28
    4: (4, 5),     #                         ->  14 x  14
    5: (5, 9),     # remaining MBConv + head ->   7 x   7
}


class HTAConvNet(nn.Module):
    """EfficientNet-B4 backbone with DBSA blocks after stages 2, 3 and 4.

    Unlike architectures that apply attention only at the penultimate layer,
    DBSA blocks operate at multiple feature resolutions, capturing fine-grained
    local texture and long-range spatial dependencies simultaneously.

    The classification head follows the paper's prose: global average pooling
    over the final 7x7x1792 tensor, dropout p = 0.4, then a fully connected
    softmax layer over the 38 classes.  (Table 8 instead shows a 448-dimensional
    pooled vector -- see README section 9.2.)
    """

    def __init__(self, n_classes: int, dbsa_stages=(2, 3, 4), n_heads: int = 8,
                 branch: str = "both", dropout: float = 0.4, pretrained: bool = False,
                 attn_max_tokens: int = 0, img_size: int = C.IMG_SIZE,
                 verbose: bool = True):
        super().__init__()
        from torchvision import models

        weights = None
        if pretrained:
            try:
                weights = models.get_model_weights("efficientnet_b4").DEFAULT
            except Exception:
                weights = None
        try:
            backbone = models.efficientnet_b4(weights=weights)
            if verbose and weights is not None:
                print("  loaded ImageNet-1k weights for the EfficientNet-B4 backbone")
        except Exception as e:
            if pretrained and verbose:
                print(f"  [!] could not fetch ImageNet weights ({type(e).__name__}: {e}).\n"
                      "      Using RANDOM initialisation -- the paper fine-tunes from "
                      "ImageNet-1k, so absolute accuracy will be well below 99.6%.")
            backbone = models.efficientnet_b4(weights=None)

        feats = backbone.features
        self.stages = nn.ModuleList([nn.Sequential(*[feats[i] for i in range(a, b)])
                                     for a, b in STAGE_SLICES.values()])
        self.stage_ids = list(STAGE_SLICES.keys())
        self.dbsa_stages = tuple(sorted(set(int(s) for s in dbsa_stages)))
        self.branch = branch
        self.img_size = int(img_size)

        widths = self._probe_widths(img_size)
        self.stage_shapes = widths
        self.dbsa = nn.ModuleDict()
        for s in self.dbsa_stages:
            ch = widths[s][0]
            self.dbsa[str(s)] = DBSA(ch, n_heads=n_heads, branch=branch,
                                     attn_max_tokens=attn_max_tokens)

        feat_dim = widths[5][0]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim
        self.n_classes = int(n_classes)
        self._last_feature: torch.Tensor | None = None      # for Grad-CAM

        if verbose:
            self._warn_attention_cost()

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def _probe_widths(self, img_size: int) -> dict[int, tuple[int, int, int]]:
        """{stage: (channels, H, W)} obtained by a dry forward pass."""
        was = self.training
        self.eval()
        x = torch.zeros(1, 3, img_size, img_size)
        out: dict[int, tuple[int, int, int]] = {}
        for sid, stage in zip(self.stage_ids, self.stages):
            x = stage(x)
            out[sid] = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
        self.train(was)
        return out

    def _warn_attention_cost(self) -> None:
        for s in self.dbsa_stages:
            c, h, w = self.stage_shapes[s]
            n = h * w
            if self.branch in ("both", "attn") and n > 2048 and not \
                    self.dbsa[str(s)].attn_max_tokens:
                gb = 4 * n * n * self.dbsa[str(s)].n_heads / 1e9
                print(f"  [!] DBSA at stage {s} attends over n = {n} tokens. The "
                      f"attention matrix alone is ~{gb:.1f} GB per image in fp32.\n"
                      "      Eq. (6)'s quadratic term dominates here; pass "
                      "--attn-tokens 1024 to cap it (README section 9.4).")

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for sid, stage in zip(self.stage_ids, self.stages):
            x = stage(x)
            if str(sid) in self.dbsa:
                x = self.dbsa[str(sid)](x)
        self._last_feature = x
        z = self.pool(x).flatten(1)
        return self.classifier(self.dropout(z))

    def features_with_grad(self, x: torch.Tensor):
        """Forward that retains the final feature map's graph, for Grad-CAM."""
        for sid, stage in zip(self.stage_ids, self.stages):
            x = stage(x)
            if str(sid) in self.dbsa:
                x = self.dbsa[str(sid)](x)
        x.retain_grad()
        z = self.pool(x).flatten(1)
        return self.classifier(self.dropout(z)), x

    def gate_values(self) -> dict[int, np.ndarray]:
        """Per-stage gate vector G from the most recent forward pass."""
        out = {}
        for s in self.dbsa_stages:
            g = self.dbsa[str(s)].last_gate
            if g is not None:
                out[s] = g.float().cpu().numpy()
        return out

    # ------------------------------------------------------------ paper tables
    def stage_table(self) -> list[dict]:
        """Paper Table 6, filled in with the widths this backbone really has."""
        rows = []
        for i, sid in enumerate(self.stage_ids):
            c, h, w = self.stage_shapes[sid]
            paper = C.PAPER_TABLE6[i]
            rows.append({
                "Stage": paper["stage"],
                "Feature Map (this run)": f"{h}x{w}",
                "Channels (this run)": c,
                "DBSA Block": "yes" if sid in self.dbsa_stages else "-",
                "Paper Feature Map": paper["size"],
                "Paper Channels": paper["channels"],
            })
        return rows

    def dbsa_cost_table(self) -> list[dict]:
        """Paper Table 7, recomputed from Eqs. (5)-(6) alongside the printed values."""
        rows = []
        tot_d = tot_m = 0
        for i, s in enumerate(self.dbsa_stages):
            c, h, w = self.stage_shapes[s]
            n = h * w
            eq = dbsa_flops(n, c)
            blk = self.dbsa[str(s)].flops(h, w)
            tot_d += blk["dwsc"]
            tot_m += blk["mhsa"]
            paper = next((p for p in C.PAPER_TABLE7
                          if p["stage"] == f"Stage {s}"), None)
            rows.append({
                "Stage": f"Stage {s}",
                "n_s (this run)": n, "C_s (this run)": c,
                "DWSC (MFLOPs, Eq. 5)": round(eq["dwsc"] / 1e6, 2),
                "MHSA (MFLOPs, Eq. 6)": round(eq["mhsa"] / 1e6, 2),
                "Paper n_s": paper["n_s"] if paper else None,
                "Paper C_s": paper["C_s"] if paper else None,
                "Paper DWSC (MFLOPs)": paper["dwsc_mflops"] if paper else None,
                "Paper MHSA (MFLOPs)": paper["mhsa_mflops"] if paper else None,
            })
        rows.append({"Stage": "Total", "n_s (this run)": "-", "C_s (this run)": "-",
                     "DWSC (MFLOPs, Eq. 5)": round(tot_d / 1e6, 2),
                     "MHSA (MFLOPs, Eq. 6)": round(tot_m / 1e6, 2),
                     "Paper n_s": "-", "Paper C_s": "-",
                     "Paper DWSC (MFLOPs)": 5.1, "Paper MHSA (MFLOPs)": 62.5})
        return rows

    def architecture_table(self, img_size: int) -> list[dict]:
        """Paper Table 8 -- layer-by-layer parameter and MAC budget."""
        rows = []
        for i, (sid, stage) in enumerate(zip(self.stage_ids, self.stages)):
            c, h, w = self.stage_shapes[sid]
            p = sum(q.numel() for q in stage.parameters())
            rows.append({"Layer/Block": f"Stage {sid}", "Size": f"{h}x{w}x{c}",
                         "Params (M)": round(p / 1e6, 4)})
            if str(sid) in self.dbsa:
                blk = self.dbsa[str(sid)]
                pp = sum(q.numel() for q in blk.parameters())
                rows.append({"Layer/Block": f"DBSA @ Stage {sid}",
                             "Size": f"{h}x{w}x{c}", "Params (M)": round(pp / 1e6, 4)})
        rows.append({"Layer/Block": "GAP + Dropout", "Size": f"{self.feat_dim}-dim",
                     "Params (M)": 0.0})
        rows.append({"Layer/Block": "FC + Softmax", "Size": f"{self.n_classes}-class",
                     "Params (M)": round(sum(q.numel() for q in self.classifier.parameters())
                                         / 1e6, 4)})
        total_p = sum(q.numel() for q in self.parameters())
        rows.append({"Layer/Block": "Total", "Size": "",
                     "Params (M)": round(total_p / 1e6, 3)})
        return rows

    def dbsa_param_overhead(self) -> int:
        return int(sum(q.numel() for s in self.dbsa_stages
                       for q in self.dbsa[str(s)].parameters()))


# =============================================================================
#  PART 3 -- INTERPRETABILITY AND ANALYSIS
# =============================================================================


def grad_cam(model: HTAConvNet, images: torch.Tensor, device,
             target: torch.Tensor | None = None) -> np.ndarray:
    """Gradient-weighted Class Activation Mapping on the final DBSA-augmented
    feature map: the gradient of the class score is global-average-pooled to
    weight each channel, the weighted sum is passed through a ReLU so only
    positive contributions remain, and the map is normalised to [0, 1].
    """
    model.eval().to(device)
    x = images.to(device).clone().requires_grad_(True)
    logits, feat = model.features_with_grad(x)
    tgt = logits.argmax(1) if target is None else target.to(device)
    score = logits.gather(1, tgt.view(-1, 1)).sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    g = feat.grad
    if g is None:                                            # pragma: no cover
        return np.zeros((len(images), 1, 1))
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=1)).detach().cpu().numpy()
    out = []
    for m in cam:
        rng = float(np.ptp(m))
        out.append((m - m.min()) / rng if rng > 1e-12 else np.zeros_like(m))
    return np.stack(out)


@torch.no_grad()
def gate_activation_analysis(model: HTAConvNet, loader, device,
                             max_batches: int = 20) -> dict:
    """Mean and std of the gate vector G at each DBSA stage.

    The paper reports a stage-dependent shift -- 0.67 +- 0.08 at Stage 2 (the
    local DWSC branch dominates), 0.51 +- 0.09 at Stage 3, and 0.38 +- 0.07 at
    Stage 4 (global attention dominates) -- as interpretable evidence that early
    features are local and late features are semantic.
    """
    model.eval().to(device)
    acc: dict[int, list[np.ndarray]] = {s: [] for s in model.dbsa_stages}
    if model.branch != "both":
        return {}
    for i, (xb, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(xb.to(device))
        for s, g in model.gate_values().items():
            acc[s].append(g)
    out = {}
    for s, chunks in acc.items():
        if not chunks:
            continue
        g = np.concatenate(chunks, axis=0)         # samples x channels
        per_sample = g.mean(axis=1)
        out[s] = {"mean": float(per_sample.mean()), "std": float(per_sample.std()),
                  "n_samples": int(len(per_sample)),
                  "channel_mean": g.mean(axis=0).tolist()}
    return out


# =============================================================================
#  PART 4 -- FIGURES THAT DRAW THE ARCHITECTURE (Figs. 2 and 3)
# =============================================================================


def _box(ax, x, y, w, h, text, face, fontsize=6.4):
    import matplotlib.patches as mpatches

    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                         linewidth=0.8, edgecolor="#555555",
                                         facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555555"))


def plot_pipeline(model: HTAConvNet, img_size: int, path, n_classes: int) -> None:
    """Paper Fig. 2 -- end-to-end pipeline, annotated with THIS run's shapes."""
    plt = D._plt()
    fig, ax = plt.subplots(figsize=(5.0, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    pal = C.PALETTE
    rows = [("Input image\n%d x %d x 3" % (img_size, img_size), "#cfe3d4")]
    rows += [("Augmentation (Table 3)\nflip / rotate / jitter / blur / erase", "#dbe4f3"),
             ("Normalise  mu=[.485,.456,.406]\nsigma=[.229,.224,.225]", "#dbe4f3")]
    for sid in model.stage_ids:
        c, h, w = model.stage_shapes[sid]
        rows.append((f"Stage {sid} (EfficientNet-B4 MBConv)\n{h} x {w} x {c}", "#f6ddcc"))
        if sid in model.dbsa_stages:
            rows.append((f"DBSA block @ Stage {sid}\nDWSC + MHSA(k={model.dbsa[str(sid)].n_heads}) "
                         f"+ gate", "#fdf1c4"))
    rows += [(f"Global average pooling -> {model.feat_dim}-d", "#dbe4f3"),
             (f"Dropout p = {model.dropout.p}", "#dbe4f3"),
             (f"FC + Softmax -> {n_classes} classes", "#f4c7c3")]
    n = len(rows)
    pitch = 0.96 / n
    h, gap = pitch * 0.60, pitch * 0.40
    y = 0.98 - h
    for i, (text, col) in enumerate(rows):
        _box(ax, 0.10, y, 0.80, h, text, col, fontsize=6.2)
        if i < n - 1:
            _arrow(ax, 0.50, y - 0.004, 0.50, y - gap + 0.004)
        y -= pitch
    ax.set_title("HTA-ConvNet end-to-end pipeline", fontweight="bold", fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    D._saved(path)


def plot_dbsa_schematic(model: HTAConvNet, path) -> None:
    """Paper Fig. 3 -- schematic of the DBSA block."""
    plt = D._plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    k = model.dbsa[str(model.dbsa_stages[0])].n_heads if model.dbsa_stages else 8
    _box(ax, 0.02, 0.44, 0.20, 0.14, r"Input  $\mathbf{X}\in\mathbb{R}^{H\times W\times C}$",
         "#dbe4f3", 7)
    _box(ax, 0.30, 0.70, 0.26, 0.24,
         "Branch 1 (Eq. 1)\nDWSConv 3x3\nBN\nReLU", "#f6ddcc", 6.6)
    _box(ax, 0.30, 0.08, 0.26, 0.24,
         f"Branch 2 (Eq. 2)\nFlatten -> LN\nMHSA (k={k} heads)\nReshape", "#fdf1c4", 6.6)
    _box(ax, 0.62, 0.40, 0.16, 0.22,
         r"Gate (Eq. 3)"  "\n" r"$[\mathbf{F}_{conv};\mathbf{F}_{attn}]$"
         "\nFC -> sigma -> G", "#e6dcf2", 6.4)
    _box(ax, 0.82, 0.40, 0.16, 0.22,
         r"$\mathbf{Y}=\mathbf{X}+\mathbf{G}\odot\mathbf{F}_{conv}$"
         "\n" r"$+(1-\mathbf{G})\odot\mathbf{F}_{attn}$", "#f4c7c3", 6.0)
    _arrow(ax, 0.22, 0.55, 0.30, 0.80)
    _arrow(ax, 0.22, 0.48, 0.30, 0.20)
    _arrow(ax, 0.56, 0.80, 0.66, 0.62)
    _arrow(ax, 0.56, 0.20, 0.66, 0.40)
    _arrow(ax, 0.78, 0.51, 0.82, 0.51)
    ax.annotate("residual skip", xy=(0.90, 0.63), xytext=(0.12, 0.63),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, ls=":", color="#888888"),
                fontsize=6, va="bottom")
    ax.set_title("Dual-Branch Spatial Attention (DBSA) block", fontweight="bold",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    D._saved(path)


def plot_gradcam_panel(model: HTAConvNet, idx, indices, rc, size, classes, path) -> None:
    """Grad-CAM overlays for a handful of test images."""
    from PIL import Image

    if len(indices) == 0:
        return
    tf = D.build_transforms(size, train=False)
    imgs, raws, caps = [], [], []
    for i in indices:
        with Image.open(idx.files[i]) as im:
            rgb = im.convert("RGB").resize((size, size))
        raws.append(np.asarray(rgb, dtype=np.float32) / 255.0)
        imgs.append(tf(rgb))
        caps.append(classes[int(idx.labels[i])])
    x = torch.stack(imgs)
    cams = grad_cam(model, x, rc.device)
    import matplotlib

    jet = matplotlib.colormaps["jet"]
    panels = []
    for raw, cam in zip(raws, cams):
        heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                          .resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        panels.append(0.55 * raw + 0.45 * jet(heat)[..., :3])
    D.plot_heatmap_panels(panels, [c[:24] for c in caps], path,
                          "Grad-CAM on the final DBSA-augmented feature map")


# =============================================================================
#  PART 5 -- MAIN
# =============================================================================


def build_model(n_classes: int, rc, stages=None, branch: str = "both",
                n_heads: int | None = None, attn_tokens: int | None = None,
                verbose: bool = True) -> HTAConvNet:
    """Single entry point used by this script AND by every experiment script."""
    return HTAConvNet(
        n_classes=n_classes,
        dbsa_stages=C.HTA["dbsa_stages"] if stages is None else stages,
        n_heads=C.HTA["n_heads"] if n_heads is None else n_heads,
        branch=branch, dropout=C.HTA["dropout"], pretrained=rc.pretrained,
        attn_max_tokens=(C.HTA["attn_max_tokens"] if attn_tokens is None else attn_tokens),
        img_size=rc.img_size, verbose=verbose)


def train_variant(idx, splits, rc, stages=None, branch: str = "both",
                  n_heads: int | None = None, attn_tokens: int | None = None,
                  epochs: int | None = None, label: str = "", loaders=None,
                  img_size: int | None = None, seed: int | None = None,
                  verbose: bool = True) -> dict:
    """Train one HTA-ConvNet configuration and score it on the test split.

    Shared by `ablation_study.py`, `attention_heads_sensitivity.py`,
    `dbsa_blocks_sensitivity.py`, `resolution_sensitivity.py`,
    `cross_validation.py`, `seed_robustness.py` and `cross_dataset_plantdoc.py`
    so that every experiment differs only in the configuration it names.
    """
    seed_used = rc.seed if seed is None else seed
    D.set_seed(seed_used)
    if loaders is None:
        loaders, size = D.build_loaders(idx, splits, rc)
    else:
        size = img_size or rc.img_size
        # shared loaders keep their own generator; reseed it so that a different
        # training seed really does change the batch order, not just the weights
        g = getattr(loaders["train"], "generator", None)
        if g is not None:
            g.manual_seed(int(seed_used))
    rc2 = rc
    if img_size and img_size != rc.img_size:
        rc2 = D.RunConfig(**{**rc.__dict__, "img_size": img_size})
    model = build_model(idx.n_classes, rc2, stages=stages, branch=branch,
                        n_heads=n_heads, attn_tokens=attn_tokens, verbose=verbose)
    hist = N.train_classifier(model, loaders, rc.device, dict(C.HTA), idx.n_classes,
                              idx.labels[splits["train"]],
                              epochs=epochs or rc.epochs, patience=rc.patience,
                              verbose=verbose, tag=label)
    val = N.evaluate(model, loaders["val"], rc.device, idx.n_classes, return_arrays=True)
    test = N.evaluate(model, loaders["test"], rc.device, idx.n_classes, return_arrays=True)
    return {"model": model, "history": hist, "val": val, "test": test,
            "params": N.count_parameters(model), "img_size": size,
            "label": label or branch}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train and evaluate HTA-ConvNet (the proposed architecture).")
    D.cli_common(ap)
    ap.add_argument("--heads", type=int, default=C.HTA["n_heads"],
                    help="attention heads k in the MHSA branch (paper: 8)")
    ap.add_argument("--stages", type=int, nargs="*", default=None,
                    help="backbone stages that get a DBSA block (paper: 2 3 4)")
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"],
                    help="cap the MHSA token count (0 = faithful full attention)")
    ap.add_argument("--gradcam-images", type=int, default=8)
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"],
                   default_patience=C.HTA["early_stopping_patience"])
    D.banner("HTA-ConvNet -- Hybrid Transformer-Augmented Convolutional Network")
    D.describe_run(rc, DISPLAY)
    D.set_seed(rc.seed)
    out = C.result_dir(RESULT_NAME)

    # ------------------------------------------------------------------ data
    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc)

    counts = idx.counts()
    D.save_table([{"Idx": i + 1, "Class": c, "Images": int(counts[i]),
                   "% of Total": round(100 * counts[i] / max(counts.sum(), 1), 2),
                   "Paper count (real PlantVillage)":
                       C.PLANTVILLAGE_REAL_COUNTS.get(c, "-")}
                  for i, c in enumerate(idx.classes)] +
                 [{"Idx": "Total", "Class": f"{idx.n_classes} classes",
                   "Images": int(counts.sum()), "% of Total": 100.0,
                   "Paper count (real PlantVillage)": C.PAPER_N_IMAGES}],
                 out / "tables" / "table02_class_distribution",
                 "Table 2 -- class distribution of the dataset actually loaded",
                 print_it=False)
    D.save_table([{"Operation": k, "Value": str(v)} for k, v in C.AUGMENT.items()],
                 out / "tables" / "table03_augmentation",
                 "Table 3 -- data augmentation configuration", print_it=False)
    D.save_table(D.split_table(idx, splits), out / "tables" / "table05_split_statistics",
                 "Table 5 -- dataset split statistics")

    D.plot_image_grid([idx.files[np.flatnonzero(idx.labels == c)[0]]
                       for c in range(min(idx.n_classes, 6))],
                      [idx.classes[c] for c in range(min(idx.n_classes, 6))],
                      out / "figures" / "fig01_representative_samples.png",
                      "Fig. 1 -- representative leaf samples")

    # ----------------------------------------------------------------- model
    D.banner("ARCHITECTURE", "-")
    model = build_model(idx.n_classes, rc, stages=args.stages, n_heads=args.heads,
                        attn_tokens=args.attn_tokens)
    n_par = N.count_parameters(model)
    dbsa_par = model.dbsa_param_overhead()
    print(f"  DBSA stages {model.dbsa_stages} | heads requested {args.heads} -> "
          f"used {[model.dbsa[str(s)].n_heads for s in model.dbsa_stages]}")
    print(f"  parameters: {n_par / 1e6:.2f} M total, of which DBSA adds "
          f"{dbsa_par / 1e6:.3f} M ({100 * dbsa_par / max(n_par, 1):.2f}%)")

    D.save_table(model.stage_table(), out / "tables" / "table06_stage_configuration",
                 "Table 6 -- backbone stage configuration and DBSA insertion points")
    D.save_table(model.dbsa_cost_table(), out / "tables" / "table07_dbsa_cost",
                 "Table 7 -- per-stage DBSA cost, Eqs. (5)-(6) vs the printed values")
    D.save_table(model.architecture_table(size),
                 out / "tables" / "table08_architecture_summary",
                 "Table 8 -- architecture summary", print_it=False)

    # -------------------------------------------------------------- training
    D.banner("TRAINING", "-")
    cfg = dict(C.HTA)
    print(f"  Adam lr {cfg['lr']:.0e}, warm-up {cfg['warmup_epochs']} ep from "
          f"{cfg['warmup_start_lr']:.0e}, cosine to {cfg['eta_min']:.0e}, "
          f"wd {cfg['weight_decay']:.0e}, label smoothing {cfg['label_smoothing']}, "
          f"mixup alpha {cfg['mixup_alpha']}")
    hist = N.train_classifier(model, loaders, rc.device, cfg, idx.n_classes,
                              idx.labels[splits["train"]], epochs=rc.epochs,
                              patience=rc.patience, tag=DISPLAY)

    # ------------------------------------------------------------ evaluation
    D.banner("EVALUATION ON THE HELD-OUT TEST SPLIT", "-")
    val = N.evaluate(model, loaders["val"], rc.device, idx.n_classes, return_arrays=True)
    test = N.evaluate(model, loaders["test"], rc.device, idx.n_classes, return_arrays=True)

    macs = N.count_macs(model, size, torch.device("cpu"))
    fps = N.measure_fps(model, size, rc.device)
    D.save_table([{"Method": DISPLAY, "Params (M)": round(n_par / 1e6, 2),
                   "DBSA params (M)": round(dbsa_par / 1e6, 3),
                   "GFLOPs (MACs)": round(macs / 1e9, 3),
                   "Input": f"{size}x{size}",
                   "FPS (this machine)": round(fps, 1),
                   "Train time (s)": round(hist["train_time_s"], 1),
                   "Paper Params (M)": C.PAPER_TABLE13[DISPLAY][0],
                   "Paper GFLOPs": C.PAPER_TABLE13[DISPLAY][1],
                   "Paper FPS (A100)": C.PAPER_TABLE13[DISPLAY][2]}],
                 out / "tables" / "table13_complexity",
                 "Table 13 -- model complexity and inference speed")

    pr = C.PAPER_TABLE10[DISPLAY]
    D.save_table([{"Method": DISPLAY,
                   "Acc (%)": round(test["accuracy"], 2),
                   "Prec (%)": round(test["precision_macro"], 2),
                   "Rec (%)": round(test["recall_macro"], 2),
                   "F1 (%)": round(test["f1_macro"], 2),
                   "Paper Acc (%)": pr[0], "Paper Prec (%)": pr[1],
                   "Paper Rec (%)": pr[2], "Paper F1 (%)": pr[3]}],
                 out / "tables" / "table10_hta_row",
                 "Table 10 -- HTA-ConvNet row (this run vs the paper)")

    per_class = D.per_class_table(test["y_true"], test["y_pred"], idx.classes)
    D.save_table(per_class, out / "tables" / "table14_per_class_metrics",
                 "Table 14 -- per-class precision, recall and F1", print_it=False)

    # ------------------------------------------------- gate activation (Sec.)
    gates = gate_activation_analysis(model, loaders["test"], rc.device)
    if gates:
        paper_g = {2: C.PAPER_MISC["gate_stage2"], 3: C.PAPER_MISC["gate_stage3"],
                   4: C.PAPER_MISC["gate_stage4"]}
        rows = [{"DBSA stage": f"Stage {s}",
                 "Mean gate G": round(v["mean"], 4), "Std": round(v["std"], 4),
                 "DWSC share (%)": round(100 * v["mean"], 1),
                 "MHSA share (%)": round(100 * (1 - v["mean"]), 1),
                 "Paper mean": paper_g.get(s, ("-",))[0],
                 "Paper std": paper_g.get(s, ("-", "-"))[1] if s in paper_g else "-"}
                for s, v in sorted(gates.items())]
        D.save_table(rows, out / "tables" / "gate_activation",
                     "Gate activation by DBSA stage (local DWSC vs global MHSA)")
        D.plot_bars([f"Stage {s}" for s in sorted(gates)],
                    [gates[s]["mean"] for s in sorted(gates)],
                    out / "figures" / "fig12_gate_activation.png",
                    "Mean DBSA gate value G per stage\n(high = local DWSC branch dominates)",
                    "Mean gate value G", rotate=0)

    # --------------------------------------------------------------- figures
    D.banner("FIGURES", "-")
    plot_pipeline(model, size, out / "figures" / "fig02_architecture_pipeline.png",
                  idx.n_classes)
    plot_dbsa_schematic(model, out / "figures" / "fig03_dbsa_block.png")
    try:
        plot_gradcam_panel(model, idx, splits["test"][:args.gradcam_images], rc, size,
                           idx.classes, out / "figures" / "fig11_gradcam.png")
    except Exception as e:                                   # pragma: no cover
        print(f"  [!] Grad-CAM skipped: {type(e).__name__}: {e}")

    payload = N.report_classification(
        DISPLAY, RESULT_NAME, idx.classes, test, hist, val=val,
        figure_prefix="", paper_row=pr,
        extra={
            "architecture": {
                "backbone": C.HTA["backbone"], "dbsa_stages": list(model.dbsa_stages),
                "heads_used": {str(s): model.dbsa[str(s)].n_heads
                               for s in model.dbsa_stages},
                "stage_shapes": {str(k): list(v) for k, v in model.stage_shapes.items()},
                "feature_dim": model.feat_dim, "dropout": C.HTA["dropout"],
                "attn_max_tokens": args.attn_tokens,
                "params_total": n_par, "params_dbsa": dbsa_par},
            "complexity": {"params": n_par, "macs": macs, "gflops": macs / 1e9,
                           "fps_measured": fps, "input_size": size},
            "gate_activation": gates,
            "dbsa_cost_eqs": model.dbsa_cost_table(),
            "run": {"epochs": rc.epochs, "img_size": size, "batch_size": rc.batch_size,
                    "split": rc.split, "seed": rc.seed, "quick": rc.quick,
                    "pretrained": rc.pretrained, "device": str(rc.device),
                    "hyperparameters": C.HTA},
            "dataset": {"root": str(idx.root), "n_images": len(idx.files),
                        "n_classes": idx.n_classes,
                        "split_sizes": {k: int(len(v)) for k, v in splits.items()}},
            "paper_reference": {"table10": pr, "table13": C.PAPER_TABLE13[DISPLAY],
                                "ece": C.PAPER_MISC["ece_hta"]},
        })

    # rename the shared figure names to the paper's figure numbers
    for src, dst in (("training_accuracy.png", "fig04_training_accuracy.png"),
                     ("per_class_f1.png", "fig05_per_class_f1.png"),
                     ("training_loss.png", "fig06_training_loss.png"),
                     ("roc_curves.png", "fig07_roc_curves.png"),
                     ("confusion_matrix.png", "fig09_confusion_matrix.png"),
                     ("reliability_diagram.png", "fig10_reliability_diagram.png")):
        s, d = out / "figures" / src, out / "figures" / dst
        if s.exists():
            s.replace(d)

    D.banner("DONE")
    print(f"{DISPLAY}: accuracy {test['accuracy']:.2f}%  macro-F1 {test['f1_macro']:.2f}%  "
          f"ECE {test.get('ece', float('nan')):.4f}")
    print(f"Paper reports {pr[0]:.2f}% / {pr[3]:.2f}% on the full 38-class "
          f"PlantVillage benchmark with ImageNet pre-training.")
    if idx.synthetic or rc.quick:
        print("\n[!] This run used a reduced schedule and/or SIMULATED data. "
              "The numbers above are NOT a reproduction of the published results.")
    print(f"All figures and tables -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
