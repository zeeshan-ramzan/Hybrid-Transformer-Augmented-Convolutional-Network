"""
ablation_study.py -- paper Table 11
===================================
Systematic ablation in which the EfficientNet-B4 backbone is held fixed and the
components of the DBSA block are added one at a time.

    Table 11  Ablation Study on PlantVillage Test Set
    -------------------------------------------------------------
      Backbone only (EfficientNet-B4)          98.92    98.68
      + DWSC branch (Stage 4 only)             99.11    98.89
      + MHSA branch (Stage 4 only)             99.18    98.97
      + DBSA block (Stage 4 only)              99.27    99.09
      + DBSA blocks (Stages 3&4)               99.43    99.26
      + DBSA (Stages 2,3&4) [Full]             99.61    99.58

Each row is a genuinely separate training run of `hta_convnet.HTAConvNet` with
`dbsa_stages` and `branch` set as the row describes -- the single-branch rows
use `branch="conv"` (Eq. 1 only) and `branch="attn"` (Eq. 2 only), which have no
gate because Eq. (3) needs two branches to fuse.  All rows share the identical
split, schedule, seed and augmentation, so the differences are attributable to
the architecture alone.

Outputs -> results/ablation/
    tables/table11_ablation.csv           the reproduced table, with paper values
    tables/ablation_deltas.csv            gain of each row over the previous one
    figures/table11_accuracy.png          accuracy and F1 per configuration
    figures/ablation_learning_curves.png  validation accuracy per epoch, all rows
    ablation_results.json

Usage
-----
    python ablation_study.py
    python ablation_study.py --quick
    python ablation_study.py --epochs 10 --cpu
    python ablation_study.py --only backbone dbsa_s234
"""

from __future__ import annotations

import argparse

import numpy as np

import config as C
import data_utils as D
import hta_convnet as H

RESULT_NAME = "ablation"


def main() -> None:
    ap = argparse.ArgumentParser(description="DBSA ablation study (paper Table 11).")
    D.cli_common(ap)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these configuration keys "
                         f"({', '.join(c['key'] for c in C.ABLATION_CONFIGS)})")
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    D.banner("ABLATION STUDY -- contribution of each DBSA component (Table 11)")
    D.describe_run(rc, "ablation")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc)

    configs = [c for c in C.ABLATION_CONFIGS
               if args.only is None or c["key"] in args.only]
    out = C.result_dir(RESULT_NAME)
    rows, curves, payload = [], {}, {}

    for cfg in configs:
        D.banner(f"{cfg['label']}", "-")
        r = H.train_variant(idx, splits, rc, stages=cfg["stages"],
                            branch=cfg["branch"] if cfg["stages"] else "none",
                            attn_tokens=args.attn_tokens, loaders=loaders,
                            label=cfg["label"])
        t = r["test"]
        paper = C.PAPER_TABLE11[cfg["key"]]
        rows.append({"Configuration": cfg["label"],
                     "DBSA stages": ",".join(map(str, cfg["stages"])) or "-",
                     "Branch": cfg["branch"],
                     "Acc (%)": round(t["accuracy"], 2),
                     "F1 (%)": round(t["f1_macro"], 2),
                     "Params (M)": round(r["params"] / 1e6, 2),
                     "Paper Acc (%)": paper[0], "Paper F1 (%)": paper[1]})
        curves[cfg["label"]] = r["history"]["val_acc"]
        payload[cfg["key"]] = {
            "label": cfg["label"], "stages": list(cfg["stages"]),
            "branch": cfg["branch"], "params": r["params"],
            "test": {k: v for k, v in t.items()
                     if k not in ("y_true", "y_pred", "probs")},
            "history": r["history"], "paper": list(paper)}
        D.save_predictions(cfg["label"], t["y_true"], t["y_pred"], t["probs"],
                           tag="ablation")
        del r

    D.save_table(rows, out / "tables" / "table11_ablation",
                 "Table 11 -- ablation study on the test split")

    deltas = []
    for i in range(1, len(rows)):
        deltas.append({"From": rows[i - 1]["Configuration"],
                       "To": rows[i]["Configuration"],
                       "Delta Acc (pp)": round(rows[i]["Acc (%)"] -
                                               rows[i - 1]["Acc (%)"], 2),
                       "Delta F1 (pp)": round(rows[i]["F1 (%)"] -
                                              rows[i - 1]["F1 (%)"], 2),
                       "Paper Delta Acc (pp)": round(rows[i]["Paper Acc (%)"] -
                                                     rows[i - 1]["Paper Acc (%)"], 2)})
    if deltas:
        D.save_table(deltas, out / "tables" / "ablation_deltas",
                     "Incremental gain of each added component")

    D.plot_grouped_bars([r["Configuration"].replace(" (", "\n(") for r in rows],
                        {"Accuracy (%)": [r["Acc (%)"] for r in rows],
                         "Macro-F1 (%)": [r["F1 (%)"] for r in rows]},
                        out / "figures" / "table11_accuracy.png",
                        "Ablation: cumulative effect of the DBSA components",
                        "Score (%)", rotate=28)
    if curves:
        L = max(len(v) for v in curves.values())
        D.plot_lines(np.arange(1, L + 1),
                     {k: np.pad(np.asarray(v, dtype=float), (0, L - len(v)), mode="edge")
                      for k, v in curves.items()},
                     out / "figures" / "ablation_learning_curves.png",
                     "Validation accuracy per epoch, all ablation configurations",
                     "Epoch", "Validation accuracy (%)", marker="")

    D.save_json({"configurations": payload,
                 "run": {"epochs": rc.epochs, "img_size": size, "seed": rc.seed,
                         "split": rc.split, "quick": rc.quick},
                 "paper_table11": C.PAPER_TABLE11},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
