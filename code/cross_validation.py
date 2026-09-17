"""
cross_validation.py -- paper Table 4
====================================
Stratified 5-fold cross-validation of HTA-ConvNet over the full dataset, run
alongside the primary train/val/test split to check that the headline number is
not an artefact of one favourable partition.

    Table 4  5-Fold Stratified Cross-Validation Results for HTA-ConvNet
    -------------------------------------------------------------------
      Fold 1   99.61   99.58   0.9997        Fold 4   99.52   99.49   0.9995
      Fold 2   99.54   99.51   0.9996        Fold 5   99.58   99.55   0.9997
      Fold 3   99.65   99.62   0.9998        Mean     99.58 +- 0.07

Protocol.  The paper says each fold "trains on 80% of the data and evaluates on
the remaining 20%".  It does not say where the validation data used for early
stopping and checkpoint selection comes from.  Selecting the checkpoint on the
same 20% that is then scored would bias every fold optimistically, so this
implementation carves an inner validation split out of the 80% training portion
(`--inner-val-frac`, default 0.1) and leaves the held-out fold untouched until
scoring.  `--val-on-test` reproduces the optimistic reading instead, and the gap
between the two runs measures the size of that bias.  See README section 9.5.

Outputs -> results/cross_validation/
    tables/table04_cross_validation.csv   per-fold accuracy, F1 and AUC + mean/std
    figures/table04_folds.png             per-fold accuracy and F1
    cross_validation_results.json

Usage
-----
    python cross_validation.py
    python cross_validation.py --quick
    python cross_validation.py --folds 3 --epochs 10
    python cross_validation.py --val-on-test        # the paper's ambiguous reading
"""

from __future__ import annotations

import argparse

import numpy as np

import config as C
import data_utils as D
import hta_convnet as H

RESULT_NAME = "cross_validation"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stratified k-fold cross-validation of HTA-ConvNet (Table 4).")
    D.cli_common(ap)
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--inner-val-frac", type=float, default=0.10,
                    help="fraction of each training fold reserved for early stopping")
    ap.add_argument("--val-on-test", action="store_true",
                    help="use the held-out fold itself for early stopping "
                         "(optimistically biased; the paper is ambiguous)")
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    folds_n = C.QUICK["folds"] if rc.quick else args.folds
    D.banner(f"{folds_n}-FOLD STRATIFIED CROSS-VALIDATION OF HTA-ConvNet (Table 4)")
    D.describe_run(rc, "cross_validation")
    print(f"  early stopping on: "
          f"{'the held-out fold (biased)' if args.val_on_test else 'an inner split of the training folds'}")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    folds = D.stratified_folds(idx.labels, folds_n, C.SPLIT_SEED)
    out = C.result_dir(RESULT_NAME)

    rows, payload = [], []
    for k in range(folds_n):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(folds_n) if j != k])
        if args.val_on_test:
            val_idx = test_idx
        else:
            rng = np.random.default_rng(C.SPLIT_SEED + k)
            perm = rng.permutation(len(train_idx))
            n_val = max(1, int(round(args.inner_val_frac * len(train_idx))))
            val_idx = np.sort(train_idx[perm[:n_val]])
            train_idx = np.sort(train_idx[perm[n_val:]])
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        D.banner(f"Fold {k + 1}/{folds_n}  "
                 f"(train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)})", "-")
        r = H.train_variant(idx, splits, rc, attn_tokens=args.attn_tokens,
                            label=f"fold{k + 1}")
        t = r["test"]
        paper = C.PAPER_TABLE4[k] if k < len(C.PAPER_TABLE4) else None
        rows.append({"Fold": k + 1,
                     "Acc (%)": round(t["accuracy"], 2),
                     "F1 (%)": round(t["f1_macro"], 2),
                     "AUC": round(t.get("auc_macro", float("nan")), 4),
                     "Test images": int(t["n"]),
                     "Paper Acc (%)": paper["acc"] if paper else None,
                     "Paper F1 (%)": paper["f1"] if paper else None,
                     "Paper AUC": paper["auc"] if paper else None})
        payload.append({"fold": k + 1,
                        "test": {kk: vv for kk, vv in t.items()
                                 if kk not in ("y_true", "y_pred", "probs")},
                        "history": r["history"]})
        del r

    acc = np.array([r["Acc (%)"] for r in rows], dtype=float)
    f1 = np.array([r["F1 (%)"] for r in rows], dtype=float)
    auc = np.array([r["AUC"] for r in rows], dtype=float)
    rows.append({"Fold": "Mean+-Std",
                 "Acc (%)": f"{acc.mean():.2f}+-{acc.std():.2f}",
                 "F1 (%)": f"{f1.mean():.2f}+-{f1.std():.2f}",
                 "AUC": f"{np.nanmean(auc):.4f}+-{np.nanstd(auc):.4f}",
                 "Test images": int(sum(int(r["Test images"]) for r in rows)),
                 "Paper Acc (%)": "99.58+-0.07", "Paper F1 (%)": "99.55+-0.08",
                 "Paper AUC": "0.9997+-0.0001"})
    D.save_table(rows, out / "tables" / "table04_cross_validation",
                 f"Table 4 -- {folds_n}-fold stratified cross-validation")
    D.plot_grouped_bars([f"Fold {i + 1}" for i in range(folds_n)],
                        {"Accuracy (%)": acc.tolist(), "Macro-F1 (%)": f1.tolist()},
                        out / "figures" / "table04_folds.png",
                        f"HTA-ConvNet {folds_n}-fold cross-validation", "Score (%)",
                        rotate=0)
    D.save_json({"folds": payload,
                 "summary": {"acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                             "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
                             "auc_mean": float(np.nanmean(auc)),
                             "auc_std": float(np.nanstd(auc))},
                 "protocol": {"folds": folds_n, "val_on_test": args.val_on_test,
                              "inner_val_frac": args.inner_val_frac},
                 "run": {"epochs": rc.epochs, "img_size": rc.img_size,
                         "seed": rc.seed, "quick": rc.quick},
                 "paper_table4": C.PAPER_TABLE4},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"Mean accuracy {acc.mean():.2f} +- {acc.std():.2f} %   "
          f"mean macro-F1 {f1.mean():.2f} +- {f1.std():.2f} %")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
