"""
seed_robustness.py -- paper Table 9
===================================
HTA-ConvNet trained from scratch under three independent random seeds, to show
that the headline number is stable rather than a lucky initialisation.

    Table 9  HTA-ConvNet Results Across Three Random Seeds
    ------------------------------------------------------
      Seed 42   99.61   99.58   4.2 h
      Seed 43   99.59   99.55   4.1 h
      Seed 44   99.63   99.60   4.3 h
      Mean      99.61   99.58   4.2 h        Std   0.02   0.03   0.1 h

The data split stays fixed at seed 42 (the paper's "split fixed by random seed
42") while the *training* seed -- weight initialisation, augmentation sampling,
batch order, mixup coefficients and dropout masks -- varies, so the spread
measures optimisation variance and not partition variance.

Outputs -> results/seed_robustness/
    tables/table09_seeds.csv         per-seed accuracy, F1 and wall-clock time
    figures/table09_seeds.png        accuracy and F1 per seed
    seed_robustness_results.json

Usage
-----
    python seed_robustness.py
    python seed_robustness.py --quick
    python seed_robustness.py --seeds 42 43 44 45
"""

from __future__ import annotations

import argparse

import numpy as np

import config as C
import data_utils as D
import hta_convnet as H

RESULT_NAME = "seed_robustness"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HTA-ConvNet stability across random seeds (paper Table 9).")
    D.cli_common(ap)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    seeds = args.seeds if not rc.quick else args.seeds[:2]
    D.banner("SEED ROBUSTNESS OF HTA-ConvNet (Table 9)")
    D.describe_run(rc, "seed_robustness")
    print(f"  training seeds: {seeds} (the data split stays fixed at seed "
          f"{C.SPLIT_SEED})")

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc)
    out = C.result_dir(RESULT_NAME)

    rows, payload = [], {}
    for s in seeds:
        D.banner(f"Seed {s}", "-")
        r = H.train_variant(idx, splits, rc, attn_tokens=args.attn_tokens,
                            loaders=loaders, seed=s, label=f"seed{s}")
        t = r["test"]
        paper = C.PAPER_TABLE9.get(s)
        rows.append({"Seed": s,
                     "Acc (%)": round(t["accuracy"], 2),
                     "F1 (%)": round(t["f1_macro"], 2),
                     "Epochs run": r["history"]["epochs_run"],
                     "Train time (s)": round(r["history"]["train_time_s"], 1),
                     "Paper Acc (%)": paper[0] if paper else None,
                     "Paper F1 (%)": paper[1] if paper else None,
                     "Paper time (h)": paper[2] if paper else None})
        payload[str(s)] = {"test": {k: v for k, v in t.items()
                                    if k not in ("y_true", "y_pred", "probs")},
                           "history": r["history"]}
        D.save_predictions(f"HTA-ConvNet seed {s}", t["y_true"], t["y_pred"],
                           t["probs"], tag="seed")
        del r

    acc = np.array([r["Acc (%)"] for r in rows], dtype=float)
    f1 = np.array([r["F1 (%)"] for r in rows], dtype=float)
    tt = np.array([r["Train time (s)"] for r in rows], dtype=float)
    rows.append({"Seed": "Mean", "Acc (%)": round(acc.mean(), 2),
                 "F1 (%)": round(f1.mean(), 2), "Epochs run": "-",
                 "Train time (s)": round(tt.mean(), 1),
                 "Paper Acc (%)": 99.61, "Paper F1 (%)": 99.58, "Paper time (h)": 4.2})
    rows.append({"Seed": "Std Dev", "Acc (%)": round(acc.std(), 3),
                 "F1 (%)": round(f1.std(), 3), "Epochs run": "-",
                 "Train time (s)": round(tt.std(), 1),
                 "Paper Acc (%)": 0.02, "Paper F1 (%)": 0.03, "Paper time (h)": 0.1})
    D.save_table(rows, out / "tables" / "table09_seeds",
                 "Table 9 -- HTA-ConvNet results across random seeds")
    D.plot_grouped_bars([f"Seed {s}" for s in seeds],
                        {"Accuracy (%)": acc.tolist(), "Macro-F1 (%)": f1.tolist()},
                        out / "figures" / "table09_seeds.png",
                        "HTA-ConvNet stability across training seeds", "Score (%)",
                        rotate=0)
    D.save_json({"seeds": payload,
                 "summary": {"acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                             "f1_mean": float(f1.mean()), "f1_std": float(f1.std())},
                 "run": {"epochs": rc.epochs, "img_size": size, "quick": rc.quick},
                 "paper_table9": C.PAPER_TABLE9},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"Accuracy {acc.mean():.2f} +- {acc.std():.3f} %   "
          f"macro-F1 {f1.mean():.2f} +- {f1.std():.3f} %")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
