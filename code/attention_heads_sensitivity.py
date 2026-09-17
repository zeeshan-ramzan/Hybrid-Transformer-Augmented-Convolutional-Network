"""
attention_heads_sensitivity.py -- paper Fig. 8
==============================================
Validation accuracy as a function of the number of attention heads k used in the
MHSA branch of every DBSA block.

    Fig. 8  Accuracy vs attention heads (MHSA)
    ------------------------------------------
      k = 1    99.27 %          k = 8    99.61 %   <- optimum
      k = 2    99.41 %          k = 16   99.56 %
      k = 4    99.52 %

The paper's argument is that a single head is clearly worse than any multi-head
configuration (diverse heads capture complementary disease-discriminative
spatial relationships), that accuracy plateaus at k = 8, and that splitting
attention across 16 heads leaves each head with too little capacity.

Eq. (4) requires d_k = C/k, i.e. k must divide the channel count of the stage.
The paper's Table 6 widths (88, 120, 208) are all divisible by 8 but not by 16,
and the real EfficientNet-B4 widths are different again, so
`hta_convnet.pick_heads` clips k to the largest feasible divisor and the table
reports the head count actually used at each stage.  See README section 9.3.

Following the paper, accuracy is reported on the VALIDATION split -- this is a
hyper-parameter selection experiment, and reporting it on the test split would
leak.  Test accuracy is recorded in the JSON for completeness.

Outputs -> results/heads_sensitivity/
    tables/fig08_attention_heads.csv    accuracy per k, with the paper's values
    figures/fig08_attention_heads.png   the reproduced Fig. 8
    heads_sensitivity_results.json

Usage
-----
    python attention_heads_sensitivity.py
    python attention_heads_sensitivity.py --quick
    python attention_heads_sensitivity.py --heads 1 4 8
"""

from __future__ import annotations

import argparse

import numpy as np

import config as C
import data_utils as D
import hta_convnet as H

RESULT_NAME = "heads_sensitivity"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sensitivity to the number of MHSA heads (paper Fig. 8).")
    D.cli_common(ap)
    ap.add_argument("--heads", type=int, nargs="*", default=list(C.HEAD_SWEEP))
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    heads = args.heads if not rc.quick else [1, 8]
    D.banner("SENSITIVITY TO THE NUMBER OF ATTENTION HEADS (Fig. 8)")
    D.describe_run(rc, "heads_sensitivity")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc)
    out = C.result_dir(RESULT_NAME)

    rows, payload = [], {}
    for k in heads:
        D.banner(f"k = {k} attention heads", "-")
        r = H.train_variant(idx, splits, rc, n_heads=k, attn_tokens=args.attn_tokens,
                            loaders=loaders, label=f"k={k}")
        used = {s: r["model"].dbsa[str(s)].n_heads for s in r["model"].dbsa_stages}
        rows.append({"Heads k (requested)": k,
                     "Heads used per stage": ", ".join(f"S{s}:{v}" for s, v in used.items()),
                     "Val Acc (%)": round(r["val"]["accuracy"], 2),
                     "Val F1 (%)": round(r["val"]["f1_macro"], 2),
                     "Test Acc (%)": round(r["test"]["accuracy"], 2),
                     "Params (M)": round(r["params"] / 1e6, 2),
                     "Paper Val Acc (%)": C.PAPER_FIG8.get(k)})
        payload[str(k)] = {
            "heads_used": {str(s): int(v) for s, v in used.items()},
            "val": {kk: vv for kk, vv in r["val"].items()
                    if kk not in ("y_true", "y_pred", "probs")},
            "test": {kk: vv for kk, vv in r["test"].items()
                     if kk not in ("y_true", "y_pred", "probs")},
            "paper_val_acc": C.PAPER_FIG8.get(k)}
        del r

    D.save_table(rows, out / "tables" / "fig08_attention_heads",
                 "Fig. 8 -- validation accuracy vs the number of MHSA heads")
    va = [r["Val Acc (%)"] for r in rows]
    best = heads[int(np.argmax(va))]
    D.plot_lines(heads, {"Validation accuracy (%)": va},
                 out / "figures" / "fig08_attention_heads.png",
                 "Accuracy vs attention heads (MHSA branch)",
                 "Number of attention heads k", "Validation accuracy (%)",
                 highlight_x=best, logx=len(heads) > 2)
    D.save_json({"heads": payload, "best_k": int(best),
                 "run": {"epochs": rc.epochs, "img_size": size, "seed": rc.seed,
                         "quick": rc.quick},
                 "paper_fig8": C.PAPER_FIG8},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"Best k on the validation split: {best} "
          f"(the paper reports k = 8)")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
