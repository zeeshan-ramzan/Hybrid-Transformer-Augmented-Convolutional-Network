"""
dbsa_blocks_sensitivity.py -- paper section "Sensitivity to Number of DBSA Blocks"
=================================================================================
How many DBSA blocks should the backbone carry, and where?

The paper evaluates configurations beyond those already covered by the Table 11
ablation and reports:

    no blocks                          98.92 %
    Stage 4 only                       99.27 %
    Stages 3 & 4                       99.43 %
    Stages 2, 3 & 4  (proposed)        99.61 %   <- best
    all five stages                    99.60 %   (+0.4 ms inference, no gain)
    adding Stage 1 instead             99.48 %   (112x112 -> n = 12,544 tokens,
                                                  "far more than 8 attention heads
                                                   can handle effectively")

This script retrains each configuration and additionally records the parameter
and MAC overhead of each, plus the inference latency, so the accuracy/cost
trade-off the paper argues for can be checked directly.

A caution about the paper's own numbers: it states that Stage 1 gives the MHSA
branch "n = 12544 tokens".  112 x 112 = 12,544, so the token count is right, but
Eq. (6)'s quadratic term at that size is 2 n^2 C = 15.1 GFLOPs for that single
block (using the paper's Stage-1 width C = 48) -- roughly seventy times the
0.21 GFLOPs the paper attributes to all DBSA blocks combined, and more than
three times its whole-network budget.  See README section 9.4.

Outputs -> results/dbsa_blocks/
    tables/dbsa_block_placement.csv     accuracy, params, MACs, latency per config
    figures/dbsa_block_placement.png    accuracy vs number of blocks
    dbsa_blocks_results.json

Usage
-----
    python dbsa_blocks_sensitivity.py
    python dbsa_blocks_sensitivity.py --quick
    python dbsa_blocks_sensitivity.py --only none s4 s234
"""

from __future__ import annotations

import argparse

import torch

import config as C
import data_utils as D
import hta_convnet as H
import nn_utils as N

RESULT_NAME = "dbsa_blocks"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sensitivity to the number and placement of DBSA blocks.")
    D.cli_common(ap)
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of {', '.join(c['key'] for c in C.DBSA_BLOCK_CONFIGS)}")
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    D.banner("SENSITIVITY TO THE NUMBER AND PLACEMENT OF DBSA BLOCKS")
    D.describe_run(rc, "dbsa_blocks")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    loaders, size = D.build_loaders(idx, splits, rc)
    out = C.result_dir(RESULT_NAME)

    configs = [c for c in C.DBSA_BLOCK_CONFIGS
               if args.only is None or c["key"] in args.only]
    rows, payload = [], {}
    for cfg in configs:
        D.banner(cfg["label"], "-")
        v = H.train_variant(idx, splits, rc, stages=cfg["stages"],
                            branch="both" if cfg["stages"] else "none",
                            attn_tokens=args.attn_tokens, loaders=loaders,
                            label=cfg["label"])
        m = v["model"]
        macs = N.count_macs(m, size, torch.device("cpu"))
        fps = N.measure_fps(m, size, rc.device)
        rows.append({"Configuration": cfg["label"],
                     "Blocks": len(cfg["stages"]),
                     "Stages": ",".join(map(str, cfg["stages"])) or "-",
                     "Acc (%)": round(v["test"]["accuracy"], 2),
                     "F1 (%)": round(v["test"]["f1_macro"], 2),
                     "Params (M)": round(v["params"] / 1e6, 2),
                     "DBSA params (M)": round(m.dbsa_param_overhead() / 1e6, 3),
                     "GFLOPs (MACs)": round(macs / 1e9, 3),
                     "Latency (ms/img)": round(1000.0 / max(fps, 1e-9), 2)})
        payload[cfg["key"]] = {
            "label": cfg["label"], "stages": list(cfg["stages"]),
            "test": {k: val for k, val in v["test"].items()
                     if k not in ("y_true", "y_pred", "probs")},
            "params": v["params"], "dbsa_params": m.dbsa_param_overhead(),
            "macs": macs, "fps_measured": fps}
        del v, m

    D.save_table(rows, out / "tables" / "dbsa_block_placement",
                 "DBSA block placement: accuracy against parameter and compute cost")
    D.plot_bars([r["Configuration"] for r in rows], [r["Acc (%)"] for r in rows],
                out / "figures" / "dbsa_block_placement.png",
                "Accuracy vs DBSA block placement", "Accuracy (%)",
                highlight="proposed", rotate=25)
    D.plot_scatter_labeled([r["GFLOPs (MACs)"] for r in rows],
                           [r["Acc (%)"] for r in rows],
                           [r["Stages"] for r in rows],
                           out / "figures" / "dbsa_accuracy_vs_cost.png",
                           "Accuracy against compute cost", "GFLOPs (MACs)",
                           "Accuracy (%)")
    D.save_json({"configurations": payload,
                 "run": {"epochs": rc.epochs, "img_size": size, "seed": rc.seed,
                         "quick": rc.quick}},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
