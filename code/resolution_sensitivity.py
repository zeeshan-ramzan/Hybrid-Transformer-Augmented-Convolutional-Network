"""
resolution_sensitivity.py -- paper Table 17
===========================================
Accuracy and throughput as a function of the network's input resolution, with
the model retrained separately at each resolution and every other
hyper-parameter held fixed.

    Table 17  Resolution Sensitivity Analysis
    -------------------------------------------------------
      96 x 96     98.14 %    1,247 FPS (A100)    132 FPS (Pi 4)
      128 x 128   98.83 %      867 FPS            91 FPS
      160 x 160   99.21 %      612 FPS            64 FPS
      224 x 224   99.61 %      312 FPS            34 FPS
      256 x 256   99.64 %      241 FPS            26 FPS

The paper's point is that performance degrades only gracefully as resolution
falls -- still above 98% at 96 x 96 -- which it attributes to EfficientNet's
resolution-aware compound scaling and to the DBSA attention adapting to whatever
number of spatial tokens is available.

FPS here is measured on THIS machine, so it will not match the paper's A100 or
Raspberry Pi columns; the paper's values are printed alongside for reference.

Note.  The cost of Eq. (6) grows with the fourth power of the input side
(n = H*W tokens, cost 2n^2 C), so the highest resolutions are by far the most
expensive rows to run.  `--attn-tokens` caps the token count if needed.

Outputs -> results/resolution_sensitivity/
    tables/table17_resolution.csv       accuracy, F1, params, MACs, measured FPS
    figures/table17_accuracy_fps.png    accuracy and throughput vs resolution
    resolution_sensitivity_results.json

Usage
-----
    python resolution_sensitivity.py
    python resolution_sensitivity.py --quick
    python resolution_sensitivity.py --resolutions 96 128 160
"""

from __future__ import annotations

import argparse

import torch

import config as C
import data_utils as D
import hta_convnet as H
import nn_utils as N

RESULT_NAME = "resolution_sensitivity"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Input-resolution sensitivity of HTA-ConvNet (paper Table 17).")
    D.cli_common(ap)
    ap.add_argument("--resolutions", type=int, nargs="*", default=list(C.RESOLUTION_SWEEP))
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    res = args.resolutions if not rc.quick else [48, 64]
    D.banner("INPUT-RESOLUTION SENSITIVITY (Table 17)")
    D.describe_run(rc, "resolution_sensitivity")
    print(f"  resolutions: {res}")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    out = C.result_dir(RESULT_NAME)

    rows, payload = [], {}
    for r_px in res:
        D.banner(f"input {r_px} x {r_px}", "-")
        rc_r = D.RunConfig(**{**rc.__dict__, "img_size": int(r_px)})
        loaders, size = D.build_loaders(idx, splits, rc_r)
        v = H.train_variant(idx, splits, rc_r, attn_tokens=args.attn_tokens,
                            loaders=loaders, img_size=size, label=f"{r_px}px")
        macs = N.count_macs(v["model"], size, torch.device("cpu"))
        fps = N.measure_fps(v["model"], size, rc.device)
        paper = C.PAPER_TABLE17.get(int(r_px))
        rows.append({"Resolution": f"{r_px}x{r_px}",
                     "Acc (%)": round(v["test"]["accuracy"], 2),
                     "F1 (%)": round(v["test"]["f1_macro"], 2),
                     "Params (M)": round(v["params"] / 1e6, 2),
                     "GFLOPs (MACs)": round(macs / 1e9, 3),
                     "FPS (this machine)": round(fps, 1),
                     "Paper Acc (%)": paper[0] if paper else None,
                     "Paper FPS (A100)": paper[1] if paper else None,
                     "Paper FPS (Pi 4)": paper[2] if paper else None})
        payload[str(r_px)] = {
            "test": {k: val for k, val in v["test"].items()
                     if k not in ("y_true", "y_pred", "probs")},
            "macs": macs, "fps_measured": fps, "params": v["params"],
            "paper": list(paper) if paper else None}
        del v, loaders

    D.save_table(rows, out / "tables" / "table17_resolution",
                 "Table 17 -- resolution sensitivity analysis")
    D.plot_grouped_bars([r["Resolution"] for r in rows],
                        {"Accuracy (%)": [r["Acc (%)"] for r in rows],
                         "Macro-F1 (%)": [r["F1 (%)"] for r in rows]},
                        out / "figures" / "table17_accuracy_fps.png",
                        "Accuracy vs input resolution", "Score (%)", rotate=0)
    D.plot_lines([r["Resolution"] for r in rows],
                 {"FPS (this machine)": [r["FPS (this machine)"] for r in rows]},
                 out / "figures" / "table17_throughput.png",
                 "Inference throughput vs input resolution", "Resolution",
                 "Images per second")
    D.save_json({"resolutions": payload,
                 "run": {"epochs": rc.epochs, "seed": rc.seed, "quick": rc.quick},
                 "paper_table17": {str(k): list(v) for k, v in C.PAPER_TABLE17.items()}},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
