"""
statistical_tests.py -- paper sections "Statistical Significance Testing" and
"Calibration Analysis"
============================================================================
Two claims of the paper that are checks on the *comparison* rather than on any
one model:

1. **Paired McNemar test.**  "A paired McNemar's test is used to compare the
   per-sample classification outcomes of HTA-ConvNet and EfficientNet-B4 on the
   8,147-sample test set. This yields a test statistic of chi^2 = 47.3 with
   p < 0.001 ...; the comparison against Ahmad et al. gives a similar result
   (chi^2 = 61.8, p < 0.001)."

   McNemar's test is the right choice here because both models are evaluated on
   exactly the same samples: it looks only at the discordant pairs -- images one
   model gets right and the other gets wrong -- and asks whether the two
   discordant counts n01 and n10 are plausibly equal.

2. **Expected Calibration Error.**  "HTA-ConvNet achieves an ECE of 0.0087,
   notably lower than the 0.0193 obtained by the backbone alone and the 0.0341
   obtained by ViT-B/16."  ECE is computed over 15 equal-width confidence bins.

This script reads the per-sample predictions that the other scripts wrote to
`results/_predictions/`, so run those first (or pass `--run-missing`).

Note on the paper's arithmetic: Table 5 gives a test split of 8,146 images while
this section says 8,147.  See README section 9.5.

Outputs -> results/statistical_tests/
    tables/mcnemar_tests.csv           discordant counts, chi^2 and p per pair
    tables/calibration_comparison.csv  ECE / NLL / accuracy per method
    figures/mcnemar_discordant.png     n01 vs n10 for every pair
    figures/calibration_comparison.png ECE per method
    statistical_tests_results.json

Usage
-----
    python statistical_tests.py
    python statistical_tests.py --run-missing --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as C
import data_utils as D

RESULT_NAME = "statistical_tests"
HERE = Path(__file__).resolve().parent

REFERENCE = "HTA-ConvNet (Ours)"
# display name -> producing script
SOURCES = {
    "HTA-ConvNet (Ours)": "hta_convnet.py",
    "EfficientNet-B4": "baseline_efficientnet_b4.py",
    "GLCM + SVM (Ahmad et al.)": "baseline_classical.py",
    "ViT-B/16": "baseline_vit_b16.py",
    "ResNet-50": "baseline_resnet50.py",
    "DenseNet-121": "baseline_densenet121.py",
    "MobileNetV2": "baseline_mobilenetv2.py",
    "InceptionV3": "baseline_inceptionv3.py",
    "VGG-16": "baseline_vgg16.py",
    "AlexNet": "baseline_alexnet.py",
    "PlantDNet": "baseline_plantdnet.py",
}
PAPER_CHI2 = {"EfficientNet-B4": C.PAPER_MISC["mcnemar_chi2_vs_effb4"],
              "GLCM + SVM (Ahmad et al.)": C.PAPER_MISC["mcnemar_chi2_vs_ahmad"]}
PAPER_ECE = {"HTA-ConvNet (Ours)": C.PAPER_MISC["ece_hta"],
             "EfficientNet-B4": C.PAPER_MISC["ece_backbone"],
             "ViT-B/16": C.PAPER_MISC["ece_vit"]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="McNemar significance tests and calibration comparison.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-missing", action="store_true",
                    help="launch any producing script whose predictions are absent")
    args = ap.parse_args()

    D.banner("STATISTICAL SIGNIFICANCE AND CALIBRATION")
    out = C.result_dir(RESULT_NAME)

    available: dict[str, tuple] = {}
    for name, script in SOURCES.items():
        p = D.load_predictions(name)
        if p is None and args.run_missing:
            cmd = [sys.executable, str(HERE / script)]
            for flag, val in (("--quick", args.quick), ("--cpu", args.cpu)):
                if val:
                    cmd.append(flag)
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            print(f"  running {script} ...")
            subprocess.run(cmd, cwd=HERE)
            p = D.load_predictions(name)
        if p is not None:
            available[name] = p

    if REFERENCE not in available:
        raise SystemExit(
            f"\n[!] No stored predictions for {REFERENCE}.\n"
            "    Run  python hta_convnet.py  first (or pass --run-missing).")
    print(f"  methods with stored test predictions: {len(available)} "
          f"({', '.join(available)})")

    y_ref, p_ref, prob_ref = available[REFERENCE]

    # -------------------------------------------------------------- McNemar
    rows = []
    for name, (y, pred, _) in available.items():
        if name == REFERENCE:
            continue
        if len(y) != len(y_ref) or not np.array_equal(y, y_ref):
            print(f"  [!] {name}: test predictions do not align sample-for-sample "
                  "with the reference (different run configuration?) -- skipped")
            continue
        r = D.mcnemar(p_ref == y_ref, pred == y)
        rows.append({"Method A": REFERENCE, "Method B": name,
                     "n (test)": int(len(y)),
                     "A right, B wrong (n01)": r["n01"],
                     "A wrong, B right (n10)": r["n10"],
                     "chi^2": round(r["chi2"], 2),
                     "p-value": f"{r['p_value']:.3g}",
                     "Significant (p<0.05)": "yes" if r["p_value"] < 0.05 else "no",
                     "Paper chi^2": PAPER_CHI2.get(name, "-")})
    if rows:
        D.save_table(rows, out / "tables" / "mcnemar_tests",
                     "Paired McNemar tests against HTA-ConvNet")
        D.plot_grouped_bars([r["Method B"] for r in rows],
                            {"HTA right / other wrong":
                                 [r["A right, B wrong (n01)"] for r in rows],
                             "HTA wrong / other right":
                                 [r["A wrong, B right (n10)"] for r in rows]},
                            out / "figures" / "mcnemar_discordant.png",
                            "Discordant test samples (McNemar)", "Number of images",
                            rotate=25)
    else:
        print("  [!] no comparable method found -- run more baselines first.")

    # ---------------------------------------------------------- calibration
    cal = []
    for name, (y, pred, probs) in available.items():
        if probs is None or probs.size == 0:
            continue
        ece = D.expected_calibration_error(probs, y)
        nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
        cal.append({"Method": name, "Accuracy (%)": round(100 * float((pred == y).mean()), 2),
                    "ECE (15 bins)": round(ece, 4), "NLL": round(nll, 4),
                    "Mean confidence": round(float(probs.max(1).mean()), 4),
                    "Paper ECE": PAPER_ECE.get(name, "-")})
    if cal:
        cal.sort(key=lambda r: r["ECE (15 bins)"])
        D.save_table(cal, out / "tables" / "calibration_comparison",
                     "Expected calibration error by method (15 bins)")
        D.plot_bars([r["Method"] for r in cal], [r["ECE (15 bins)"] for r in cal],
                    out / "figures" / "calibration_comparison.png",
                    "Expected calibration error (lower is better)", "ECE",
                    highlight="HTA-ConvNet", rotate=25)

    D.save_json({"mcnemar": rows, "calibration": cal,
                 "paper": {"chi2": PAPER_CHI2, "ece": PAPER_ECE,
                           "test_set_size_table5": C.PAPER_MISC["test_set_size_table5"],
                           "test_set_size_mcnemar": C.PAPER_MISC["test_set_size_mcnemar"]}},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
