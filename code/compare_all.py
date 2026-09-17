"""
compare_all.py -- Tables 10, 13, 15 and 16, plus the cross-method figures
========================================================================
Collects the results every other script has already written and builds the
paper's comparison outputs:

    Table 10  Comparison of classification performance on the PlantVillage test
              set -- all eleven rows, this run beside the published values
    Table 13  Model complexity and inference speed (parameters, MACs, FPS)
    Table 15  Comparison with classical feature-based methods
    Table 16  Progression of state of the art on PlantVillage (2016-present).
              This is a literature table: no code can reproduce it, so it is
              reprinted verbatim from `config.PAPER_TABLE16` with this run's
              HTA-ConvNet number appended for context.
    Fig.      grouped Acc/Prec/Rec/F1 bars, accuracy-versus-cost scatter,
              per-class F1 comparison, and a paper-vs-reproduction scatter

It reads `results/<method>/<method>_results.json`, so run the algorithms first:

    python hta_convnet.py
    python baseline_efficientnet_b4.py       (and the other baseline_*.py)
    python baseline_classical.py
    python compare_all.py

or let this script launch whatever is missing:

    python compare_all.py --run-missing

Outputs -> results/comparison/

Usage
-----
    python compare_all.py
    python compare_all.py --quick --run-missing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as C
import data_utils as D

RESULT_NAME = "comparison"
HERE = Path(__file__).resolve().parent

# display name, results folder, producing script, Table 10 family
METHODS: list[tuple[str, str, str, str]] = [
    ("AlexNet", "alexnet", "baseline_alexnet.py", "CNN"),
    ("VGG-16", "vgg16", "baseline_vgg16.py", "CNN"),
    ("ResNet-50", "resnet50", "baseline_resnet50.py", "CNN"),
    ("InceptionV3", "inceptionv3", "baseline_inceptionv3.py", "CNN"),
    ("DenseNet-121", "densenet121", "baseline_densenet121.py", "CNN"),
    ("MobileNetV2", "mobilenetv2", "baseline_mobilenetv2.py", "CNN"),
    ("EfficientNet-B4", "efficientnet_b4", "baseline_efficientnet_b4.py", "CNN"),
    ("GLCM + SVM (Ahmad et al.)", "glcm_svm", "baseline_classical.py", "Feature-based"),
    ("PlantDNet", "plantdnet", "baseline_plantdnet.py", "CNN"),
    ("ViT-B/16", "vit_b16", "baseline_vit_b16.py", "Transformer"),
    ("HTA-ConvNet (Ours)", "hta_convnet", "hta_convnet.py", "Hybrid (proposed)"),
]
CLASSICAL: list[tuple[str, str]] = [
    ("GLCM + SVM (Ahmad et al.)", "glcm_svm"),
    ("Color Hist + kNN", "colorhist_knn"),
    ("LBP + SVM", "lbp_svm"),
    ("HOG + Random Forest", "hog_rf"),
    ("Color+GLCM+HOG + SVM", "fused_svm"),
    ("HTA-ConvNet (Ours)", "hta_convnet"),
]


def _run(script: str, args) -> None:
    cmd = [sys.executable, str(HERE / script)]
    if args.quick:
        cmd.append("--quick")
    if args.cpu:
        cmd.append("--cpu")
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    print("\n" + "#" * 78 + f"\n# {' '.join(cmd[1:])}\n" + "#" * 78, flush=True)
    subprocess.run(cmd, cwd=HERE)


def collect(args) -> dict[str, dict]:
    found: dict[str, dict] = {}
    launched: set[str] = set()
    for display, res, script, _ in METHODS:
        r = D.read_result(res)
        if r is None and args.run_missing and script not in launched:
            _run(script, args)
            launched.add(script)
            r = D.read_result(res)
        if r is not None:
            found[display] = r
    for display, res in CLASSICAL:
        if display in found:
            continue
        r = D.read_result(res)
        if r is None and args.run_missing and "baseline_classical.py" not in launched:
            _run("baseline_classical.py", args)
            launched.add("baseline_classical.py")
            r = D.read_result(res)
        if r is not None:
            found[display] = r
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate every method into the paper's comparison tables.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-missing", action="store_true",
                    help="launch any producing script whose results are absent")
    args = ap.parse_args()

    D.banner("CROSS-METHOD COMPARISON (Tables 10, 13, 15, 16)")
    found = collect(args)
    if not found:
        raise SystemExit(
            "\n[!] No results found in results/.\n"
            "    Run at least  python hta_convnet.py  first, or pass --run-missing.")
    print(f"  {len(found)} method(s) with stored results: {', '.join(found)}")
    out = C.result_dir(RESULT_NAME)

    # ------------------------------------------------------------- Table 10
    rows10 = []
    for display, res, _, family in METHODS:
        r = found.get(display)
        if r is None:
            continue
        t = r["test"]
        paper = C.PAPER_TABLE10.get(display, ("-",) * 4)
        rows10.append({"Method": display, "Family": family,
                       "Acc (%)": round(t["accuracy"], 2),
                       "Prec (%)": round(t["precision_macro"], 2),
                       "Rec (%)": round(t["recall_macro"], 2),
                       "F1 (%)": round(t["f1_macro"], 2),
                       "AUC": round(t.get("auc_macro", float("nan")), 4),
                       "ECE": round(t.get("ece", float("nan")), 4),
                       "Paper Acc (%)": paper[0], "Paper F1 (%)": paper[3]})
    D.save_table(rows10, out / "tables" / "table10_comparison",
                 "Table 10 -- classification performance on the test split")

    if len(rows10) >= 2:
        D.plot_grouped_bars([r["Method"] for r in rows10],
                            {"Accuracy": [r["Acc (%)"] for r in rows10],
                             "Precision": [r["Prec (%)"] for r in rows10],
                             "Recall": [r["Rec (%)"] for r in rows10],
                             "F1": [r["F1 (%)"] for r in rows10]},
                            out / "figures" / "table10_metrics.png",
                            "Test-set performance by method", "Score (%)", rotate=30)
        D.plot_scatter_labeled([r["Paper Acc (%)"] for r in rows10 if r["Paper Acc (%)"]],
                               [r["Acc (%)"] for r in rows10 if r["Paper Acc (%)"]],
                               [r["Method"] for r in rows10 if r["Paper Acc (%)"]],
                               out / "figures" / "paper_vs_reproduction.png",
                               "Published accuracy vs this run",
                               "Paper accuracy (%)", "This run accuracy (%)",
                               highlight="HTA-ConvNet")

    # ------------------------------------------------------------- Table 13
    rows13 = []
    for display, res, _, _ in METHODS:
        r = found.get(display)
        if r is None or "complexity" not in r:
            continue
        cx = r["complexity"]
        paper = C.PAPER_TABLE13.get(display, ("-",) * 3)
        rows13.append({"Method": display,
                       "Params (M)": round(cx["params"] / 1e6, 2),
                       "GFLOPs (MACs)": round(cx["gflops"], 3),
                       "Input": f"{cx['input_size']}x{cx['input_size']}",
                       "FPS (this machine)": round(cx["fps_measured"], 1),
                       "Paper Params (M)": paper[0], "Paper GFLOPs": paper[1],
                       "Paper FPS (A100)": paper[2]})
    if rows13:
        D.save_table(rows13, out / "tables" / "table13_complexity",
                     "Table 13 -- model complexity and inference speed")
        acc_by = {r["Method"]: r["Acc (%)"] for r in rows10}
        pts = [(r["Params (M)"], acc_by.get(r["Method"]), r["Method"])
               for r in rows13 if r["Method"] in acc_by]
        if pts:
            D.plot_scatter_labeled([p[0] for p in pts], [p[1] for p in pts],
                                   [p[2] for p in pts],
                                   out / "figures" / "accuracy_vs_params.png",
                                   "Accuracy against model size",
                                   "Parameters (M)", "Accuracy (%)",
                                   highlight="HTA-ConvNet")

    # ------------------------------------------------------------- Table 15
    rows15 = []
    for display, res in CLASSICAL:
        r = found.get(display)
        if r is None:
            continue
        t = r["test"]
        rows15.append({"Method": display,
                       "Features": C.PAPER_TABLE15[display][0],
                       "Acc (%)": round(t["accuracy"], 2),
                       "F1 (%)": round(t["f1_macro"], 2),
                       "Paper Acc (%)": C.PAPER_TABLE15[display][1]})
    if rows15:
        D.save_table(rows15, out / "tables" / "table15_classical",
                     "Table 15 -- classical feature-based methods vs HTA-ConvNet")

    # ------------------------------------------------------------- Table 16
    rows16 = [dict(Method=r["method"], Year=r["year"], **{"Acc (%)": r["acc"]},
                   Venue=r["venue"], Source="published literature")
              for r in C.PAPER_TABLE16]
    hta = found.get("HTA-ConvNet (Ours)")
    if hta:
        rows16.append({"Method": "HTA-ConvNet (this reproduction)", "Year": 2025,
                       "Acc (%)": round(hta["test"]["accuracy"], 2),
                       "Venue": "-", "Source": "this run"})
    D.save_table(rows16, out / "tables" / "table16_sota_progression",
                 "Table 16 -- progression of state of the art on PlantVillage "
                 "(literature values, not reproduced)", print_it=False)

    # ------------------------------------------- per-class F1 comparison plot
    per_class = {d: np.asarray(r["per_class_f1"], dtype=float)
                 for d, r in found.items() if r.get("per_class_f1")}
    if len(per_class) >= 2:
        n = min(len(v) for v in per_class.values())
        classes = found["HTA-ConvNet (Ours)"]["classes"] if hta else \
            [f"C{i + 1}" for i in range(n)]
        top = sorted(per_class, key=lambda k: -float(per_class[k][:n].mean()))[:4]
        D.plot_grouped_bars([c[:18] for c in classes[:n]],
                            {k: per_class[k][:n].tolist() for k in top},
                            out / "figures" / "per_class_f1_comparison.png",
                            "Per-class F1 -- best methods", "F1 (%)", rotate=60)

    # ---------------------------------------------------- ranked summary plot
    if rows10:
        ranked = sorted(rows10, key=lambda r: r["Acc (%)"])
        D.plot_bars([r["Method"] for r in ranked], [r["Acc (%)"] for r in ranked],
                    out / "figures" / "accuracy_ranking.png",
                    "Test accuracy ranking", "Accuracy (%)",
                    highlight="HTA-ConvNet", rotate=30)

    D.save_json({"table10": rows10, "table13": rows13, "table15": rows15,
                 "table16": rows16,
                 "methods_found": sorted(found),
                 "note": "Table 16 is a literature table; no code reproduces it."},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    if rows10:
        best = max(rows10, key=lambda r: r["Acc (%)"])
        print(f"Best method in this run: {best['Method']} at {best['Acc (%)']}% accuracy")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
