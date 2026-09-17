"""
run_all.py -- convenience driver
================================
Runs every algorithm and every experiment in the order the paper presents them,
each in its own process, and finishes with the cross-method comparison and the
statistical tests.

This is only a convenience wrapper: each script is fully standalone and can be
run on its own (that is the recommended way -- "one execution per algorithm").

Usage
-----
    python run_all.py --quick          # smoke test: everything, tiny schedule
    python run_all.py                  # full paper protocol (slow: use a GPU)
    python run_all.py --skip vgg16 vit_b16
    python run_all.py --only hta ablation comparison
    python run_all.py --list           # print the stage list and exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent

STAGES: list[tuple[str, str]] = [
    # proposed architecture first, so every later comparison has it available
    ("hta", "hta_convnet.py"),
    # Table 10 / Table 13 baselines
    ("alexnet", "baseline_alexnet.py"),
    ("vgg16", "baseline_vgg16.py"),
    ("resnet50", "baseline_resnet50.py"),
    ("inceptionv3", "baseline_inceptionv3.py"),
    ("densenet121", "baseline_densenet121.py"),
    ("mobilenetv2", "baseline_mobilenetv2.py"),
    ("efficientnet_b4", "baseline_efficientnet_b4.py"),
    ("vit_b16", "baseline_vit_b16.py"),
    ("plantdnet", "baseline_plantdnet.py"),
    ("classical", "baseline_classical.py"),
    # experiments
    ("ablation", "ablation_study.py"),
    ("cross_dataset", "cross_dataset_plantdoc.py"),
    ("cross_validation", "cross_validation.py"),
    ("seeds", "seed_robustness.py"),
    ("heads", "attention_heads_sensitivity.py"),
    ("resolution", "resolution_sensitivity.py"),
    ("dbsa_blocks", "dbsa_blocks_sensitivity.py"),
    # aggregation
    ("comparison", "compare_all.py"),
    ("stats", "statistical_tests.py"),
]

# scripts that do not accept the full data CLI (they only read stored results)
LIMITED_CLI = {"compare_all.py", "statistical_tests.py"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run every script in sequence.")
    ap.add_argument("--quick", action="store_true",
                    help="tiny schedule for a full smoke test")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    ap.add_argument("--only", nargs="*", default=None, help="run only these stages")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    args = ap.parse_args()

    if args.list:
        for name, script in STAGES:
            print(f"  {name:<18s} {script}")
        return

    has_images = any(p.is_dir() and not p.name.startswith(".")
                     for p in C.DATASET_DIR.iterdir())
    if not has_images:
        print(f"[!] {C.DATASET_DIR} holds no image folders.\n"
              "    Put PlantVillage there (README section 3), or create a\n"
              "    SIMULATED stand-in with\n"
              "        python make_synthetic_dataset.py")
        sys.exit(1)

    ok, failed, t_all = [], [], time.perf_counter()
    for name, script in STAGES:
        if name in args.skip or (args.only and name not in args.only):
            continue
        cmd = [sys.executable, str(HERE / script)]
        if args.quick:
            cmd.append("--quick")
        if args.cpu:
            cmd.append("--cpu")
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if script not in LIMITED_CLI:
            if args.pretrained:
                cmd.append("--pretrained")
            if args.img_size:
                cmd += ["--img-size", str(args.img_size)]
        print("\n" + "#" * 78 + f"\n# {name}:  {' '.join(cmd[1:])}\n" + "#" * 78,
              flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=HERE).returncode
        dt = time.perf_counter() - t0
        (ok if rc == 0 else failed).append((name, dt, rc))
        print(f"# {name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'}  [{dt:.1f}s]",
              flush=True)

    print("\n" + "=" * 78)
    print(f"Completed {len(ok)}/{len(ok) + len(failed)} stages "
          f"in {(time.perf_counter() - t_all) / 60:.1f} min")
    for n, dt, _ in ok:
        print(f"   OK      {n:<18s} {dt:8.1f}s")
    for n, dt, rc in failed:
        print(f"   FAILED  {n:<18s} {dt:8.1f}s  (exit {rc})")
    print(f"\nAll results in: {C.RESULTS_DIR}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
