"""Multi-seed repetition (Table 'seeds'): trains HTA-ConvNet with seeds 42-51 on the fixed split and reports
mean, SD and the 95% t-interval (9 d.f.) of accuracy and macro-F1.

Usage: python scripts/51_multi_seed.py --data-root data/PlantVillage/color [--model-script 17_efficientnet_b4.py]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.metrics import t_ci
from common.utils import save_json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--model-script", default="10_hta_convnet.py")
    ap.add_argument("--name", default="hta_convnet")
    a, extra = ap.parse_known_args()
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        cmd = [sys.executable, os.path.join(HERE, a.model_script), "--seed", str(s), "--out-dir", "runs/seeds"]
        if a.model_script == "10_hta_convnet.py":
            cmd += ["--name", a.name]
        subprocess.run(cmd + extra, check=True)
        d = os.path.join("runs/seeds", a.name + (f"_seed{s}" if s != 42 else ""))
        with open(os.path.join(d, "results.json")) as f:
            r = json.load(f)
        mins = sum(h["sec"] for h in r["history"]) / 60
        rows.append(dict(seed=s, acc=r["test"]["acc"], f1=r["test"]["f1"], train_minutes=mins))
        print(rows[-1])
    out = dict(runs=rows, acc=dict(zip(["mean", "sd", "ci_low", "ci_high"], t_ci([r["acc"] for r in rows]))),
               f1=dict(zip(["mean", "sd", "ci_low", "ci_high"], t_ci([r["f1"] for r in rows]))))
    save_json(out, f"runs/seeds/{a.name}_multiseed_summary.json")
    print(out["acc"], out["f1"])


if __name__ == "__main__":
    main()
