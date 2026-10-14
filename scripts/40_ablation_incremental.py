"""Ablation (a): incremental construction (Table 'ablation', part a).

Backbone-only = scripts/17_efficientnet_b4.py. Usage: python scripts/40_ablation_incremental.py --data-root ...
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'abl_dwsc_stage4': ['--stages', '4', '--branches', 'conv'], 'abl_mhsa_stage4': ['--stages', '4', '--branches', 'attn'], 'abl_dbsa_stage4': ['--stages', '4'], 'abl_dbsa_stage34': ['--stages', '3,4'], 'hta_convnet': ['--stages', '2,3,4']}


def main():
    passthrough = sys.argv[1:]          # e.g. --data-root ... --epochs ...
    only = os.environ.get("ONLY")       # optional: run a single configuration by name
    for name, flags in CONFIGS.items():
        if only and name != only:
            continue
        cmd = [sys.executable, os.path.join(HERE, "10_hta_convnet.py"), "--name", name] + flags + passthrough
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
