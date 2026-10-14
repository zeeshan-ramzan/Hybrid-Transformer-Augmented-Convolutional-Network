"""Ablation (b): fusion variants with DBSA at Stages 2-4 (Table 'ablation', part b; also used by
58_fusion_stability.py).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'fusion_avg': ['--fusion', 'avg'], 'fusion_add': ['--fusion', 'add'], 'fusion_concat': ['--fusion', 'concat'], 'fusion_gate_nores': ['--fusion', 'gate_nores']}


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
