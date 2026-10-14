"""Sensitivity analyses: attention heads (Fig. 'heads'), number/placement of DBSA blocks, and input
resolution (Table 'resolution').
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'hta_k1': ['--heads', '1'], 'hta_k2': ['--heads', '2'], 'hta_k4': ['--heads', '4'], 'hta_k16': ['--heads', '16'], 'hta_stages_2345': ['--stages', '2,3,4,5'], 'hta_stages_1234': ['--stages', '1,2,3,4'], 'hta_res96': ['--img-size', '96'], 'hta_res128': ['--img-size', '128'], 'hta_res160': ['--img-size', '160'], 'hta_res256': ['--img-size', '256']}


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
