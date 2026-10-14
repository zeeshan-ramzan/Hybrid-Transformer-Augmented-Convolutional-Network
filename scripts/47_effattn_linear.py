"""[NEW in revision, R3.1] Stage-2 DBSA with kernelised linear attention, phi=elu+1 (Table 'effattn').
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'effattn_linear': ['--stage2-attention', 'linear']}


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
