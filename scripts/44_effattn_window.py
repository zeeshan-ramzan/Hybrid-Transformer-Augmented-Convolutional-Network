"""[NEW in revision, R3.1] Stage-2 DBSA with Swin-style window attention (non-overlapping 7x7 windows).
Stages 3 and 4 keep dense MHSA (Table 'effattn').
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'effattn_window7': ['--stage2-attention', 'window']}


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
