"""Ablation (c): DBSA replaced by SE, CBAM or a MobileViT block at Stages 2-4 (Table 'ablation', part c).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {'replace_se': ['--module', 'se'], 'replace_cbam': ['--module', 'cbam'], 'replace_mobilevit': ['--module', 'mobilevit']}


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
