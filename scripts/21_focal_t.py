"""Focal-T (Yang et al., NeurIPS 2021) [NEW in revision, R3.4].

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').
Requires FOCAL_REPO (clone of microsoft/Focal-Transformer) and FOCAL_T_CKPT (ImageNet-1k Focal-T checkpoint).

Usage:
    python scripts/21_focal_t.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("Focal-T (Yang et al., NeurIPS 2021)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("focal_t", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "focal_t",
                   extra={"arch": "focal_t"})


if __name__ == "__main__":
    main()
