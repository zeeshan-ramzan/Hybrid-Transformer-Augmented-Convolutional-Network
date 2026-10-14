"""Swin-T (Liu et al., ICCV 2021) [NEW in revision, R3.4].

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').
timm weights: swin_tiny_patch4_window7_224.ms_in1k

Usage:
    python scripts/20_swin_t.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("Swin-T (Liu et al., ICCV 2021)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("swin_t", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "swin_t",
                   extra={"arch": "swin_t"})


if __name__ == "__main__":
    main()
