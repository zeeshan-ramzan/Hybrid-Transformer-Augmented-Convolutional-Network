"""ViT-B/16 (Dosovitskiy et al., 2021).

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').

Usage:
    python scripts/18_vit_b16.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("ViT-B/16 (Dosovitskiy et al., 2021)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("vit_b16", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "vit_b16",
                   extra={"arch": "vit_b16"})


if __name__ == "__main__":
    main()
