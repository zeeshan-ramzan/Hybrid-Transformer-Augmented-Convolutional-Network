"""AlexNet (Krizhevsky et al., 2012).

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main').

Usage:
    python scripts/11_alexnet.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("AlexNet (Krizhevsky et al., 2012)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("alexnet", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "alexnet",
                   extra={"arch": "alexnet"})


if __name__ == "__main__":
    main()
