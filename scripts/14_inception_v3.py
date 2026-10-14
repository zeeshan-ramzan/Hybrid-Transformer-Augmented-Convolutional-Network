"""InceptionV3 (Szegedy et al., 2016), 224x224 input.

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'complexity').

Usage:
    python scripts/14_inception_v3.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("InceptionV3 (Szegedy et al., 2016), 224x224 input")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("inception_v3", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "inception_v3",
                   extra={"arch": "inception_v3"})


if __name__ == "__main__":
    main()
