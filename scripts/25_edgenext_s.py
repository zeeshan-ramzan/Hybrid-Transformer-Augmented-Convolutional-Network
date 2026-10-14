"""EdgeNeXt-S (Maaz et al., ECCVW 2022) [NEW in revision, R3.4].

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').
timm weights: edgenext_small.usi_in1k

Usage:
    python scripts/25_edgenext_s.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("EdgeNeXt-S (Maaz et al., ECCVW 2022)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("edgenext_s", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "edgenext_s",
                   extra={"arch": "edgenext_s"})


if __name__ == "__main__":
    main()
