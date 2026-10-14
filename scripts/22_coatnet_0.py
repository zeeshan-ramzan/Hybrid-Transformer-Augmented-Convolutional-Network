"""CoAtNet-0 (Dai et al., NeurIPS 2021) [NEW in revision, R3.4/R3.7].

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').
timm weights: coatnet_0_rw_224.sw_in1k

Usage:
    python scripts/22_coatnet_0.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.models import build_model
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("CoAtNet-0 (Dai et al., NeurIPS 2021)")
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: build_model("coatnet_0", n_classes, args.dropout, pretrained=not args.no_pretrained), args, "coatnet_0",
                   extra={"arch": "coatnet_0"})


if __name__ == "__main__":
    main()
