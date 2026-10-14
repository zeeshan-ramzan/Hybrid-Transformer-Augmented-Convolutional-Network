"""PlantXViT (Thakur et al., Ecol. Inform. 2023) [NEW in revision, R3.7], trained from scratch.

Re-trained under the identical protocol of the manuscript (Tables 'baselines', 'main', 'cross', 'complexity').
No ImageNet weights exist; trained from scratch (default 100 epochs, set with --epochs).

Usage:
    python scripts/26_plantxvit.py --data-root data/PlantVillage/color --lr-grid 5e-5,1e-4,3e-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.plantxvit import PlantXViT
from common.trainer import run_experiment
from common.utils import base_parser


def main():
    p = base_parser("PlantXViT (Thakur et al., Ecol. Inform. 2023)")
    p.set_defaults(epochs=100, lr=1e-4)
    args = p.parse_args()
    n_classes = len(pv_classes(args.data_root))
    run_experiment(lambda: PlantXViT(num_classes=n_classes, img_size=args.img_size, dropout=args.dropout), args, "plantxvit",
                   extra={"arch": "plantxvit"})


if __name__ == "__main__":
    main()
