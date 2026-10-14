"""Classical baseline: 32-bin-per-channel RGB colour histogram + k-NN (k tuned, default 5) (Table 'classical').

Usage: python scripts/31_colorhist_knn.py --data-root data/PlantVillage/color
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.neighbors import KNeighborsClassifier
from common.handcrafted import colour_hist
from common.classical import classical_parser, run_classical


def features(img):
    return colour_hist(img, 32)


def main():
    args = classical_parser("colorhist_knn").parse_args()
    run_classical(args, "colorhist_knn", features, lambda k: KNeighborsClassifier(n_neighbors=k, n_jobs=-1), {"k": [3, 5, 7, 9]})


if __name__ == "__main__":
    main()
