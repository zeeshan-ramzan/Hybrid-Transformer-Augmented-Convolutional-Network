"""Classical baseline: uniform rotation-invariant LBP (P=8, R=1) + RBF-kernel SVM (Table 'classical').

Usage: python scripts/32_lbp_svm.py --data-root data/PlantVillage/color
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.svm import SVC
from common.handcrafted import lbp_hist
from common.classical import classical_parser, run_classical


def features(img):
    return lbp_hist(img, 8, 1)


def main():
    args = classical_parser("lbp_svm").parse_args()
    run_classical(args, "lbp_svm", features, lambda C, gamma: SVC(C=C, gamma=gamma, kernel="rbf"), {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]})


if __name__ == "__main__":
    main()
