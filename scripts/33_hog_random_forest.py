"""Classical baseline: HOG (9 orientations, 8x8 cells, 2x2 blocks) + 500-tree random forest (Table 'classical').

Usage: python scripts/33_hog_random_forest.py --data-root data/PlantVillage/color
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from common.handcrafted import hog_feat
from common.classical import classical_parser, run_classical


def features(img):
    return hog_feat(img)


def main():
    args = classical_parser("hog_rf").parse_args()
    run_classical(args, "hog_rf", features, lambda max_features: RandomForestClassifier(500, max_features=max_features, n_jobs=-1, random_state=42), {"max_features": ["sqrt", 0.05]})


if __name__ == "__main__":
    main()
