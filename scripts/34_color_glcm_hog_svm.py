"""Classical baseline: concatenated colour + GLCM + HOG features + RBF-kernel SVM (Table 'classical').

Usage: python scripts/34_color_glcm_hog_svm.py --data-root data/PlantVillage/color
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.svm import SVC
from common.handcrafted import colour6, colour_hist, glcm22, hog_feat
from common.classical import classical_parser, run_classical


def features(img):
    return np.concatenate([colour6(img), colour_hist(img, 16), glcm22(img), hog_feat(img)])


def main():
    args = classical_parser("color_glcm_hog_svm").parse_args()
    run_classical(args, "color_glcm_hog_svm", features, lambda C, gamma: SVC(C=C, gamma=gamma, kernel="rbf"), {"C": [1, 10, 100], "gamma": ["scale", 0.001]})


if __name__ == "__main__":
    main()
