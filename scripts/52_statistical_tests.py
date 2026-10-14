"""Statistical tests (Results/Discussion): paired McNemar test of HTA-ConvNet versus every other model on the
same test images, Wilson 95% interval of each single-run accuracy, and percentile bootstrap interval
(10,000 resamples).

Usage: python scripts/52_statistical_tests.py --reference runs/hta_convnet \
          --others runs/efficientnet_b4 runs/swin_t runs/convnext_t runs/ahmad2021 ...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.metrics import bootstrap_ci, mcnemar, wilson_ci
from common.utils import save_json


def load(d):
    z = np.load(os.path.join(d, "test_predictions.npz"), allow_pickle=True)
    return z["labels"], z["probs"].argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    y, pr = load(a.reference)
    res = {"reference": a.reference}
    for d in [a.reference] + a.others:
        yy, p = load(d)
        assert np.array_equal(yy, y), f"{d}: test images differ from the reference"
        k = int((p == y).sum())
        r = dict(acc=100 * k / len(y), correct=k, wilson95=wilson_ci(k, len(y)),
                 bootstrap95=bootstrap_ci(y, p, a.n_boot))
        if d != a.reference:
            chi2, pval, b, c = mcnemar(y, pr, p)
            r.update(mcnemar_chi2=chi2, mcnemar_p=pval, ref_right_other_wrong=b, ref_wrong_other_right=c)
        res[os.path.basename(d.rstrip("/"))] = r
        print(os.path.basename(d.rstrip("/")), r)
    save_json(res, "runs/statistical_tests.json")


if __name__ == "__main__":
    main()
