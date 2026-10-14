"""Evaluation metrics and statistics used in the manuscript."""
import math
from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)


def classification_metrics(y_true, probs) -> Dict[str, float]:
    y_pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = dict(acc=100 * accuracy_score(y_true, y_pred), prec=100 * p, rec=100 * r, f1=100 * f,
               n=int(len(y_true)), correct=int((y_pred == y_true).sum()), ece=ece(y_true, probs))
    return out


def per_class_metrics(y_true, probs, n_classes):
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    correct = [int(((y_pred == c) & (y_true == c)).sum()) for c in range(n_classes)]
    return [dict(idx=c + 1, n=int(s[c]), correct=correct[c], prec=100 * p[c], rec=100 * r[c], f1=100 * f[c])
            for c in range(n_classes)]


def ece(y_true, probs, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def auc_ovr(y_true, probs, classes=None):
    n = probs.shape[1]
    Y = np.eye(n)[y_true]
    res = {}
    for c in (classes if classes is not None else range(n)):
        if Y[:, c].min() != Y[:, c].max():
            res[int(c)] = float(roc_auc_score(Y[:, c], probs[:, c]))
    present = Y.sum(0) > 0
    res["micro"] = float(roc_auc_score(Y[:, present].ravel(), probs[:, present].ravel()))
    return res


def roc_points(y_true, probs, c):
    fpr, tpr, _ = roc_curve((y_true == c).astype(int), probs[:, c])
    return fpr, tpr


def wilson_ci(k: int, n: int, z: float = 1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def bootstrap_ci(y_true, y_pred, n_boot=10000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (y_true == y_pred).astype(float)
    accs = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return 100 * np.quantile(accs, alpha / 2), 100 * np.quantile(accs, 1 - alpha / 2)


def t_ci(values, conf=0.95):
    v = np.asarray(values, float)
    m, s = v.mean(), v.std(ddof=1)
    h = stats.t.ppf((1 + conf) / 2, len(v) - 1) * s / math.sqrt(len(v))
    return m, s, m - h, m + h


def mcnemar(y_true, pred_a, pred_b, continuity=True):
    """Paired McNemar test. Returns chi2, p, b (A right/B wrong), c (A wrong/B right)."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - (1 if continuity else 0)) ** 2 / (b + c)
    return float(chi2), float(stats.chi2.sf(chi2, 1)), b, c


def confusion(y_true, y_pred, n):
    return confusion_matrix(y_true, y_pred, labels=list(range(n)))


def risk_coverage(y_true, confidence, y_pred):
    """Risk-coverage curve and AURC (selective classification)."""
    order = np.argsort(-confidence)
    err = (y_pred[order] != y_true[order]).astype(float)
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk, float(np.mean(risk))


def entropy(probs):
    return -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
