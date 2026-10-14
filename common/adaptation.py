"""Target-domain data for the adaptation experiments (Table 'adapt', NEW in revision, R3.3).

PlantDoc : adaptation pool = mapped images of the official TRAIN split; evaluation = mapped TEST images.
PP2020   : the 1,730 single-label images are split 80/20 (stratified, seed 42); adaptation pool = 80%,
           evaluation = held-out 20%.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.data import load_plantdoc, load_pp2020


def target_splits(dataset: str, root: str, classes, seed: int = 42):
    if dataset == "plantdoc":
        pool = load_plantdoc(root, "train", classes)
        test = load_plantdoc(root, "test", classes)
    elif dataset == "pp2020":
        df = load_pp2020(root, classes)
        pool, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    else:
        raise ValueError(dataset)
    return pool.reset_index(drop=True), test.reset_index(drop=True)


def few_shot(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in pool.groupby("label"):
        parts.append(g.iloc[rng.permutation(len(g))[:k]])
    return pd.concat(parts).reset_index(drop=True)


def inner_val(df: pd.DataFrame, frac=0.15, seed=42):
    """Validation subset of the target training data for early stopping."""
    counts = df.label.value_counts()
    strat = df.label if counts.min() >= 2 else None
    tr, va = train_test_split(df, test_size=frac, stratify=strat, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)
