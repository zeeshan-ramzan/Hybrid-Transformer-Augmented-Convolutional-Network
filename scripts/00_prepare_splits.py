"""Stratified 70/15/15 PlantVillage split (seed 42) shared by every model (Methods, 'Data Partitioning').

Usage: python scripts/00_prepare_splits.py --data-root data/PlantVillage/color --out splits/standard_split.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import make_stratified_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/PlantVillage/color")
    p.add_argument("--out", default="splits/standard_split.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    df = make_stratified_split(a.data_root, a.out, a.seed)
    print(df.groupby("split").size().to_string())
    print(f"classes: {df.label.nunique()}  images: {len(df)}  -> {a.out}")


if __name__ == "__main__":
    main()
