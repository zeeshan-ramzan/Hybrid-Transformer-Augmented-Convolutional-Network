"""Label audit (Methods, 'Label Provenance and Ground-Truth Construction').

Flags every image whose out-of-fold probability for its assigned class (5-fold CV, script 50) is < 0.5 and
writes a review sheet for two independent annotators (columns reviewer1/reviewer2: ok / mislabelled /
ambiguous). With --reviewed, counts the images judged mislabelled or ambiguous by consensus.

Usage: python scripts/02_label_audit.py                       # after 50_cross_validation_5fold.py
       python scripts/02_label_audit.py --reviewed label_audit_sheet.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="runs/cv/out_of_fold_predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sheet", default="label_audit_sheet.csv")
    ap.add_argument("--reviewed", default="")
    a = ap.parse_args()
    if a.reviewed:
        r = pd.read_csv(a.reviewed)
        bad = r[(r.reviewer1 != "ok") & (r.reviewer2 != "ok")]
        print(f"flagged: {len(r)}; judged mislabelled/ambiguous by both reviewers: {len(bad)}")
        return
    oof = pd.read_csv(a.oof)
    flagged = oof[oof.p_true < a.threshold].assign(reviewer1="", reviewer2="")
    flagged.to_csv(a.sheet, index=False)
    print(f"{len(flagged)} images flagged (p_true < {a.threshold}) -> {a.sheet}")


if __name__ == "__main__":
    main()
