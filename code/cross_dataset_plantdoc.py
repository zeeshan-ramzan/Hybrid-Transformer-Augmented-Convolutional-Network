"""
cross_dataset_plantdoc.py -- paper Table 12
===========================================
Cross-dataset generalisation: train on PlantVillage (controlled laboratory
imagery), then evaluate on PlantDoc field photographs **with no fine-tuning at
all**, exactly as the paper specifies.

    Table 12  Cross-Dataset Generalization: PlantVillage -> PlantDoc
    ----------------------------------------------------------------
      ResNet-50                    72.14 %   71.38
      EfficientNet-B4              81.67 %   80.93
      ViT-B/16                     79.44 %   78.71
      Ahmad et al. (GLCM + SVM)    76.22 %   75.48
      HTA-ConvNet (Ours)           88.34 %   87.79

Each method is trained on the PlantVillage split and then run over the PlantDoc
images without any adaptation.  PlantDoc uses different folder names from
PlantVillage, so classes are matched by normalised name and only the
intersection is evaluated; the number of matched classes and images is reported
with every row, because a cross-dataset accuracy is meaningless without it.

Where to put PlantDoc
---------------------
    <PROJECT_ROOT>/dataset/PlantDoc/<ClassName>/*.jpg

(or a `plantdoc_dir` entry in `dataset/dataset_config.json`).  If it is absent,
`make_synthetic_dataset.py` writes a SIMULATED field-condition stand-in there.

Outputs -> results/cross_dataset/
    tables/table12_cross_dataset.csv     the reproduced table with paper values
    tables/plantdoc_class_coverage.csv   which classes matched, and their sizes
    tables/per_class_metrics.csv         per-class results on PlantDoc
    figures/table12_accuracy.png         in-domain vs cross-domain accuracy
    figures/plantdoc_confusion.png       normalised confusion matrix
    cross_dataset_results.json

Usage
-----
    python cross_dataset_plantdoc.py
    python cross_dataset_plantdoc.py --quick
    python cross_dataset_plantdoc.py --methods hta resnet50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import config as C
import data_utils as D
import hta_convnet as H
import nn_utils as N

RESULT_NAME = "cross_dataset"

# paper Table 12 rows -> how this package produces them
METHODS = {
    "resnet50": dict(display="ResNet-50", kind="tv"),
    "efficientnet_b4": dict(display="EfficientNet-B4", kind="tv"),
    "vit_b16": dict(display="ViT-B/16", kind="tv"),
    "ahmad": dict(display="GLCM + SVM (Ahmad et al.)", kind="classical"),
    "hta": dict(display="HTA-ConvNet (Ours)", kind="hta"),
}


def evaluate_on_plantdoc(model, pd_idx, rc, size, n_classes, forward=None) -> dict:
    loader = D.build_eval_loader(pd_idx, np.arange(len(pd_idx.files)), rc, size)
    return N.evaluate(model, loader, rc.device, n_classes, forward=forward,
                      return_arrays=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-dataset PlantVillage -> PlantDoc evaluation (Table 12).")
    D.cli_common(ap)
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"subset of {', '.join(METHODS)}")
    ap.add_argument("--attn-tokens", type=int, default=C.HTA["attn_max_tokens"])
    args = ap.parse_args()

    rc = D.resolve(args, default_epochs=C.HTA["epochs"])
    D.banner("CROSS-DATASET GENERALISATION: PlantVillage -> PlantDoc (Table 12)")
    D.describe_run(rc, "cross_dataset")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)

    pd_idx = D.load_plantdoc(classes=idx.classes)
    if pd_idx is None:
        raise SystemExit(
            "\n[!] No PlantDoc-style dataset found.\n"
            f"    Put field images in {C.DATASET_DIR / 'PlantDoc'}/<ClassName>/*.jpg\n"
            "    (folder names must match the PlantVillage class names so the two\n"
            "     label spaces can be aligned), or set 'plantdoc_dir' in\n"
            "     dataset/dataset_config.json.\n"
            "    PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset\n"
            "    For a SIMULATED stand-in run:  python make_synthetic_dataset.py")

    matched = sorted({idx.classes[int(l)] for l in pd_idx.labels})
    print(f"  matched {len(matched)}/{idx.n_classes} classes, "
          f"{len(pd_idx.files)} PlantDoc images "
          f"(the paper evaluates {C.PLANTDOC['paper_n_images']} of the "
          f"{C.PLANTDOC['paper_total']} in the full release)")

    out = C.result_dir(RESULT_NAME)
    cov = np.bincount(pd_idx.labels, minlength=idx.n_classes)
    D.save_table([{"Class": c, "PlantDoc images": int(cov[i]),
                   "In PlantVillage split": int((idx.labels == i).sum())}
                  for i, c in enumerate(idx.classes) if cov[i] > 0],
                 out / "tables" / "plantdoc_class_coverage",
                 "PlantDoc class coverage after label-space alignment", print_it=False)

    keys = list(METHODS) if not args.methods else [k for k in METHODS if k in args.methods]
    rows, payload, best = [], {}, None

    for key in keys:
        spec = METHODS[key]
        disp = spec["display"]
        D.banner(disp, "-")
        if spec["kind"] == "classical":
            res = _run_classical(idx, splits, pd_idx, rc)
        elif spec["kind"] == "hta":
            r = H.train_variant(idx, splits, rc, attn_tokens=args.attn_tokens,
                                label=disp)
            loaders_size = r["img_size"]
            res = dict(
                in_domain=r["test"],
                cross=evaluate_on_plantdoc(r["model"], pd_idx, rc, loaders_size,
                                           idx.n_classes))
            best = (disp, res["cross"], idx.classes)
            del r
        else:
            loaders, size = D.build_loaders(idx, splits, rc,
                                            min_size=C.DEEP_BASELINES[key]["min_size"])
            if key == "vit_b16" and size % 16:
                size = int(round(size / 16)) * 16
                loaders, size = D.build_loaders(idx, splits, rc, min_size=size)
            model, _ = N.build_torchvision_model(key, idx.n_classes, size, rc.pretrained)
            N.train_classifier(model, loaders, rc.device, dict(C.BASELINE_TRAIN),
                               idx.n_classes, idx.labels[splits["train"]],
                               epochs=rc.epochs, patience=rc.patience, tag=disp)
            res = dict(in_domain=N.evaluate(model, loaders["test"], rc.device,
                                            idx.n_classes, return_arrays=True),
                       cross=evaluate_on_plantdoc(model, pd_idx, rc, size,
                                                  idx.n_classes))
            del model

        paper = C.PAPER_TABLE12.get(disp, (None, None))
        rows.append({"Method": disp,
                     "PlantVillage Acc (%)": round(res["in_domain"]["accuracy"], 2),
                     "PlantDoc Acc (%)": round(res["cross"]["accuracy"], 2),
                     "PlantDoc F1 (%)": round(res["cross"]["f1_macro"], 2),
                     "Domain drop (pp)": round(res["in_domain"]["accuracy"] -
                                               res["cross"]["accuracy"], 2),
                     "Paper PlantDoc Acc (%)": paper[0],
                     "Paper PlantDoc F1 (%)": paper[1]})
        payload[key] = {
            "display": disp,
            "in_domain": {k: v for k, v in res["in_domain"].items()
                          if k not in ("y_true", "y_pred", "probs")},
            "plantdoc": {k: v for k, v in res["cross"].items()
                         if k not in ("y_true", "y_pred", "probs")},
            "paper": list(paper)}
        D.save_predictions(disp, res["cross"]["y_true"], res["cross"]["y_pred"],
                           res["cross"]["probs"], tag="plantdoc")

    D.save_table(rows, out / "tables" / "table12_cross_dataset",
                 "Table 12 -- cross-dataset generalisation PlantVillage -> PlantDoc")
    D.plot_grouped_bars([r["Method"] for r in rows],
                        {"PlantVillage (in-domain)": [r["PlantVillage Acc (%)"] for r in rows],
                         "PlantDoc (cross-domain)": [r["PlantDoc Acc (%)"] for r in rows]},
                        out / "figures" / "table12_accuracy.png",
                        "In-domain vs cross-domain accuracy (no fine-tuning)",
                        "Accuracy (%)", rotate=22)
    if best is not None:
        disp, cross, classes = best
        cm = D.confusion(cross["y_true"], cross["y_pred"], len(classes))
        D.plot_confusion(cm, out / "figures" / "plantdoc_confusion.png",
                         f"PlantDoc confusion matrix -- {disp}", classes)
        D.save_table(D.per_class_table(cross["y_true"], cross["y_pred"], classes),
                     out / "tables" / "per_class_metrics",
                     f"Per-class PlantDoc performance -- {disp}", print_it=False)

    D.save_json({"methods": payload,
                 "plantdoc": {"root": str(pd_idx.root), "n_images": len(pd_idx.files),
                              "matched_classes": matched,
                              "paper_n_images": C.PLANTDOC["paper_n_images"]},
                 "run": {"epochs": rc.epochs, "img_size": rc.img_size,
                         "seed": rc.seed, "quick": rc.quick}},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)}")


def _run_classical(idx, splits, pd_idx, rc) -> dict:
    """Ahmad et al.'s GLCM+HSV+SVM pipeline, trained on PlantVillage only."""
    import baseline_classical as B

    size = C.CLASSICAL["feature_size"] if not rc.quick else 48
    max_train = 400 if rc.quick else C.CLASSICAL["max_train"]
    tr = splits["train"]
    if len(tr) > max_train:
        rng = np.random.default_rng(rc.seed)
        tr = np.sort(rng.choice(tr, size=max_train, replace=False))
    print(f"    extracting GLCM+HSV features ({len(tr)} train, "
          f"{len(splits['test'])} test, {len(pd_idx.files)} PlantDoc) ...")
    Xtr = B.extract([idx.files[i] for i in tr], "glcm_hsv", size)
    Xte = B.extract([idx.files[i] for i in splits["test"]], "glcm_hsv", size)
    Xpd = B.extract(list(pd_idx.files), "glcm_hsv", size)
    clf = B.make_classifier("glcm_svm", rc.seed)
    clf.fit(Xtr, idx.labels[tr])

    def score(X, y):
        pred = clf.predict(X)
        proba = np.asarray(clf.predict_proba(X))
        full = np.zeros((len(y), idx.n_classes))
        for j, c in enumerate(clf[-1].classes_):
            full[:, int(c)] = proba[:, j]
        full = np.clip(full, 1e-9, 1.0)
        full /= full.sum(1, keepdims=True)
        m = D.classification_metrics(y, pred, full, idx.n_classes)
        m["y_true"], m["y_pred"], m["probs"] = y, pred, full
        return m

    return dict(in_domain=score(Xte, idx.labels[splits["test"]]),
                cross=score(Xpd, pd_idx.labels))


if __name__ == "__main__":
    main()
