"""
baseline_classical.py
=====================
The handcrafted feature-based methods of paper Table 15, including the primary
comparison baseline of the whole paper -- Ahmad et al. [2]'s GLCM + HSV colour
histogram + multi-class SVM pipeline, which also supplies the "Ahmad et al."
row of Table 10 and the "Ahmad et al." row of the Table 12 cross-dataset test.

    Table 15  Comparison with Classical Feature-Based Methods
    ------------------------------------------------------------------
      GLCM + SVM (Ahmad et al.)    Texture (GLCM)        98.79 %
      Color Hist + kNN             Color (RGB)           87.32 %
      LBP + SVM                    Texture (LBP)         89.14 %
      HOG + Random Forest          Shape (HOG)           91.47 %
      Color+GLCM+HOG + SVM         Hybrid                93.58 %
      HTA-ConvNet (Ours)           End-to-End Deep       99.61 %

Descriptors are implemented here from first principles with NumPy, so no
scikit-image / OpenCV dependency is required:

    GLCM   gray-level co-occurrence matrix over `glcm_levels` quantisation
           levels at distance 1 and angles 0, 45, 90 and 135 degrees, reduced to
           five Haralick statistics per angle -- angular second moment (energy),
           contrast, correlation, homogeneity (inverse difference moment) and
           entropy.                                     -> `glcm_features`
    HSV    joint 8x8x8 hue/saturation/value colour histogram.  -> `color_features`
    LBP    rotation-agnostic 8-neighbour local binary pattern histogram
           (256 bins, L1-normalised).                    -> `lbp_features`
    HOG    histogram of oriented gradients, 9 unsigned orientation bins over
           8x8 pixel cells, block-normalised.            -> `hog_features`

Ahmad et al.'s pipeline is GLCM at four angles combined with HSV colour
histograms in a multi-class SVM, which is exactly the `glcm_hsv` configuration.

One execution evaluates all five pipelines on the same stratified split the
neural methods use and writes one result folder per pipeline plus a joint
Table 15 into `results/classical/`.

Usage
-----
    python baseline_classical.py
    python baseline_classical.py --quick
    python baseline_classical.py --only glcm_svm
    python baseline_classical.py --max-train 3000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import config as C
import data_utils as D

RESULT_NAME = "classical"
AHMAD = "GLCM + SVM (Ahmad et al.)"


# ============================================================================
#  1. FEATURE EXTRACTORS
# ============================================================================


def _to_gray(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def glcm_features(gray: np.ndarray, levels: int = 32,
                  angles=(0, 45, 90, 135)) -> np.ndarray:
    """Five Haralick statistics per angle from the gray-level co-occurrence matrix.

    The GLCM captures second-order statistics by counting how often pairs of
    gray levels occur at a given offset; the statistics extracted here are the
    ones Ahmad et al. use -- energy, contrast, correlation, homogeneity and
    entropy.
    """
    q = np.clip((gray / 256.0 * levels).astype(np.int64), 0, levels - 1)
    offsets = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}
    feats: list[float] = []
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    for a in angles:
        dy, dx = offsets[a]
        ys = slice(max(0, -dy), q.shape[0] - max(0, dy))
        xs = slice(max(0, -dx), q.shape[1] - max(0, dx))
        ys2 = slice(max(0, dy), q.shape[0] + min(0, dy))
        xs2 = slice(max(0, dx), q.shape[1] + min(0, dx))
        a1, a2 = q[ys, xs].ravel(), q[ys2, xs2].ravel()
        if a1.size == 0:
            feats += [0.0] * 5
            continue
        m = np.bincount(a1 * levels + a2, minlength=levels * levels) \
            .reshape(levels, levels).astype(np.float64)
        m = m + m.T                                # symmetric GLCM
        s = m.sum()
        p = m / s if s > 0 else m
        mu_i = float((p * i_idx).sum())
        mu_j = float((p * j_idx).sum())
        sd_i = float(np.sqrt(((i_idx - mu_i) ** 2 * p).sum()))
        sd_j = float(np.sqrt(((j_idx - mu_j) ** 2 * p).sum()))
        energy = float((p ** 2).sum())
        contrast = float((p * (i_idx - j_idx) ** 2).sum())
        corr = (float((p * (i_idx - mu_i) * (j_idx - mu_j)).sum()) / (sd_i * sd_j)
                if sd_i > 1e-9 and sd_j > 1e-9 else 0.0)
        homog = float((p / (1.0 + (i_idx - j_idx) ** 2)).sum())
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        feats += [energy, contrast, corr, homog, entropy]
    return np.asarray(feats, dtype=np.float32)


def color_features(img: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """Joint HSV colour histogram, L1-normalised."""
    from matplotlib.colors import rgb_to_hsv

    hsv = rgb_to_hsv(np.clip(img / 255.0, 0, 1))
    idx = np.zeros(hsv.shape[:2], dtype=np.int64)
    mult = 1
    for c in range(3):
        b = bins[c]
        q = np.clip((hsv[..., c] * b).astype(np.int64), 0, b - 1)
        idx += q * mult
        mult *= b
    h = np.bincount(idx.ravel(), minlength=mult).astype(np.float32)
    return h / max(h.sum(), 1.0)


def lbp_features(gray: np.ndarray) -> np.ndarray:
    """8-neighbour local binary pattern histogram (256 bins), L1-normalised."""
    g = gray
    c = g[1:-1, 1:-1]
    nb = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:],
          g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2]]
    code = np.zeros(c.shape, dtype=np.int64)
    for k, n in enumerate(nb):
        code |= ((n >= c).astype(np.int64) << k)
    h = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return h / max(h.sum(), 1.0)


def hog_features(gray: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients with L2-hys style cell normalisation."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins       # unsigned, [0, bins)
    b = np.clip(ang.astype(np.int64), 0, bins - 1)
    H, W = gray.shape
    ny, nx = H // cell, W // cell
    out = np.zeros((ny, nx, bins), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            mm = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            bb = b[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            out[i, j] = np.bincount(bb, weights=mm, minlength=bins)
    out = out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-6)
    return out.ravel()


def extract(files, kind: str, size: int) -> np.ndarray:
    """Feature matrix for a list of image paths."""
    from PIL import Image

    rows = []
    for k, p in enumerate(files):
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR),
                             dtype=np.float32)
        gray = _to_gray(img)
        if kind == "glcm_hsv":
            f = np.concatenate([glcm_features(gray, C.CLASSICAL["glcm_levels"],
                                              C.CLASSICAL["glcm_angles_deg"]),
                                color_features(img, C.CLASSICAL["hsv_bins"])])
        elif kind == "color":
            f = color_features(img, C.CLASSICAL["hsv_bins"])
        elif kind == "lbp":
            f = lbp_features(gray)
        elif kind == "hog":
            f = hog_features(gray, C.CLASSICAL["hog_cell"], C.CLASSICAL["hog_bins"])
        elif kind == "fused":
            f = np.concatenate([color_features(img, C.CLASSICAL["hsv_bins"]),
                                glcm_features(gray, C.CLASSICAL["glcm_levels"],
                                              C.CLASSICAL["glcm_angles_deg"]),
                                hog_features(gray, C.CLASSICAL["hog_cell"],
                                             C.CLASSICAL["hog_bins"])])
        else:                                                # pragma: no cover
            raise ValueError(kind)
        rows.append(f.astype(np.float32))
        if (k + 1) % 500 == 0:
            print(f"      {k + 1}/{len(files)} images", flush=True)
    return np.stack(rows) if rows else np.zeros((0, 1), dtype=np.float32)


# ============================================================================
#  2. CLASSIFIERS
# ============================================================================


def make_classifier(key: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if key.endswith("_svm"):
        return make_pipeline(StandardScaler(),
                             SVC(C=C.CLASSICAL["svm_C"], kernel="rbf",
                                 gamma=C.CLASSICAL["svm_gamma"], probability=True,
                                 random_state=seed))
    if key.endswith("_knn"):
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=C.CLASSICAL["knn_k"]))
    if key.endswith("_rf"):
        return RandomForestClassifier(n_estimators=C.CLASSICAL["rf_trees"],
                                      n_jobs=-1, random_state=seed)
    raise ValueError(key)                                    # pragma: no cover


def run_one(key: str, spec: dict, idx, splits, size: int, seed: int,
            max_train: int, feature_cache: dict) -> dict:
    display = spec["display"]
    D.banner(f"{display}", "-")
    kind = spec["features"]

    tr = splits["train"]
    if len(tr) > max_train:
        rng = np.random.default_rng(seed)
        tr = np.sort(rng.choice(tr, size=max_train, replace=False))
        print(f"  training set sub-sampled to {max_train} images for tractability")

    t0 = time.perf_counter()
    for name, ids in (("train", tr), ("test", splits["test"]), ("val", splits["val"])):
        ck = (kind, name, len(ids))
        if ck not in feature_cache:
            print(f"    extracting {kind} features on {name} ({len(ids)} images) ...")
            feature_cache[ck] = extract([idx.files[i] for i in ids], kind, size)
    Xtr = feature_cache[(kind, "train", len(tr))]
    Xte = feature_cache[(kind, "test", len(splits["test"]))]
    ytr, yte = idx.labels[tr], idx.labels[splits["test"]]
    print(f"    feature dimension: {Xtr.shape[1]}")

    clf = make_classifier(key, seed)
    clf.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    pred = clf.predict(Xte)
    if hasattr(clf, "predict_proba"):
        probs = np.asarray(clf.predict_proba(Xte), dtype=np.float64)
        full = np.zeros((len(yte), idx.n_classes))
        cls = clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_
        for j, c in enumerate(cls):
            full[:, int(c)] = probs[:, j]
        probs = np.clip(full, 1e-9, 1.0)
        probs /= probs.sum(1, keepdims=True)
    else:                                                    # pragma: no cover
        probs = np.eye(idx.n_classes)[pred]

    m = D.classification_metrics(yte, pred, probs, idx.n_classes)
    m["y_true"], m["y_pred"], m["probs"] = yte, pred, probs
    m["fit_time_s"] = fit_s
    m["feature_dim"] = int(Xtr.shape[1])

    import nn_utils as N

    N.report_classification(display, D.slug(key).lower(), idx.classes, m,
                            history=None,
                            paper_row=C.PAPER_TABLE10.get(display),
                            extra={"features": kind, "feature_dim": int(Xtr.shape[1]),
                                   "fit_time_s": fit_s,
                                   "n_train": int(len(tr)),
                                   "paper_table15": C.PAPER_TABLE15.get(display)})
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classical handcrafted-feature baselines (paper Table 15).")
    D.cli_common(ap)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these pipelines "
                         f"({', '.join(C.CLASSICAL_BASELINES)})")
    ap.add_argument("--max-train", type=int, default=C.CLASSICAL["max_train"],
                    help="cap the number of training images (SVMs scale poorly)")
    ap.add_argument("--feature-size", type=int, default=C.CLASSICAL["feature_size"])
    args = ap.parse_args()

    rc = D.resolve(args)
    D.banner("CLASSICAL FEATURE-BASED BASELINES (paper Table 15)")
    D.describe_run(rc, "classical")
    D.set_seed(rc.seed)

    idx = D.load_index(max_per_class=rc.max_per_class, max_classes=rc.max_classes)
    splits = D.make_splits(idx, rc.split, C.SPLIT_SEED)
    size = args.feature_size if not rc.quick else min(args.feature_size, 48)
    max_train = args.max_train if not rc.quick else 400
    print(f"  descriptors computed on {size}x{size} images")

    keys = list(C.CLASSICAL_BASELINES) if not args.only else [
        k for k in C.CLASSICAL_BASELINES if k in args.only]
    cache: dict = {}
    results: dict[str, dict] = {}
    for k in keys:
        results[k] = run_one(k, C.CLASSICAL_BASELINES[k], idx, splits, size,
                             rc.seed, max_train, cache)

    out = C.result_dir(RESULT_NAME)
    rows = []
    for k in keys:
        disp = C.CLASSICAL_BASELINES[k]["display"]
        m = results[k]
        rows.append({"Method": disp,
                     "Features": C.PAPER_TABLE15[disp][0],
                     "Acc (%)": round(m["accuracy"], 2),
                     "Prec (%)": round(m["precision_macro"], 2),
                     "Rec (%)": round(m["recall_macro"], 2),
                     "F1 (%)": round(m["f1_macro"], 2),
                     "Feature dim": m["feature_dim"],
                     "Fit time (s)": round(m["fit_time_s"], 1),
                     "Paper Acc (%)": C.PAPER_TABLE15[disp][1]})
    hta = D.read_result("hta_convnet")
    if hta:
        rows.append({"Method": "HTA-ConvNet (Ours)", "Features": "End-to-End Deep",
                     "Acc (%)": round(hta["test"]["accuracy"], 2),
                     "Prec (%)": round(hta["test"]["precision_macro"], 2),
                     "Rec (%)": round(hta["test"]["recall_macro"], 2),
                     "F1 (%)": round(hta["test"]["f1_macro"], 2),
                     "Feature dim": "-", "Fit time (s)": "-",
                     "Paper Acc (%)": C.PAPER_TABLE15["HTA-ConvNet (Ours)"][1]})
    D.save_table(rows, out / "tables" / "table15_classical_comparison",
                 "Table 15 -- comparison with classical feature-based methods")
    D.plot_bars([r["Method"] for r in rows], [r["Acc (%)"] for r in rows],
                out / "figures" / "table15_accuracy.png",
                "Classical feature pipelines vs HTA-ConvNet", "Accuracy (%)",
                highlight="HTA-ConvNet", rotate=25)
    D.save_json({k: {kk: vv for kk, vv in v.items()
                     if kk not in ("y_true", "y_pred", "probs")}
                 for k, v in results.items()},
                out / f"{RESULT_NAME}_results.json")

    D.banner("DONE")
    print(f"All outputs -> {out.relative_to(C.PROJECT_ROOT)} and results/<pipeline>/")


if __name__ == "__main__":
    main()
