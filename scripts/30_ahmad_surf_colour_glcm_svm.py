"""Re-implementation of Ahmad et al. (Wireless Pers. Commun. 2021): two-stage classical pipeline
(Tables 'main', 'cross', 'classical').

Stage 1: healthy vs diseased from a SURF bag-of-visual-words histogram + SVM.
Stage 2: class identification from 6 colour + 22 GLCM texture features with a multi-class RBF-kernel SVM
         (one SVM over disease classes for leaves predicted diseased, one over healthy classes otherwise),
         C and gamma chosen by grid search on the validation set.
Note: SURF is patented and absent from standard OpenCV wheels; SIFT is used automatically if
opencv-contrib with the non-free module is not installed (reported in results.json).

Usage: python scripts/30_ahmad_surf_colour_glcm_svm.py --data-root data/PlantVillage/color \
           [--plantdoc-root data/PlantDoc-Dataset] [--pp2020-root data/plant-pathology-2020]
"""
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.classical import classical_parser, grid_search, one_hot_probs
from common.data import load_plantdoc, load_pp2020, load_split, pv_classes
from common.handcrafted import bovw_hist, colour6, descriptors, glcm22, keypoint_detector, read
from common.metrics import classification_metrics
from common.utils import save_json

GRID = {"C": [1, 10, 100], "gamma": [0.001, 0.01, 0.1]}


def _desc(path):
    det, _ = keypoint_detector()
    return descriptors(read(path), det)


def _cg(path):
    img = read(path)
    return np.concatenate([colour6(img), glcm22(img)])


class AhmadPipeline:
    def __init__(self, classes, k_words=200, jobs=-1):
        self.classes, self.k, self.jobs = classes, k_words, jobs
        self.healthy = np.array(["healthy" in c.lower() for c in classes])

    def feats(self, paths):
        d = Parallel(n_jobs=self.jobs, batch_size=32)(delayed(_desc)(p) for p in paths)
        cg = np.stack(Parallel(n_jobs=self.jobs, batch_size=32)(delayed(_cg)(p) for p in paths))
        return d, cg

    def fit(self, tr, va):
        dtr, cgtr = self.feats(tr.path.tolist())
        dva, cgva = self.feats(va.path.tolist())
        sample = np.vstack([x for x in dtr if len(x)])
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), min(len(sample), 200000), replace=False)]
        self.km = MiniBatchKMeans(self.k, random_state=0, batch_size=4096, n_init=3).fit(sample.astype(np.float32))
        btr = np.stack([bovw_hist(x, self.km) for x in dtr])
        bva = np.stack([bovw_hist(x, self.km) for x in dva])
        ytr, yva = tr.label.values, va.label.values
        htr, hva = self.healthy[ytr].astype(int), self.healthy[yva].astype(int)
        self.sc1 = StandardScaler().fit(btr)
        p1, _ = grid_search(lambda C, gamma: SVC(C=C, gamma=gamma), GRID, self.sc1.transform(btr), htr,
                            self.sc1.transform(bva), hva)
        self.svm1 = SVC(**p1).fit(self.sc1.transform(btr), htr)
        self.sc2 = StandardScaler().fit(cgtr)
        self.svm2, self.params = {}, {"stage1": p1}
        for flag in (0, 1):
            mtr, mva = htr == flag, hva == flag
            p, _ = grid_search(lambda C, gamma: SVC(C=C, gamma=gamma), GRID, self.sc2.transform(cgtr[mtr]),
                               ytr[mtr], self.sc2.transform(cgva[mva]), yva[mva])
            self.svm2[flag] = SVC(**p).fit(self.sc2.transform(cgtr[mtr]), ytr[mtr])
            self.params["stage2_" + ("healthy" if flag else "diseased")] = p
        return self

    def predict(self, paths):
        d, cg = self.feats(paths)
        b = self.sc1.transform(np.stack([bovw_hist(x, self.km) for x in d]))
        h = self.svm1.predict(b)
        out = np.zeros(len(paths), int)
        for flag in (0, 1):
            m = h == flag
            if m.any():
                out[m] = self.svm2[flag].predict(self.sc2.transform(cg[m]))
        return out


def main():
    p = classical_parser("Ahmad et al. 2021 re-implementation")
    p.add_argument("--k-words", type=int, default=200)
    args = p.parse_args()
    t0 = time.time()
    classes = pv_classes(args.data_root)
    df = load_split(args.data_root, args.split_csv, args.seed)
    if args.max_per_split:
        df = df.sample(frac=1.0, random_state=0).groupby("split").head(args.max_per_split)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    model = AhmadPipeline(classes, args.k_words, args.jobs).fit(tr, va)
    pred = model.predict(te.path.tolist())
    _, det_name = keypoint_detector()
    res = dict(model="ahmad2021", keypoints=det_name, params=model.params,
               test=classification_metrics(te.label.values, one_hot_probs(pred, len(classes))))
    if args.plantdoc_root:
        fd = load_plantdoc(args.plantdoc_root, "test", classes)
        res["plantdoc"] = classification_metrics(fd.label.values, one_hot_probs(model.predict(fd.path.tolist()),
                                                                                len(classes)))
    if args.pp2020_root:
        fd = load_pp2020(args.pp2020_root, classes)
        res["pp2020"] = classification_metrics(fd.label.values, one_hot_probs(model.predict(fd.path.tolist()),
                                                                              len(classes)))
    res["minutes"] = (time.time() - t0) / 60
    out = os.path.join(args.out_dir, "ahmad2021")
    save_json(res, os.path.join(out, "results.json"))
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), probs=one_hot_probs(pred, len(classes)),
                        labels=te.label.values)
    print(f"[ahmad2021] TEST acc {res['test']['acc']:.2f} F1 {res['test']['f1']:.2f} ({det_name})")


if __name__ == "__main__":
    main()
