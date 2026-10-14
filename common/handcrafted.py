"""Hand-crafted features for the classical baselines (Table 'classical').

Images are resized to 256x256 (the PlantVillage resolution) before feature extraction.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, hog, local_binary_pattern

SIZE = 256


def read(path, size=SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)  # BGR uint8


# ---------------------------------------------------------------- colour (6 features)
def colour6(img_bgr):
    """Mean and standard deviation of the R, G, B channels."""
    rgb = img_bgr[..., ::-1].reshape(-1, 3).astype(np.float64) / 255.0
    return np.concatenate([rgb.mean(0), rgb.std(0)])


def colour_hist(img_bgr, bins=32):
    """32-bin-per-channel RGB histogram (normalised)."""
    h = [cv2.calcHist([img_bgr], [c], None, [bins], [0, 256]).ravel() for c in (2, 1, 0)]
    h = np.concatenate(h)
    return h / max(h.sum(), 1)


# ---------------------------------------------------------------- GLCM (22 features)
def glcm22(img_bgr, distances=(1,), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), levels=64):
    """22 GLCM texture features (Haralick / Soh & Tsatsoulis / Clausi set), averaged over the 4 angles."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = (g.astype(np.int32) * levels // 256).astype(np.uint8)
    P = graycomatrix(g, distances, angles, levels=levels, symmetric=True, normed=True)
    feats = []
    for a in range(P.shape[3]):
        feats.append(_glcm_feats(P[:, :, 0, a]))
    return np.mean(feats, axis=0)


def _glcm_feats(p):
    eps = 1e-12
    L = p.shape[0]
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    px, py = p.sum(1), p.sum(0)
    mx, my = (np.arange(L) * px).sum(), (np.arange(L) * py).sum()
    sx = np.sqrt(((np.arange(L) - mx) ** 2 * px).sum())
    sy = np.sqrt(((np.arange(L) - my) ** 2 * py).sum())
    pxy_sum = np.array([p[(i + j) == k].sum() for k in range(2 * L - 1)])
    pxy_diff = np.array([p[np.abs(i - j) == k].sum() for k in range(L)])
    ks, kd = np.arange(2 * L - 1), np.arange(L)
    hxy = -(p * np.log(p + eps)).sum()
    hx, hy = -(px * np.log(px + eps)).sum(), -(py * np.log(py + eps)).sum()
    hxy1 = -(p * np.log(px[:, None] * py[None, :] + eps)).sum()
    hxy2 = -(px[:, None] * py[None, :] * np.log(px[:, None] * py[None, :] + eps)).sum()
    asm = (p ** 2).sum()
    contrast = (p * (i - j) ** 2).sum()
    corr = ((p * (i - mx) * (j - my)).sum()) / (sx * sy + eps)
    sum_avg = (ks * pxy_sum).sum()
    sum_ent = -(pxy_sum * np.log(pxy_sum + eps)).sum()
    diff_avg = (kd * pxy_diff).sum()
    return np.array([
        asm,                                              # 1 angular second moment
        np.sqrt(asm),                                     # 2 energy
        contrast,                                         # 3 contrast
        corr,                                             # 4 correlation
        (p * np.abs(i - j)).sum(),                        # 5 dissimilarity
        (p / (1 + (i - j) ** 2)).sum(),                   # 6 homogeneity (IDM)
        (p / (1 + np.abs(i - j))).sum(),                  # 7 inverse difference
        hxy,                                              # 8 entropy
        p.max(),                                          # 9 maximum probability
        (p * (i - mx) ** 2).sum(),                        # 10 sum of squares: variance
        (p * i * j).sum(),                                # 11 autocorrelation
        ((i + j - mx - my) ** 3 * p).sum(),               # 12 cluster shade
        ((i + j - mx - my) ** 4 * p).sum(),               # 13 cluster prominence
        sum_avg,                                          # 14 sum average
        ((ks - sum_avg) ** 2 * pxy_sum).sum(),            # 15 sum variance
        sum_ent,                                          # 16 sum entropy
        ((kd - diff_avg) ** 2 * pxy_diff).sum(),          # 17 difference variance
        -(pxy_diff * np.log(pxy_diff + eps)).sum(),       # 18 difference entropy
        (hxy - hxy1) / (max(hx, hy) + eps),               # 19 information measure of correlation 1
        np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - hxy)))), # 20 information measure of correlation 2
        (p / (1 + np.abs(i - j) / L)).sum(),              # 21 inverse difference normalised
        (p / (1 + (i - j) ** 2 / L ** 2)).sum(),          # 22 inverse difference moment normalised
    ])


# ---------------------------------------------------------------- LBP / HOG
def lbp_hist(img_bgr, P=8, R=1):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(g, P, R, method="uniform")
    h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
    return h / max(h.sum(), 1)


def hog_feat(img_bgr, size=128):
    g = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2GRAY)
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


# ---------------------------------------------------------------- SURF bag of visual words
def keypoint_detector():
    """SURF if the OpenCV build has the non-free module, otherwise SIFT (documented substitution)."""
    try:
        return cv2.xfeatures2d.SURF_create(400), "SURF"
    except Exception:
        return cv2.SIFT_create(), "SIFT (SURF unavailable in this OpenCV build)"


def descriptors(img_bgr, det):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, d = det.detectAndCompute(g, None)
    return d if d is not None else np.zeros((0, det.descriptorSize()), np.float32)


def bovw_hist(desc, kmeans):
    k = kmeans.n_clusters
    if len(desc) == 0:
        return np.zeros(k)
    h = np.bincount(kmeans.predict(desc.astype(np.float32)), minlength=k).astype(float)
    return h / h.sum()
