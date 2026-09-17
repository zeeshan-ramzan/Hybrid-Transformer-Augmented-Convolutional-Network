"""
config.py
=========
Central configuration for the reproduction of

    "Hybrid Transformer-Augmented Convolutional Network for Robust Plant Disease
     Recognition from Leaf Images"   (HTA-ConvNet)

Every path, protocol setting, hyper-parameter and paper-reported reference value
used anywhere in the project is defined here, so a single edit propagates to
every algorithm.

Directory layout assumed by ALL scripts
---------------------------------------
    <PROJECT_ROOT>/
        code/            <-- every .py file (this folder)
        dataset/         <-- YOU place PlantVillage (and optionally PlantDoc) here
        results/         <-- created automatically on first run

Nothing outside <PROJECT_ROOT> is ever read or written.

Where the paper contradicts itself the conflicting values are BOTH recorded
here, the default follows the majority reading, and README section 9 explains
the choice.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------------
CODE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = CODE_DIR.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

DATASET_CONFIG_FILE: Path = DATASET_DIR / "dataset_config.json"
INDEX_CACHE_FILE: Path = DATASET_DIR / ".hta_index_cache.json"

for _d in (DATASET_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def result_dir(name: str) -> Path:
    """results/<name>/{figures,tables}/ -- created on demand."""
    d = RESULTS_DIR / name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------
# 2. DATASET DEFINITION  (paper "Dataset Description", Tables 2 and 14)
# ----------------------------------------------------------------------------
PAPER_N_CLASSES: int = 38
PAPER_N_IMAGES: int = 54_309          # as claimed in the abstract and Table 2
PAPER_N_SPECIES: int = 14
PAPER_NATIVE_RES: int = 256           # PlantVillage images are 256 x 256

# Canonical PlantVillage (colour) folder names, in the order of paper Table 14.
# The synthetic generator uses exactly these names; the loader accepts any
# names at all and simply sorts them.
PLANTVILLAGE_CLASSES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_Black_Measles",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Image counts of the real, publicly distributed PlantVillage colour set.  Used
# only (a) by the synthetic generator to imitate the class-imbalance profile and
# (b) by data_utils to warn when the folder you supplied does not look like
# PlantVillage.  Their sum is 54,305, NOT the 54,309 the paper claims -- see
# README section 9.
PLANTVILLAGE_REAL_COUNTS: dict[str, int] = {
    "Apple___Apple_scab": 630, "Apple___Black_rot": 621,
    "Apple___Cedar_apple_rust": 275, "Apple___healthy": 1645,
    "Blueberry___healthy": 1502, "Cherry___Powdery_mildew": 1052,
    "Cherry___healthy": 854, "Corn___Cercospora_leaf_spot": 513,
    "Corn___Common_rust": 1192, "Corn___Northern_Leaf_Blight": 985,
    "Corn___healthy": 1162, "Grape___Black_rot": 1180,
    "Grape___Esca_Black_Measles": 1383, "Grape___Leaf_blight": 1076,
    "Grape___healthy": 423, "Orange___Haunglongbing": 5507,
    "Peach___Bacterial_spot": 2297, "Peach___healthy": 360,
    "Pepper_bell___Bacterial_spot": 997, "Pepper_bell___healthy": 1478,
    "Potato___Early_blight": 1000, "Potato___Late_blight": 1000,
    "Potato___healthy": 152, "Raspberry___healthy": 371,
    "Soybean___healthy": 5090, "Squash___Powdery_mildew": 1835,
    "Strawberry___Leaf_scorch": 1109, "Strawberry___healthy": 456,
    "Tomato___Bacterial_spot": 2127, "Tomato___Early_blight": 1000,
    "Tomato___Late_blight": 1909, "Tomato___Leaf_Mold": 952,
    "Tomato___Septoria_leaf_spot": 1771, "Tomato___Spider_mites": 1676,
    "Tomato___Target_Spot": 1404,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 5357,
    "Tomato___Tomato_mosaic_virus": 373, "Tomato___healthy": 1591,
}

# Paper Table 2 -- "PlantVillage Dataset: Selected Classes"
PAPER_TABLE2: list[tuple[str, str, int]] = [
    ("Apple", "Apple scab", 630), ("Apple", "Black rot", 621),
    ("Apple", "Cedar apple rust", 275), ("Apple", "Healthy", 1645),
    ("Corn (Maize)", "Common rust", 1192), ("Corn (Maize)", "Northern leaf blight", 985),
    ("Grape", "Black rot", 1180), ("Grape", "Esca", 1383),
    ("Orange", "Citrus greening", 5507), ("Potato", "Early blight", 1000),
    ("Potato", "Late blight", 1000), ("Tomato", "Bacterial spot", 2127),
    ("Tomato", "Late blight", 1909), ("Tomato", "YL curl virus", 5357),
    ("Tomato", "Healthy", 1591),
]

# ----------------------------------------------------------------------------
# 3. PREPROCESSING AND AUGMENTATION  (paper Table 3 + "Data Augmentation")
# ----------------------------------------------------------------------------
IMG_SIZE: int = 224                       # network input, paper Sec. "Image Quality"
RESIZE_SIZE: int = 256                    # centre-crop source size for val/test
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Table 3 is the authoritative augmentation table.  The prose in "Data
# Preprocessing and Augmentation" (p.6) and in "Data Augmentation Strategy"
# (p.11) disagree with it and with each other -- see README section 9.
AUGMENT = dict(
    hflip_p=0.50,                 # Table 3
    vflip_p=0.50,                 # Table 3 (prose p.6 says 0.30)
    rotation_deg=30.0,            # Table 3, p=1.00
    rrc_scale=(0.70, 1.00),       # prose p.6; Table 3 only gives the output size
    jitter_p=0.80,                # Table 3
    jitter_brightness=0.20,
    jitter_contrast=0.20,
    jitter_saturation=0.10,       # p.11 says 0.1, p.6 says 0.2
    jitter_hue=0.05,              # p.6 only
    blur_p=0.30,                  # Table 3 and p.11 (p.6 says 0.20)
    blur_sigma=(0.1, 2.0),
    blur_kernel=3,
    erasing_p=0.20,               # Table 3
    erasing_scale=(0.02, 0.33),
)

# ----------------------------------------------------------------------------
# 4. EVALUATION PROTOCOL
# ----------------------------------------------------------------------------
# CONFLICT.  "Data Partitioning" (p.6) says stratified 80/10/10 -> 43,447 /
# 5,431 / 5,431.  Table 5 and "Implementation Details" (p.13) both say
# 70/15/15 -> 38,017 / 8,146 / 8,146.  Two of the three statements say 70/15/15,
# so that is the default; --split 80/10/10 reproduces the other reading.
SPLIT_RATIOS: dict[str, tuple[float, float, float]] = {
    "70/15/15": (0.70, 0.15, 0.15),      # Table 5, Implementation Details
    "80/10/10": (0.80, 0.10, 0.10),      # Data Partitioning paragraph
}
DEFAULT_SPLIT: str = "70/15/15"
SPLIT_SEED: int = 42

N_FOLDS: int = 5                          # Table 4, stratified 5-fold CV
SEEDS: tuple[int, ...] = (42, 43, 44)     # Table 9
ECE_BINS: int = 15                        # "Calibration Analysis"

# ----------------------------------------------------------------------------
# 5. HTA-ConvNet HYPER-PARAMETERS  (Tables 6-8 + "Classification Head and
#    Training Protocol", "Hyperparameter Optimization", "Warm-Up ...")
# ----------------------------------------------------------------------------
HTA = dict(
    backbone="efficientnet_b4",   # Fig. 2 / Table 6: EfficientNet-B4, ImageNet-1k
    pretrained=False,             # --pretrained tries to download ImageNet weights
    dbsa_stages=(2, 3, 4),        # Table 6: DBSA after backbone stages 2, 3, 4
    n_heads=8,                    # Eq. (4), k = 8 (Fig. 8 optimum)
    dropout=0.40,                 # "Classification Head": p = 0.4 (Fig. 2 says 0.5)
    label_smoothing=0.10,         # Eq. (7), epsilon = 0.1
    mixup_alpha=0.20,             # "Regularization and Generalization Strategy"
    class_weighting="inv_sqrt",   # Eq. (7), w_c = 1 / sqrt(N_c)
    weighted_sampler=False,       # see README section 9 (double correction)
    optimizer="adam",             # Adam (Kingma & Ba)
    lr=1e-4,
    weight_decay=1e-5,            # excluded from bias and BatchNorm parameters
    warmup_epochs=5,              # linear 1e-6 -> 1e-4 over the first 5 epochs
    warmup_start_lr=1e-6,
    eta_min=5e-6,                 # cosine floor at epoch 50
    epochs=50,
    batch_size=32,
    early_stopping_patience=10,   # on validation loss
    grad_clip=0.0,                # not mentioned in the paper -> disabled
    attn_max_tokens=0,            # 0 = faithful full attention (see README 9)
)

# Table 6 -- EfficientNet-B4 stage configuration WITH DBSA insertion points, as
# printed in the paper.  The spatial sizes are right for EfficientNet-B4 at
# 224x224; the channel counts are not (see README section 9).
PAPER_TABLE6: list[dict] = [
    dict(stage="Stage 1", size="112x112", channels=48, dbsa=False),
    dict(stage="Stage 2", size="56x56", channels=88, dbsa=True),
    dict(stage="Stage 3", size="28x28", channels=120, dbsa=True),
    dict(stage="Stage 4", size="14x14", channels=208, dbsa=True),
    dict(stage="Stage 5 (Head)", size="7x7", channels=1792, dbsa=False),
]

# Table 7 -- per-stage DBSA cost as printed (MFLOPs).  Reproduced for reference
# only; hta_convnet.py recomputes Eqs. (5)-(6) and the two disagree by ~1-2
# orders of magnitude (README section 9).
PAPER_TABLE7: list[dict] = [
    dict(stage="Stage 2", n_s=3136, C_s=88, dwsc_mflops=3.1, mhsa_mflops=37.4),
    dict(stage="Stage 3", n_s=784, C_s=120, dwsc_mflops=1.1, mhsa_mflops=11.3),
    dict(stage="Stage 4", n_s=196, C_s=208, dwsc_mflops=0.9, mhsa_mflops=13.8),
]

# Table 8 -- architecture summary as printed
PAPER_TABLE8: list[dict] = [
    dict(block="Input", size="224x224x3", params_m=None, gflops=None),
    dict(block="Stage 1 (Conv)", size="112x112x48", params_m=0.32, gflops=0.41),
    dict(block="Stage 2 (MBConv)", size="56x56x88", params_m=1.11, gflops=0.55),
    dict(block="DBSA Block 1", size="56x56x88", params_m=0.02, gflops=0.04),
    dict(block="Stage 3 (MBConv)", size="28x28x120", params_m=2.58, gflops=0.71),
    dict(block="DBSA Block 2", size="28x28x120", params_m=0.04, gflops=0.05),
    dict(block="Stage 4 (MBConv)", size="14x14x208", params_m=4.37, gflops=0.83),
    dict(block="DBSA Block 3", size="14x14x208", params_m=0.07, gflops=0.06),
    dict(block="Stages 5-7 (MBConv)", size="7x7x448", params_m=8.91, gflops=0.42),
    dict(block="GAP + Dropout", size="448-dim", params_m=None, gflops=None),
    dict(block="FC + Softmax", size="38-class", params_m=0.17, gflops=None),
    dict(block="Total", size="", params_m=17.49, gflops=3.07),
]

# ----------------------------------------------------------------------------
# 6. BASELINES  (Table 10, Table 13, Table 15)
# ----------------------------------------------------------------------------
# Deep baselines share HTA-ConvNet's optimisation protocol ("the same split
# protocol and evaluation metrics are used throughout").
BASELINE_TRAIN = dict(
    optimizer="adam", lr=1e-4, weight_decay=1e-5, epochs=50, batch_size=32,
    warmup_epochs=5, warmup_start_lr=1e-6, eta_min=5e-6,
    early_stopping_patience=10, label_smoothing=0.10,
    mixup_alpha=0.0,              # regularisation cocktail is HTA-ConvNet's own
    class_weighting="inv_sqrt",
)

# torchvision constructor name, display name, minimum input size the graph needs
DEEP_BASELINES: dict[str, dict] = {
    "alexnet":         dict(display="AlexNet", tv="alexnet", min_size=64),
    "vgg16":           dict(display="VGG-16", tv="vgg16", min_size=32),
    "resnet50":        dict(display="ResNet-50", tv="resnet50", min_size=32),
    "inceptionv3":     dict(display="InceptionV3", tv="inception_v3", min_size=96),
    "densenet121":     dict(display="DenseNet-121", tv="densenet121", min_size=32),
    "mobilenetv2":     dict(display="MobileNetV2", tv="mobilenet_v2", min_size=32),
    "efficientnet_b4": dict(display="EfficientNet-B4", tv="efficientnet_b4", min_size=32),
    "vit_b16":         dict(display="ViT-B/16", tv="vit_b_16", min_size=32, patch=16),
    "plantdnet":       dict(display="PlantDNet", tv="resnet50", min_size=32),
}

# Table 15 -- classical feature pipelines
CLASSICAL_BASELINES: dict[str, dict] = {
    "glcm_svm":        dict(display="GLCM + SVM (Ahmad et al.)", features="glcm_hsv"),
    "colorhist_knn":   dict(display="Color Hist + kNN", features="color"),
    "lbp_svm":         dict(display="LBP + SVM", features="lbp"),
    "hog_rf":          dict(display="HOG + Random Forest", features="hog"),
    "fused_svm":       dict(display="Color+GLCM+HOG + SVM", features="fused"),
}
CLASSICAL = dict(
    feature_size=64,          # images are resampled to this before feature extraction
    glcm_levels=32,
    glcm_distances=(1,),
    glcm_angles_deg=(0, 45, 90, 135),
    hsv_bins=(8, 8, 8),
    lbp_points=8,
    hog_cell=8, hog_bins=9,
    svm_C=10.0, svm_gamma="scale",
    knn_k=5,
    rf_trees=300,
    max_train=6000,           # subsample above this many training images (speed)
)

# ----------------------------------------------------------------------------
# 7. EXPERIMENT GRIDS
# ----------------------------------------------------------------------------
# Table 11 -- ablation study
ABLATION_CONFIGS: list[dict] = [
    dict(key="backbone", label="Backbone only (EfficientNet-B4)",
         stages=(), branch="none"),
    dict(key="dwsc_s4", label="+ DWSC branch (Stage 4 only)",
         stages=(4,), branch="conv"),
    dict(key="mhsa_s4", label="+ MHSA branch (Stage 4 only)",
         stages=(4,), branch="attn"),
    dict(key="dbsa_s4", label="+ DBSA block (Stage 4 only)",
         stages=(4,), branch="both"),
    dict(key="dbsa_s34", label="+ DBSA blocks (Stages 3&4)",
         stages=(3, 4), branch="both"),
    dict(key="dbsa_s234", label="+ DBSA (Stages 2,3&4) [Full]",
         stages=(2, 3, 4), branch="both"),
]

# "Sensitivity to Number of DBSA Blocks"
DBSA_BLOCK_CONFIGS: list[dict] = [
    dict(key="none", label="No DBSA block", stages=()),
    dict(key="s4", label="Stage 4 only", stages=(4,)),
    dict(key="s34", label="Stages 3&4", stages=(3, 4)),
    dict(key="s234", label="Stages 2,3&4 (proposed)", stages=(2, 3, 4)),
    dict(key="s1234", label="Stages 1,2,3&4", stages=(1, 2, 3, 4)),
    dict(key="s12345", label="All five stages", stages=(1, 2, 3, 4, 5)),
]

HEAD_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16)                # Fig. 8
RESOLUTION_SWEEP: tuple[int, ...] = (96, 128, 160, 224, 256)  # Table 17

# ----------------------------------------------------------------------------
# 8. CROSS-DATASET EVALUATION  (Table 12)
# ----------------------------------------------------------------------------
PLANTDOC = dict(
    dir_names=("PlantDoc", "plantdoc", "PlantDoc-Dataset", "plantdoc_dataset"),
    paper_n_images=312,          # the subset actually evaluated
    paper_total=2569,            # the size of the full PlantDoc release
)

# ----------------------------------------------------------------------------
# 9. PAPER-REPORTED REFERENCE VALUES (never used as results, only for
#    side-by-side "paper vs this run" columns)
# ----------------------------------------------------------------------------
PAPER_TABLE10: dict[str, tuple[float, float, float, float]] = {
    # method: (Acc, Prec, Rec, F1)
    "AlexNet": (95.64, 95.31, 94.98, 95.14),
    "VGG-16": (96.43, 96.11, 95.87, 95.99),
    "ResNet-50": (97.28, 97.05, 96.84, 96.94),
    "InceptionV3": (97.84, 97.62, 97.41, 97.51),
    "DenseNet-121": (98.12, 97.93, 97.78, 97.85),
    "MobileNetV2": (97.53, 97.29, 97.14, 97.21),
    "EfficientNet-B4": (98.92, 98.74, 98.63, 98.68),
    "GLCM + SVM (Ahmad et al.)": (98.79, 98.51, 98.44, 98.47),
    "PlantDNet": (98.34, 98.12, 98.01, 98.06),
    "ViT-B/16": (98.67, 98.49, 98.38, 98.43),
    "HTA-ConvNet (Ours)": (99.61, 99.59, 99.57, 99.58),
}

PAPER_TABLE11: dict[str, tuple[float, float]] = {
    "backbone": (98.92, 98.68), "dwsc_s4": (99.11, 98.89),
    "mhsa_s4": (99.18, 98.97), "dbsa_s4": (99.27, 99.09),
    "dbsa_s34": (99.43, 99.26), "dbsa_s234": (99.61, 99.58),
}

PAPER_TABLE12: dict[str, tuple[float, float]] = {
    "ResNet-50": (72.14, 71.38), "EfficientNet-B4": (81.67, 80.93),
    "ViT-B/16": (79.44, 78.71), "GLCM + SVM (Ahmad et al.)": (76.22, 75.48),
    "HTA-ConvNet (Ours)": (88.34, 87.79),
}

PAPER_TABLE13: dict[str, tuple[float, float, int]] = {
    # method: (params M, GFLOPs, FPS on an A100)
    "VGG-16": (138.4, 15.5, 214), "ResNet-50": (25.6, 4.1, 289),
    "InceptionV3": (27.2, 5.7, 271), "DenseNet-121": (8.0, 2.9, 263),
    "MobileNetV2": (3.4, 0.3, 1124), "EfficientNet-B4": (19.3, 4.4, 319),
    "ViT-B/16": (86.6, 17.6, 181), "HTA-ConvNet (Ours)": (19.4, 4.6, 312),
}

PAPER_TABLE15: dict[str, tuple[str, float]] = {
    "GLCM + SVM (Ahmad et al.)": ("Texture (GLCM)", 98.79),
    "Color Hist + kNN": ("Color (RGB)", 87.32),
    "LBP + SVM": ("Texture (LBP)", 89.14),
    "HOG + Random Forest": ("Shape (HOG)", 91.47),
    "Color+GLCM+HOG + SVM": ("Hybrid", 93.58),
    "HTA-ConvNet (Ours)": ("End-to-End Deep", 99.61),
}

# Table 16 -- literature progression (static; no code reproduces these)
PAPER_TABLE16: list[dict] = [
    dict(method="Mohanty et al.", year=2016, acc=99.35, venue="Front. Plant Sci."),
    dict(method="Ferentinos", year=2018, acc=99.53, venue="Comput. Electron. Agric."),
    dict(method="Too et al.", year=2019, acc=99.30, venue="Comput. Electron. Agric."),
    dict(method="Ahmad et al.", year=2021, acc=98.79, venue="Wireless Pers. Commun."),
    dict(method="Wang et al.", year=2021, acc=99.28, venue="IEEE Access"),
    dict(method="Yang et al.", year=2021, acc=99.41, venue="IEEE Access"),
    dict(method="Ahmed et al.", year=2023, acc=99.48, venue="Appl. Sci."),
    dict(method="HTA-ConvNet (Ours)", year=2025, acc=99.61, venue="IEEE Access"),
]

PAPER_TABLE17: dict[int, tuple[float, int, int]] = {
    # resolution: (Acc, FPS on A100, FPS on Raspberry Pi 4)
    96: (98.14, 1247, 132), 128: (98.83, 867, 91), 160: (99.21, 612, 64),
    224: (99.61, 312, 34), 256: (99.64, 241, 26),
}

PAPER_FIG8: dict[int, float] = {1: 99.27, 2: 99.41, 4: 99.52, 8: 99.61, 16: 99.56}

PAPER_TABLE4: list[dict] = [
    dict(fold=1, acc=99.61, f1=99.58, auc=0.9997),
    dict(fold=2, acc=99.54, f1=99.51, auc=0.9996),
    dict(fold=3, acc=99.65, f1=99.62, auc=0.9998),
    dict(fold=4, acc=99.52, f1=99.49, auc=0.9995),
    dict(fold=5, acc=99.58, f1=99.55, auc=0.9997),
]

PAPER_TABLE9: dict[int, tuple[float, float, float]] = {
    42: (99.61, 99.58, 4.2), 43: (99.59, 99.55, 4.1), 44: (99.63, 99.60, 4.3),
}

# Table 14 -- per-class precision / recall / F1 (paper values)
PAPER_TABLE14: list[tuple[str, float, float, float]] = [
    ("Apple---Apple Scab", 99.81, 99.83, 99.82),
    ("Apple---Black Rot", 99.72, 99.76, 99.74),
    ("Apple---Cedar Apple Rust", 99.86, 99.90, 99.88),
    ("Apple---Healthy", 99.69, 99.73, 99.71),
    ("Blueberry---Healthy", 99.63, 99.67, 99.65),
    ("Cherry---Powdery Mildew", 99.54, 99.58, 99.56),
    ("Cherry---Healthy", 99.76, 99.80, 99.78),
    ("Corn---Cercospora Leaf Spot", 99.61, 99.65, 99.63),
    ("Corn---Common Rust", 99.89, 99.93, 99.91),
    ("Corn---Northern Leaf Blight", 99.44, 99.50, 99.47),
    ("Corn---Healthy", 99.55, 99.61, 99.58),
    ("Grape---Black Rot", 99.80, 99.84, 99.82),
    ("Grape---Esca (Black Measles)", 99.71, 99.75, 99.73),
    ("Grape---Leaf Blight", 99.59, 99.63, 99.61),
    ("Grape---Healthy", 99.35, 99.41, 99.38),
    ("Orange---Haunglongbing", 99.67, 99.71, 99.69),
    ("Peach---Bacterial Spot", 99.82, 99.86, 99.84),
    ("Peach---Healthy", 99.49, 99.53, 99.51),
    ("Pepper---Bacterial Spot", 99.60, 99.64, 99.62),
    ("Pepper---Healthy", 99.75, 99.79, 99.77),
    ("Potato---Early Blight", 99.40, 99.46, 99.43),
    ("Potato---Late Blight", 99.66, 99.70, 99.68),
    ("Potato---Healthy", 99.80, 99.84, 99.82),
    ("Raspberry---Healthy", 99.53, 99.57, 99.55),
    ("Soybean---Healthy", 99.69, 99.73, 99.71),
    ("Squash---Powdery Mildew", 99.37, 99.41, 99.39),
    ("Strawberry---Leaf Scorch", 99.81, 99.85, 99.83),
    ("Strawberry---Healthy", 99.56, 99.60, 99.58),
    ("Tomato---Bacterial Spot", 99.65, 99.69, 99.67),
    ("Tomato---Early Blight", 99.72, 99.76, 99.74),
    ("Tomato---Late Blight", 99.05, 99.11, 99.08),
    ("Tomato---Leaf Mold", 98.99, 99.05, 99.02),
    ("Tomato---Septoria Leaf Spot", 99.26, 99.32, 99.29),
    ("Tomato---Spider Mites", 99.45, 99.51, 99.48),
    ("Tomato---Target Spot", 99.60, 99.64, 99.62),
    ("Tomato---Yellow Leaf Curl Virus", 99.73, 99.77, 99.75),
    ("Tomato---Mosaic Virus", 99.79, 99.83, 99.81),
    ("Tomato---Healthy", 99.52, 99.56, 99.54),
]

# Miscellaneous scalars quoted in the text
PAPER_MISC = dict(
    ece_hta=0.0087, ece_backbone=0.0193, ece_vit=0.0341,
    gate_stage2=(0.67, 0.08), gate_stage3=(0.51, 0.09), gate_stage4=(0.38, 0.07),
    mcnemar_chi2_vs_effb4=47.3, mcnemar_chi2_vs_ahmad=61.8,
    micro_auc=0.9997, params_m=19.4, gflops=4.6, fps_a100=312,
    train_hours=4.2, test_set_size_table5=8146, test_set_size_mcnemar=8147,
)

# ----------------------------------------------------------------------------
# 10. RUNTIME
# ----------------------------------------------------------------------------
NUM_WORKERS: int = int(os.environ.get("HTA_WORKERS", "0"))
USE_AMP: bool = True              # torch.amp -- automatically a no-op on CPU
FIG_DPI: int = 200
SEED: int = 42

# --quick schedule: deliberately tiny so a full smoke test finishes in minutes
QUICK = dict(img_size=64, epochs=1, batch_size=16, max_per_class=20,
             max_classes=8, folds=2, patience=2)

PALETTE = {
    "primary": "#4c8fd1", "secondary": "#d9694f", "accent": "#5aa469",
    "muted": "#8a7fc2", "warn": "#e0a458", "grey": "#8b8b8b",
}

__all__ = [n for n in dir() if not n.startswith("_")]
