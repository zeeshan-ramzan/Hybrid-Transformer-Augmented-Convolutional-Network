"""Datasets, splits, transforms and label mapping (Methods section of the manuscript)."""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

# The 38 PlantVillage classes (folder names of the public release, alphabetical order).
PV_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# Table "Label mapping" of the manuscript: PlantDoc class -> PlantVillage class.
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Peach leaf": "Peach___healthy",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Strawberry leaf": "Strawberry___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato leaf": "Tomato___healthy",
}

# Plant Pathology 2020 (train.csv columns) -> PlantVillage class.
PP2020_TO_PV = {"healthy": "Apple___healthy", "scab": "Apple___Apple_scab", "rust": "Apple___Cedar_apple_rust"}

CROP_OF = lambda pv_name: pv_name.split("___")[0]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ----------------------------------------------------------------------------- transforms
def train_transform(img_size: int = 224):
    """Online augmentation of Table 'augment' (identical for all models)."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(30),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.33)),
    ])


def eval_transform(img_size: int = 224):
    resize = int(round(img_size * 256 / 224))
    return T.Compose([T.Resize(resize), T.CenterCrop(img_size), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ----------------------------------------------------------------------------- datasets
class ImageListDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None, return_index=False):
        self.paths, self.labels = list(paths), list(labels)
        self.transform, self.return_index = transform, return_index

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_index:
            return img, self.labels[i], i
        return img, self.labels[i]


def scan_imagefolder(root: str, classes: Optional[List[str]] = None) -> pd.DataFrame:
    if classes is None:
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    rows = []
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"class folder missing: {d}")
        for f in sorted(os.listdir(d)):
            if f.endswith(IMG_EXT):
                rows.append((os.path.join(d, f), ci, c))
    return pd.DataFrame(rows, columns=["path", "label", "class_name"])


def pv_classes(root: str) -> List[str]:
    found = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if set(found) == set(PV_CLASSES):
        return PV_CLASSES
    return found  # allows small debug subsets


def make_stratified_split(root: str, out_csv: str, seed: int = 42, val=0.15, test=0.15) -> pd.DataFrame:
    """Stratified 70/15/15 split with a fixed seed (identical indices for every model)."""
    df = scan_imagefolder(root, pv_classes(root))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    tr_idx, rest_idx = next(sss.split(df.path, df.label))
    rest = df.iloc[rest_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    va_rel, te_rel = next(sss2.split(rest.path, rest.label))
    df["split"] = "train"
    df.loc[rest.index[va_rel], "split"] = "val"
    df.loc[rest.index[te_rel], "split"] = "test"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_split(root: str, split_csv: str, seed: int = 42) -> pd.DataFrame:
    if not os.path.exists(split_csv):
        print(f"[data] split file {split_csv} not found -> creating stratified 70/15/15 split (seed {seed})")
        return make_stratified_split(root, split_csv, seed)
    return pd.read_csv(split_csv)


def class_counts(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    return np.bincount(df.label.values, minlength=n_classes).astype(np.float64)


def make_loaders(df: pd.DataFrame, n_classes: int, img_size=224, batch_size=32, workers=8,
                 weighted_sampler=True, seed=42):
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ds_tr = ImageListDataset(tr.path, tr.label, train_transform(img_size))
    ds_va = ImageListDataset(va.path, va.label, eval_transform(img_size))
    ds_te = ImageListDataset(te.path, te.label, eval_transform(img_size))
    g = torch.Generator().manual_seed(seed)
    if weighted_sampler:
        cnt = class_counts(tr, n_classes)
        w = (1.0 / np.maximum(cnt, 1))[tr.label.values]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True, generator=g)
        dl_tr = DataLoader(ds_tr, batch_size, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size, shuffle=True, generator=g, num_workers=workers, pin_memory=True,
                           drop_last=True)
    kw = dict(batch_size=batch_size * 2, shuffle=False, num_workers=workers, pin_memory=True)
    return dl_tr, DataLoader(ds_va, **kw), DataLoader(ds_te, **kw), tr


# ----------------------------------------------------------------------------- field datasets
def load_plantdoc(root: str, split: str = "test", pv_class_list: List[str] = PV_CLASSES) -> pd.DataFrame:
    """PlantDoc-Dataset/{train,test}/<class>/*.jpg -> mapped PlantVillage labels (unmapped classes dropped)."""
    pv_idx = {c: i for i, c in enumerate(pv_class_list)}
    norm_map = {_norm(k): v for k, v in PLANTDOC_TO_PV.items()}
    base = os.path.join(root, split)
    rows, skipped = [], []
    for c in sorted(os.listdir(base)):
        d = os.path.join(base, c)
        if not os.path.isdir(d):
            continue
        pv = norm_map.get(_norm(c))
        if pv is None or pv not in pv_idx:
            skipped.append(c)
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(IMG_EXT):
                rows.append((os.path.join(d, f), pv_idx[pv], c, CROP_OF(pv)))
    if skipped:
        print(f"[PlantDoc] classes without PlantVillage counterpart (excluded): {skipped}")
    return pd.DataFrame(rows, columns=["path", "label", "source_class", "crop"])


def load_pp2020(root: str, pv_class_list: List[str] = PV_CLASSES, include_multiple=False) -> pd.DataFrame:
    """Plant Pathology 2020 (Kaggle FGVC7): root/train.csv + root/images/<image_id>.jpg."""
    pv_idx = {c: i for i, c in enumerate(pv_class_list)}
    csv = pd.read_csv(os.path.join(root, "train.csv"))
    rows = []
    for _, r in csv.iterrows():
        path = os.path.join(root, "images", r.image_id + ".jpg")
        if r.multiple_diseases == 1:
            if include_multiple:
                rows.append((path, -1, "multiple_diseases"))
            continue
        for k, pv in PP2020_TO_PV.items():
            if r[k] == 1:
                rows.append((path, pv_idx[pv], k))
    return pd.DataFrame(rows, columns=["path", "label", "source_class"])


def simple_loader(df: pd.DataFrame, img_size=224, batch_size=64, workers=8, train=False, return_index=False):
    tf = train_transform(img_size) if train else eval_transform(img_size)
    ds = ImageListDataset(df.path, df.label, tf, return_index=return_index)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=True,
                      drop_last=train and len(ds) > batch_size)
