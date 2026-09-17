"""
make_synthetic_dataset.py
=========================
Creates a SYNTHETIC stand-in image dataset at

    <PROJECT_ROOT>/dataset/PlantVillage/<ClassName>/*.jpg
    <PROJECT_ROOT>/dataset/PlantDoc/<ClassName>/*.jpg      (--plantdoc, default on)

so that every script in code/ can be executed end-to-end *before* the real
PlantVillage archive is available.

WHAT THIS IS
------------
Each image is drawn procedurally: a leaf-shaped polygon with a midrib and veins
on a neutral background, then a class-specific lesion pattern painted on top --
scab spots, concentric black-rot rings, orange rust pustules, elongated blight
streaks, powdery-mildew speckle, mosaic mottling, leaf-scorch margins, and so
on -- plus per-image variation in leaf colour, size, orientation, lighting and
sensor noise.  The classes really are visually separable, so a classifier
trained on them produces meaningful (non-degenerate) accuracy, F1, confusion and
calibration output, and every figure and table in this package renders properly.

The PlantDoc stand-in reuses the same class list but adds the field-condition
nuisances the cross-dataset experiment is about: cluttered textured
backgrounds, oblique viewpoints, colour casts, blur and JPEG-scale noise.

**THIS IS SIMULATED DATA.  It is NOT PlantVillage, it is NOT PlantDoc, and no
number computed from it may be reported as a reproduction of the published
results.**  Delete `dataset/PlantVillage/` and drop in the real archive when you
have it (README section 3).

Usage
-----
    python make_synthetic_dataset.py                 # 8 classes x 20 images, 64 px
    python make_synthetic_dataset.py --classes 38 --per-class 60 --size 128
    python make_synthetic_dataset.py --no-plantdoc
    python make_synthetic_dataset.py --clean         # delete what it wrote before
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np

import config as C
from data_utils import banner

# --------------------------------------------------------------------------
#  Lesion "phenotypes".  Chosen per class from keywords in the class name so
#  that, e.g., every *_rust class really does get orange pustules.
# --------------------------------------------------------------------------
KEYWORD_KINDS: list[tuple[str, str]] = [
    ("healthy", "healthy"),
    ("scab", "spots_dark"),
    ("black_rot", "rings"),
    ("rust", "pustules"),
    ("powdery", "powder"),
    ("cercospora", "spots_gray"),
    ("northern_leaf_blight", "streaks"),
    ("leaf_blight", "streaks"),
    ("esca", "stripes"),
    ("haunglongbing", "mottle"),
    ("bacterial", "spots_wet"),
    ("early_blight", "rings"),
    ("late_blight", "blotch"),
    ("leaf_mold", "fuzz"),
    ("septoria", "spots_small"),
    ("spider_mites", "stipple"),
    ("target_spot", "rings"),
    ("curl", "curl"),
    ("mosaic", "mosaic"),
    ("scorch", "scorch"),
]
FALLBACK_KINDS = ["spots_dark", "rings", "pustules", "streaks", "blotch",
                  "powder", "mottle", "stipple"]

LEAF_HUES = {
    "Apple": (86, 132, 62), "Blueberry": (74, 126, 78), "Cherry": (96, 142, 70),
    "Corn": (110, 148, 58), "Grape": (78, 120, 60), "Orange": (66, 118, 66),
    "Peach": (100, 140, 66), "Pepper_bell": (72, 124, 68), "Potato": (88, 128, 64),
    "Raspberry": (94, 134, 74), "Soybean": (82, 130, 70), "Squash": (104, 146, 72),
    "Strawberry": (76, 118, 58), "Tomato": (92, 136, 66),
}


def kind_for(class_name: str, index: int) -> str:
    low = class_name.lower()
    for key, kind in KEYWORD_KINDS:
        if key in low:
            return kind
    return FALLBACK_KINDS[index % len(FALLBACK_KINDS)]


def hue_for(class_name: str) -> tuple[int, int, int]:
    genus = class_name.split("___")[0]
    return LEAF_HUES.get(genus, (88, 130, 66))


# --------------------------------------------------------------------------
#  Drawing
# --------------------------------------------------------------------------


def _leaf_polygon(cx, cy, rx, ry, angle, rng, lobes=5):
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    wobble = 1.0 + 0.13 * np.sin(lobes * t + rng.uniform(0, 6.28)) \
        + 0.05 * np.sin(2 * t + rng.uniform(0, 6.28))
    x = rx * wobble * np.cos(t)
    y = ry * wobble * np.sin(t)
    ca, sa = np.cos(angle), np.sin(angle)
    return [(float(cx + xi * ca - yi * sa), float(cy + xi * sa + yi * ca))
            for xi, yi in zip(x, y)]


def _shade(rgb, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in rgb)


def draw_leaf(size: int, class_name: str, kind: str, rng: np.random.Generator,
              field: bool = False):
    """Render one synthetic leaf image as a uint8 HxWx3 array."""
    from PIL import Image, ImageDraw, ImageFilter

    S = int(size * 2)                      # supersample, then downscale
    base = hue_for(class_name)
    if field:
        bg = np.zeros((S, S, 3), dtype=np.float32)
        for ch, m in enumerate((90, 96, 74)):
            bg[..., ch] = m + rng.normal(0, 26, (S, S))
        blob = rng.normal(0, 1, (S // 8, S // 8, 3))
        blob = np.asarray(Image.fromarray(
            np.clip(blob * 40 + 128, 0, 255).astype(np.uint8)).resize((S, S)),
            dtype=np.float32) - 128.0
        bg = np.clip(bg + blob * 0.9, 0, 255)
        img = Image.fromarray(bg.astype(np.uint8))
    else:
        g = int(rng.uniform(140, 190))
        img = Image.fromarray(np.clip(
            np.full((S, S, 3), g, dtype=np.float32) + rng.normal(0, 5, (S, S, 3)),
            0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")

    jitter = rng.normal(0, 12, 3)
    leaf = tuple(int(np.clip(b + j, 25, 235)) for b, j in zip(base, jitter))
    cx = S / 2 + rng.normal(0, S * (0.09 if field else 0.03))
    cy = S / 2 + rng.normal(0, S * (0.09 if field else 0.03))
    rx = S * rng.uniform(0.30, 0.40) * (rng.uniform(0.7, 1.15) if field else 1.0)
    ry = rx * rng.uniform(0.58, 0.86)
    ang = rng.uniform(0, np.pi)
    poly = _leaf_polygon(cx, cy, rx, ry, ang, rng)
    d.polygon(poly, fill=leaf + (255,), outline=_shade(leaf, 0.72) + (255,))

    # midrib + veins
    ca, sa = np.cos(ang), np.sin(ang)
    d.line([(cx - rx * ca, cy - rx * sa), (cx + rx * ca, cy + rx * sa)],
           fill=_shade(leaf, 1.30) + (210,), width=max(1, S // 90))
    for u in np.linspace(-0.72, 0.72, 9):
        px, py = cx + rx * u * ca, cy + rx * u * sa
        for s in (1, -1):
            ex = px + 0.55 * ry * (-sa) * s + 0.16 * rx * ca
            ey = py + 0.55 * ry * (ca) * s + 0.16 * rx * sa
            d.line([(px, py), (ex, ey)], fill=_shade(leaf, 1.18) + (150,),
                   width=max(1, S // 170))

    def rnd_pt(scale=0.62):
        while True:
            u, v = rng.uniform(-1, 1), rng.uniform(-1, 1)
            if u * u + v * v <= 1:
                x = cx + rx * scale * u * ca - ry * scale * v * sa
                y = cy + rx * scale * u * sa + ry * scale * v * ca
                return x, y

    # ---- class-specific lesions -----------------------------------------
    if kind == "spots_dark":
        for _ in range(rng.integers(10, 20)):
            x, y = rnd_pt(); r = S * rng.uniform(0.014, 0.036)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(58, 44, 26, 235))
    elif kind == "spots_gray":
        for _ in range(rng.integers(6, 12)):
            x, y = rnd_pt(); r = S * rng.uniform(0.02, 0.05)
            d.ellipse([x - r * 2.2, y - r * 0.7, x + r * 2.2, y + r * 0.7],
                      fill=(176, 172, 150, 225))
    elif kind == "spots_small":
        for _ in range(rng.integers(26, 46)):
            x, y = rnd_pt(); r = S * rng.uniform(0.006, 0.016)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(96, 78, 52, 230))
            d.ellipse([x - r * .5, y - r * .5, x + r * .5, y + r * .5], fill=(215, 210, 190, 220))
    elif kind == "spots_wet":
        for _ in range(rng.integers(14, 26)):
            x, y = rnd_pt(); r = S * rng.uniform(0.010, 0.026)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(46, 58, 38, 235))
            d.ellipse([x - r * 1.8, y - r * 1.8, x + r * 1.8, y + r * 1.8],
                      outline=(196, 190, 96, 130), width=max(1, S // 200))
    elif kind == "rings":
        for _ in range(rng.integers(3, 6)):
            x, y = rnd_pt(0.5); r = S * rng.uniform(0.05, 0.10)
            for j, f in enumerate(np.linspace(1.0, 0.25, 4)):
                d.ellipse([x - r * f, y - r * f, x + r * f, y + r * f],
                          outline=(72, 50, 30, 240) if j % 2 == 0 else (150, 118, 70, 220),
                          width=max(1, S // 150))
            d.ellipse([x - r * .22, y - r * .22, x + r * .22, y + r * .22],
                      fill=(60, 42, 26, 240))
    elif kind == "pustules":
        for _ in range(rng.integers(35, 70)):
            x, y = rnd_pt(); r = S * rng.uniform(0.006, 0.015)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(196, 108, 34, 245))
    elif kind == "streaks":
        for _ in range(rng.integers(3, 7)):
            x, y = rnd_pt(0.45)
            L = S * rng.uniform(0.16, 0.30); w = S * rng.uniform(0.014, 0.03)
            dx, dy = L * ca, L * sa
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(148, 132, 92, 235),
                   width=int(max(2, w)))
    elif kind == "stripes":
        for u in np.linspace(-0.6, 0.6, int(rng.integers(4, 8))):
            px, py = cx + rx * u * ca, cy + rx * u * sa
            d.line([(px - 0.5 * ry * sa, py + 0.5 * ry * ca),
                    (px + 0.5 * ry * sa, py - 0.5 * ry * ca)],
                   fill=(188, 168, 82, 220), width=int(max(2, S * 0.016)))
    elif kind == "blotch":
        for _ in range(rng.integers(1, 4)):
            x, y = rnd_pt(0.45); r = S * rng.uniform(0.10, 0.19)
            pts = _leaf_polygon(x, y, r, r * rng.uniform(0.6, 1.0),
                                rng.uniform(0, 3.14), rng, lobes=7)
            d.polygon(pts, fill=(64, 52, 40, 235))
    elif kind == "powder":
        for _ in range(rng.integers(120, 220)):
            x, y = rnd_pt(0.78); r = S * rng.uniform(0.004, 0.012)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(232, 232, 224, 190))
    elif kind == "fuzz":
        for _ in range(rng.integers(8, 16)):
            x, y = rnd_pt(); r = S * rng.uniform(0.025, 0.055)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(196, 186, 128, 165))
    elif kind == "mottle":
        for _ in range(rng.integers(10, 20)):
            x, y = rnd_pt(0.8); r = S * rng.uniform(0.035, 0.085)
            d.ellipse([x - r, y - r * .7, x + r, y + r * .7], fill=(206, 200, 96, 145))
    elif kind == "mosaic":
        for _ in range(rng.integers(16, 30)):
            x, y = rnd_pt(0.8); r = S * rng.uniform(0.02, 0.06)
            col = (210, 214, 120, 165) if rng.random() < 0.5 else (58, 96, 48, 165)
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    elif kind == "stipple":
        for _ in range(rng.integers(180, 320)):
            x, y = rnd_pt(0.85); r = S * rng.uniform(0.002, 0.007)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(190, 184, 118, 205))
    elif kind == "curl":
        d.polygon(_leaf_polygon(cx, cy, rx * 1.005, ry * 1.005, ang, rng),
                  outline=(212, 206, 96, 255), width=int(max(2, S * 0.024)))
        for _ in range(rng.integers(10, 20)):
            x, y = rnd_pt(0.9); r = S * rng.uniform(0.02, 0.05)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(206, 200, 100, 120))
    elif kind == "scorch":
        d.polygon(_leaf_polygon(cx, cy, rx * 1.005, ry * 1.005, ang, rng),
                  outline=(120, 66, 40, 255), width=int(max(2, S * 0.035)))
        for _ in range(rng.integers(4, 9)):
            x, y = rnd_pt(0.9); r = S * rng.uniform(0.03, 0.07)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(126, 74, 44, 190))
    # "healthy" -> no lesions at all

    # ---- global photometry ----------------------------------------------
    if field:
        img = img.rotate(rng.uniform(-25, 25), resample=Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 1.4)))
    a = np.asarray(img, dtype=np.float32)
    gain = rng.uniform(0.82, 1.18) if not field else rng.uniform(0.62, 1.35)
    cast = rng.normal(0, 6 if not field else 20, 3)
    a = a * gain + cast + rng.normal(0, 4.5 if not field else 11.0, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.LANCZOS)
    return np.asarray(img)


# --------------------------------------------------------------------------
#  Generation
# --------------------------------------------------------------------------


def generate(root, classes, per_class: int, size: int, seed: int,
             field: bool = False, quality: int = 92) -> int:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for ci, cname in enumerate(classes):
        kind = kind_for(cname, ci)
        d = root / cname
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + 1013 * ci)
        # mild class-size imbalance, imitating PlantVillage's 35:1 spread
        n = per_class
        if not field:
            real = C.PLANTVILLAGE_REAL_COUNTS.get(cname)
            if real is not None and per_class >= 8:
                scale = real / max(C.PLANTVILLAGE_REAL_COUNTS.values())
                n = int(round(per_class * (0.55 + 0.45 * scale ** 0.35)))
                n = max(6, n)
        for i in range(n):
            arr = draw_leaf(size, cname, kind, rng, field=field)
            Image.fromarray(arr).save(d / f"{'field' if field else 'pv'}_{i:04d}.jpg",
                                      quality=quality)
        total += n
        print(f"    {cname:<44s} {kind:<12s} {n:4d} images")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a SIMULATED leaf-image dataset.")
    ap.add_argument("--classes", type=int, default=8,
                    help="number of PlantVillage classes to simulate (max 38)")
    ap.add_argument("--per-class", type=int, default=20,
                    help="target images per class before the imbalance scaling")
    ap.add_argument("--size", type=int, default=64, help="image side in pixels")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--plantdoc", dest="plantdoc", action="store_true", default=True,
                    help="also generate the PlantDoc-style field set (default)")
    ap.add_argument("--no-plantdoc", dest="plantdoc", action="store_false")
    ap.add_argument("--plantdoc-per-class", type=int, default=6)
    ap.add_argument("--clean", action="store_true",
                    help="delete previously generated synthetic folders and exit")
    # accepted for uniformity with every other script in this package
    ap.add_argument("--quick", action="store_true",
                    help="even smaller set (6 classes, 12 images each, 48 px)")
    ap.add_argument("--epochs", type=int, default=None,
                    help=argparse.SUPPRESS)      # ignored: nothing is trained here
    ap.add_argument("--cpu", action="store_true",
                    help=argparse.SUPPRESS)      # ignored: nothing is trained here
    args = ap.parse_args()

    if args.quick:
        args.classes = min(args.classes, 6)
        args.per_class = min(args.per_class, 12)
        args.size = min(args.size, 48)
        args.plantdoc_per_class = min(args.plantdoc_per_class, 4)

    pv_root = C.DATASET_DIR / "PlantVillage"
    pd_root = C.DATASET_DIR / "PlantDoc"

    if args.clean:
        for r in (pv_root, pd_root):
            if r.exists():
                shutil.rmtree(r)
                print(f"  removed {r}")
        for m in (C.DATASET_DIR / ".synthetic", C.INDEX_CACHE_FILE):
            if m.exists():
                m.unlink()
        return

    banner("SYNTHETIC LEAF-IMAGE DATASET GENERATOR")
    print("  *** SIMULATED DATA -- procedurally drawn leaves, NOT PlantVillage. ***")
    print("  *** Results computed from it must never be reported as a          ***")
    print("  *** reproduction of the published numbers.                        ***\n")

    n_cls = max(2, min(int(args.classes), len(C.PLANTVILLAGE_CLASSES)))
    classes = C.PLANTVILLAGE_CLASSES[:n_cls]
    # keep at least one healthy class in small selections, for a realistic task
    if not any("healthy" in c for c in classes):
        classes[-1] = "Apple___healthy"

    print(f"  PlantVillage stand-in -> {pv_root}")
    n_pv = generate(pv_root, classes, args.per_class, args.size, args.seed)
    print(f"    total {n_pv} images in {len(classes)} classes\n")

    n_pd = 0
    if args.plantdoc:
        print(f"  PlantDoc stand-in (field conditions) -> {pd_root}")
        n_pd = generate(pd_root, classes, args.plantdoc_per_class, args.size,
                        args.seed + 7717, field=True, quality=72)
        print(f"    total {n_pd} images in {len(classes)} classes\n")

    (C.DATASET_DIR / ".synthetic").write_text(
        "Generated by code/make_synthetic_dataset.py -- SIMULATED data, not PlantVillage.\n")
    if C.INDEX_CACHE_FILE.exists():
        C.INDEX_CACHE_FILE.unlink()

    banner("DONE")
    print(f"  {n_pv} PlantVillage-style images, {n_pd} PlantDoc-style images, "
          f"{args.size}x{args.size} px")
    print("  Next:")
    print("      python run_all.py --quick      # smoke-test every script")
    print("      python hta_convnet.py --quick  # just the proposed architecture")
    print("\n  Remember: delete dataset/PlantVillage and dataset/PlantDoc "
          "(python make_synthetic_dataset.py --clean)")
    print("  before dropping in the real archives.")


if __name__ == "__main__":
    main()
