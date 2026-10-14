"""Grad-CAM (Selvaraju et al. 2017) from the last DBSA-augmented stage (Results, 'Gradient-Weighted Class
Activation Mapping'). Saves overlays for up to N random test images per class, for HTA-ConvNet and,
optionally, the EfficientNet-B4 baseline (target layer = output of the stage-4 group, features[5]).

Usage: python scripts/61_gradcam.py --run runs/hta_convnet --baseline-run runs/efficientnet_b4 --per-class 100
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES, eval_transform, load_split
from common.loading import load_run
from common.utils import get_device


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls=None):
        self.model.zero_grad()
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        out.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True)
        return (cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-8)).detach().cpu().numpy(), cls


def target_layer(model, info):
    if "hta_config" in info:
        return model.blocks[max(model.blocks.keys(), key=int)]  # last DBSA block (Stage 4)
    return model.features[5]  # torchvision EfficientNet-B4: end of stage 4


def overlay(path, cam, size=224):
    img = np.array(Image.open(path).convert("RGB").resize((256, 256)))
    o = (256 - size) // 2
    img = img[o:o + size, o:o + size]
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * heat).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline-run", default="")
    ap.add_argument("--data-root", default="data/PlantVillage/color")
    ap.add_argument("--split-csv", default="splits/standard_split.csv")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--out", default="runs/gradcam")
    a = ap.parse_args()
    dev = get_device()
    test = load_split(a.data_root, a.split_csv)
    test = test[test.split == "test"]
    tf = eval_transform(224)
    for run in [r for r in (a.run, a.baseline_run) if r]:
        model, info = load_run(run, len(PV_CLASSES), dev)
        cam = GradCAM(model, target_layer(model, info))
        for c, g in test.groupby("label"):
            g = g.sample(min(len(g), a.per_class), random_state=0)
            d = os.path.join(a.out, info["model"], PV_CLASSES[c])
            os.makedirs(d, exist_ok=True)
            for p in g.path:
                x = tf(Image.open(p).convert("RGB"))[None].to(dev)
                m, pred = cam(x)
                Image.fromarray(overlay(p, m[0])).save(
                    os.path.join(d, f"{os.path.splitext(os.path.basename(p))[0]}_pred{int(pred)}.jpg"))
        print("saved", os.path.join(a.out, info["model"]))


if __name__ == "__main__":
    main()
