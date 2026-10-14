"""Baseline model factory (all models end with dropout(p=0.4) + 38-way linear layer, ImageNet-1k init).

Sources of pre-trained weights:
  torchvision : AlexNet, VGG-16, ResNet-50, InceptionV3, DenseNet-121, MobileNetV2, EfficientNet-B4, ViT-B/16
  timm        : Swin-T, ConvNeXt-T, CoAtNet-0, EdgeNeXt-S
  HuggingFace : CvT-13 (microsoft/cvt-13)
  official repo (microsoft/Focal-Transformer) : Focal-T  -> see README (clone + checkpoint path)
  PlantXViT   : re-implemented in plantxvit.py, trained from scratch
"""
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tvm


class HeadWrapper(nn.Module):
    """Feature extractor + dropout + linear head."""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone, self.dropout, self.fc = backbone, nn.Dropout(dropout), nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        return self.fc(self.dropout(f))


def _tv(name, num_classes, dropout, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if name == "alexnet":
        m = tvm.alexnet(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(4096, num_classes))
    elif name == "vgg16":
        m = tvm.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(4096, num_classes))
    elif name == "resnet50":
        m = tvm.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    elif name == "inception_v3":
        m = tvm.inception_v3(weights="IMAGENET1K_V1" if pretrained else None, aux_logits=True)
        m.aux_logits, m.AuxLogits = False, None  # 224x224 input, auxiliary head disabled
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, num_classes))
    elif name == "densenet121":
        m = tvm.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, num_classes))
    elif name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
    elif name == "efficientnet_b4":
        m = tvm.efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1792, num_classes))
    elif name == "vit_b16":
        m = tvm.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        m.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, num_classes))
    else:
        raise KeyError(name)
    return m


TIMM_NAMES = {
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
    "convnext_t": "convnext_tiny.fb_in1k",
    "coatnet_0": "coatnet_0_rw_224.sw_in1k",
    "edgenext_s": "edgenext_small.usi_in1k",
}


def _timm(name, num_classes, dropout, pretrained=True):
    import timm
    bb = timm.create_model(TIMM_NAMES[name], pretrained=pretrained, num_classes=0)  # pooled features
    return HeadWrapper(bb, bb.num_features, num_classes, dropout)


class _HFCvT(nn.Module):
    def __init__(self, num_classes, dropout, pretrained=True):
        super().__init__()
        from transformers import CvtConfig, CvtModel
        self.m = CvtModel.from_pretrained("microsoft/cvt-13") if pretrained else CvtModel(CvtConfig())
        dim = self.m.config.embed_dim[-1]
        self.norm = nn.LayerNorm(dim)
        self.dropout, self.fc = nn.Dropout(dropout), nn.Linear(dim, num_classes)

    def forward(self, x):
        out = self.m(pixel_values=x)
        cls = out.cls_token_value[:, 0] if out.cls_token_value is not None else out.last_hidden_state.mean((2, 3))
        return self.fc(self.dropout(self.norm(cls)))


def _focal_t(num_classes, dropout, pretrained=True):
    """Focal Transformer (Yang et al., NeurIPS 2021) from the official implementation.

    git clone https://github.com/microsoft/Focal-Transformer  ;  set FOCAL_REPO=<path>/Focal-Transformer
    download the Focal-T ImageNet-1k checkpoint from the repo README ; set FOCAL_T_CKPT=<path>.pth
    """
    repo = os.environ.get("FOCAL_REPO")
    if not repo or not os.path.isdir(repo):
        raise RuntimeError("Focal-T: set FOCAL_REPO to a clone of https://github.com/microsoft/Focal-Transformer")
    cls_dir = os.path.join(repo, "classification")
    sys.path.insert(0, cls_dir)
    import config as focal_config                       # official yacs config of the repo
    from models import build_model as focal_build       # official model factory
    cfg_file = os.environ.get("FOCAL_T_CFG", os.path.join(cls_dir, "configs", "focal_tiny_patch4_window7_224.yaml"))
    cfg = focal_config._C.clone()
    cfg.defrost()
    focal_config._update_config_from_file(cfg, cfg_file)
    cfg.freeze()
    m = focal_build(cfg)
    if pretrained:
        ck = os.environ.get("FOCAL_T_CKPT")
        if not ck:
            raise RuntimeError("Focal-T: set FOCAL_T_CKPT to the ImageNet-1k Focal-T checkpoint")
        sd = torch.load(ck, map_location="cpu")
        sd = sd.get("model", sd)
        print("[Focal-T] load:", m.load_state_dict(sd, strict=False))
    m.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.num_features, num_classes))
    return m


def build_model(name: str, num_classes: int = 38, dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ("alexnet", "vgg16", "resnet50", "inception_v3", "densenet121", "mobilenet_v2",
                "efficientnet_b4", "vit_b16"):
        return _tv(name, num_classes, dropout, pretrained)
    if name in TIMM_NAMES:
        return _timm(name, num_classes, dropout, pretrained)
    if name == "cvt_13":
        return _HFCvT(num_classes, dropout, pretrained)
    if name == "focal_t":
        return _focal_t(num_classes, dropout, pretrained)
    if name == "plantxvit":
        from common.plantxvit import PlantXViT
        return PlantXViT(num_classes=num_classes, dropout=dropout)
    if name == "hta_convnet":
        from common.dbsa import HTAConvNet
        return HTAConvNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    raise KeyError(f"unknown model {name}")


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
