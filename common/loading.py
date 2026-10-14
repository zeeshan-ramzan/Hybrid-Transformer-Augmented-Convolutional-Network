"""Re-create a trained model from its run directory (results.json + best.pt)."""
import json
import os

import torch

from common.dbsa import HTAConvNet
from common.models import build_model


def hta_kwargs_from_results(res: dict) -> dict:
    cfg = res.get("hta_config", {})
    cfg = dict(cfg)
    if "stage_attention" in cfg and cfg["stage_attention"]:
        cfg["stage_attention"] = {int(k): v for k, v in cfg["stage_attention"].items()}
    return cfg


def load_run(run_dir: str, num_classes: int = 38, device="cpu"):
    """Loads either an HTA-ConvNet variant or a baseline saved by common.trainer.run_experiment."""
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    if "hta_config" in res:
        model = HTAConvNet(num_classes=num_classes, pretrained=False, **hta_kwargs_from_results(res))
    else:
        model = build_model(res.get("arch", res["model"]), num_classes, pretrained=False)
    sd = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(sd)
    return model.to(device).eval(), res
