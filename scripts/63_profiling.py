"""Computational profiling (Tables 'dbsa_cost', 'arch', 'complexity', 'profiling', 'effattn').

* analytical DBSA cost from the manuscript equations (per stage, per attention variant)
* measured parameters, FLOPs (fvcore, if installed), peak GPU memory at batch 1 and 32, latency at batch 1
  (50 warm-up + 500 timed iterations, FP16 autocast, cuDNN benchmark, explicit synchronisation; mean, SD,
  p95) and throughput at batch 32.

Usage: python scripts/63_profiling.py --models hta_convnet efficientnet_b4 swin_t convnext_t \
          --resolutions 96,128,160,224,256 --stage2-variants dense,window,sra4,sra8,linear
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.dbsa import HTAConvNet, analytical_dbsa_cost
from common.models import build_model, count_params
from common.utils import get_device, save_json


def flops_g(model, res, dev):
    try:
        from fvcore.nn import FlopCountAnalysis
        f = FlopCountAnalysis(model, torch.zeros(1, 3, res, res, device=dev))
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        return f.total() / 1e9
    except Exception as e:  # fvcore missing or op unsupported
        return None


@torch.no_grad()
def measure(model, res, dev, warm=50, iters=500, bs_tp=32):
    model.eval().to(dev)
    out = {}
    if dev.type != "cuda":
        iters, warm = 20, 5
    for bs in (1, bs_tp):
        x = torch.randn(bs, 3, res, res, device=dev)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            for _ in range(warm):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times = []
            n = iters if bs == 1 else max(iters // 5, 10)
            for _ in range(n):
                t0 = time.perf_counter()
                model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        times = np.array(times)
        mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None
        if bs == 1:
            out.update(latency_ms_mean=1e3 * times.mean(), latency_ms_sd=1e3 * times.std(),
                       latency_ms_p95=1e3 * np.percentile(times, 95), peak_mem_mb_bs1=mem)
        else:
            out.update(throughput_img_s=bs / times.mean(), peak_mem_mb_bs32=mem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hta_convnet", "efficientnet_b4"])
    ap.add_argument("--resolutions", default="224")
    ap.add_argument("--stage2-variants", default="dense")
    ap.add_argument("--iters", type=int, default=500)
    a = ap.parse_args()
    dev = get_device()
    torch.backends.cudnn.benchmark = True
    res_list = [int(r) for r in a.resolutions.split(",")]
    out = {"analytical_dbsa": {}, "measured": {}}
    for v in a.stage2_variants.split(","):
        for r in res_list:
            rows = analytical_dbsa_cost(r, stage2_attention=v)
            out["analytical_dbsa"][f"{v}@{r}"] = dict(stages=rows,
                                                      total_gflops=sum(x["dwsc_mflops"] + x["mhsa_mflops"]
                                                                       for x in rows) / 1e3)
    for name in a.models:
        variants = a.stage2_variants.split(",") if name == "hta_convnet" else [None]
        for v in variants:
            model = (HTAConvNet(38, pretrained=False, stage_attention={2: v}) if name == "hta_convnet"
                     else build_model(name, 38, pretrained=False))
            for r in res_list:
                key = f"{name}{'_' + v if v else ''}@{r}"
                m = dict(params_m=count_params(model), fvcore_gflops=flops_g(model.to(dev).eval(), r, dev))
                m.update(measure(model, r, dev, iters=a.iters))
                out["measured"][key] = m
                print(key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}, flush=True)
            del model
    save_json(out, "runs/profiling.json")


if __name__ == "__main__":
    main()
