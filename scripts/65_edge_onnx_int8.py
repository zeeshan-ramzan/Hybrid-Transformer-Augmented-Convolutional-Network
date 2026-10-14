"""Edge deployment (Discussion, 'Practical Deployment Considerations', Table 'resolution'): ONNX export, INT8
post-training dynamic quantisation with onnxruntime, file size and CPU throughput (run this script on the
Raspberry Pi 4B for the Pi4 column; it also runs on any CPU).

Usage: python scripts/65_edge_onnx_int8.py --run runs/hta_convnet --img-size 224
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import PV_CLASSES
from common.loading import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model, _ = load_run(a.run, len(PV_CLASSES), "cpu")
    fp32 = os.path.join(a.run, f"model_{a.img_size}.onnx")
    int8 = os.path.join(a.run, f"model_{a.img_size}_int8.onnx")
    torch.onnx.export(model, torch.randn(1, 3, a.img_size, a.img_size), fp32, input_names=["x"],
                      output_names=["logits"], opset_version=17)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    for path in (fp32, int8):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, a.img_size, a.img_size).astype(np.float32)
        for _ in range(5):
            s.run(None, {"x": x})
        t0 = time.perf_counter()
        for _ in range(a.iters):
            s.run(None, {"x": x})
        dt = (time.perf_counter() - t0) / a.iters
        print(f"{os.path.basename(path)}: {os.path.getsize(path) / 2 ** 20:.1f} MB, {1 / dt:.2f} img/s "
              f"({dt:.3f} s/img)")


if __name__ == "__main__":
    main()
