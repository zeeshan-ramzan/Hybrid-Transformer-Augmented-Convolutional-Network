"""
baseline_efficientnet_b4.py
===========================
EfficientNet-B4 baseline (Tan and Le [5]) -- the backbone HTA-ConvNet builds on

Paper reference
---------------
    Table 10  "Comparison of Classification Performance on PlantVillage Test Set"
    Table 13  "Model Complexity and Inference Speed Comparison"
    Paper values for this method: Acc 98.92%, Prec 98.74%, Rec 98.63%, F1 98.68%

Why this baseline matters most
------------------------------
HTA-ConvNet is exactly this network plus three DBSA blocks, so the gap between
the two rows of Table 10 (98.92% vs 99.61%) is the paper's headline claim about
the DBSA block.  The same configuration is also the first row of the Table 11
ablation, and `ablation_study.py` re-runs it there under an identical schedule.

This script is standalone: one execution loads the dataset, applies the paper's
stratified split, trains the network under the shared protocol of
`nn_utils.train_classifier` (Adam 1e-4, 5-epoch warm-up, cosine annealing,
label smoothing 0.1, class weights 1/sqrt(N_c), early stopping on validation
loss) and writes its complete result bundle to

    <PROJECT_ROOT>/results/efficientnet_b4/

        tables/overall_performance.csv   Acc / macro P, R, F1 / AUC / ECE
        tables/per_class_metrics.csv     per-class precision, recall and F1
        tables/top_confusions.csv        largest off-diagonal confusion rates
        tables/complexity.csv            params / GFLOPs / FPS  (Table 13)
        tables/training_history.csv      per-epoch loss, accuracy, learning rate
        figures/training_accuracy.png    figures/training_loss.png
        figures/per_class_f1.png         figures/confusion_matrix.png
        figures/roc_curves.png           figures/reliability_diagram.png
        efficientnet_b4_results.json

Usage
-----
    python baseline_efficientnet_b4.py
    python baseline_efficientnet_b4.py --quick
    python baseline_efficientnet_b4.py --epochs 20 --cpu
    python baseline_efficientnet_b4.py --pretrained          # ImageNet-1k initialisation, as in the paper
"""

from __future__ import annotations

import argparse

import data_utils as D
import nn_utils as N

RESULT_NAME = "efficientnet_b4"


def main() -> None:
    ap = argparse.ArgumentParser(description="EfficientNet-B4 baseline (Tan and Le [5]) -- the backbone HTA-ConvNet builds on")
    D.cli_common(ap)
    args = ap.parse_args()
    N.run_baseline("efficientnet_b4", args, result_name=RESULT_NAME)


if __name__ == "__main__":
    main()
