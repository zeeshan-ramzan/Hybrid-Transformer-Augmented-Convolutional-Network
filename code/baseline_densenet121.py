"""
baseline_densenet121.py
=======================
DenseNet-121 baseline (Huang et al. [9])

Paper reference
---------------
    Table 10  "Comparison of Classification Performance on PlantVillage Test Set"
    Table 13  "Model Complexity and Inference Speed Comparison"
    Paper values for this method: Acc 98.12%, Prec 97.93%, Rec 97.78%, F1 97.85%

This script is standalone: one execution loads the dataset, applies the paper's
stratified split, trains the network under the shared protocol of
`nn_utils.train_classifier` (Adam 1e-4, 5-epoch warm-up, cosine annealing,
label smoothing 0.1, class weights 1/sqrt(N_c), early stopping on validation
loss) and writes its complete result bundle to

    <PROJECT_ROOT>/results/densenet121/

        tables/overall_performance.csv   Acc / macro P, R, F1 / AUC / ECE
        tables/per_class_metrics.csv     per-class precision, recall and F1
        tables/top_confusions.csv        largest off-diagonal confusion rates
        tables/complexity.csv            params / GFLOPs / FPS  (Table 13)
        tables/training_history.csv      per-epoch loss, accuracy, learning rate
        figures/training_accuracy.png    figures/training_loss.png
        figures/per_class_f1.png         figures/confusion_matrix.png
        figures/roc_curves.png           figures/reliability_diagram.png
        densenet121_results.json

Usage
-----
    python baseline_densenet121.py
    python baseline_densenet121.py --quick
    python baseline_densenet121.py --epochs 20 --cpu
    python baseline_densenet121.py --pretrained          # ImageNet-1k initialisation, as in the paper
"""

from __future__ import annotations

import argparse

import data_utils as D
import nn_utils as N

RESULT_NAME = "densenet121"


def main() -> None:
    ap = argparse.ArgumentParser(description="DenseNet-121 baseline (Huang et al. [9])")
    D.cli_common(ap)
    args = ap.parse_args()
    N.run_baseline("densenet121", args, result_name=RESULT_NAME)


if __name__ == "__main__":
    main()
