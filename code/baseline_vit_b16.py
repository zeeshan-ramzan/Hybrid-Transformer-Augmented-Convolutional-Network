"""
baseline_vit_b16.py
===================
ViT-B/16 baseline (Dosovitskiy et al. [6])

Paper reference
---------------
    Table 10  "Comparison of Classification Performance on PlantVillage Test Set"
    Table 13  "Model Complexity and Inference Speed Comparison"
    Paper values for this method: Acc 98.67%, Prec 98.49%, Rec 98.38%, F1 98.43%

Note
----
ViT-B/16 requires the input side to be a multiple of its 16-pixel patch size, so
the loader rounds the requested resolution.  torchvision's ImageNet weights
exist only for 224x224; at any other resolution the model is constructed from
scratch and `--pretrained` reports that it could not be honoured.

This script is standalone: one execution loads the dataset, applies the paper's
stratified split, trains the network under the shared protocol of
`nn_utils.train_classifier` (Adam 1e-4, 5-epoch warm-up, cosine annealing,
label smoothing 0.1, class weights 1/sqrt(N_c), early stopping on validation
loss) and writes its complete result bundle to

    <PROJECT_ROOT>/results/vit_b16/

        tables/overall_performance.csv   Acc / macro P, R, F1 / AUC / ECE
        tables/per_class_metrics.csv     per-class precision, recall and F1
        tables/top_confusions.csv        largest off-diagonal confusion rates
        tables/complexity.csv            params / GFLOPs / FPS  (Table 13)
        tables/training_history.csv      per-epoch loss, accuracy, learning rate
        figures/training_accuracy.png    figures/training_loss.png
        figures/per_class_f1.png         figures/confusion_matrix.png
        figures/roc_curves.png           figures/reliability_diagram.png
        vit_b16_results.json

Usage
-----
    python baseline_vit_b16.py
    python baseline_vit_b16.py --quick
    python baseline_vit_b16.py --epochs 20 --cpu
    python baseline_vit_b16.py --pretrained          # ImageNet-1k initialisation, as in the paper
"""

from __future__ import annotations

import argparse

import data_utils as D
import nn_utils as N

RESULT_NAME = "vit_b16"


def main() -> None:
    ap = argparse.ArgumentParser(description="ViT-B/16 baseline (Dosovitskiy et al. [6])")
    D.cli_common(ap)
    args = ap.parse_args()
    N.run_baseline("vit_b16", args, result_name=RESULT_NAME)


if __name__ == "__main__":
    main()
