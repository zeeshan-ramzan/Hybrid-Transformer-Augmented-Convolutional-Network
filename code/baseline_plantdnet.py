"""
baseline_plantdnet.py
=====================
PlantDNet baseline (Thapa et al. [18])

Paper reference
---------------
    Table 10  "Comparison of Classification Performance on PlantVillage Test Set"
    Table 13  "Model Complexity and Inference Speed Comparison"
    Paper values for this method: Acc 98.34%, Prec 98.12%, Rec 98.01%, F1 98.06%

IMPORTANT -- this is a reconstruction
-------------------------------------
The paper lists "PlantDNet" in Table 10 but never defines the architecture, and
the reference it cites (Thapa et al. [18]) describes a ResNet-50 applied to the
Plant Pathology 2020 apple dataset -- not a network called PlantDNet.  This
script therefore runs `nn_utils.PlantDNetHead`: a ResNet-50 trunk whose
global-average-pooled descriptors are taken from three depths, concatenated and
classified by a two-layer head -- the multi-scale feature fusion that the
PlantDiseaseNet family is characterised by.  Read the resulting row as "a
ResNet-50-based multi-scale plant disease network", not as a faithful
re-implementation of an unspecified model.  See README section 9.7.

This script is standalone: one execution loads the dataset, applies the paper's
stratified split, trains the network under the shared protocol of
`nn_utils.train_classifier` (Adam 1e-4, 5-epoch warm-up, cosine annealing,
label smoothing 0.1, class weights 1/sqrt(N_c), early stopping on validation
loss) and writes its complete result bundle to

    <PROJECT_ROOT>/results/plantdnet/

        tables/overall_performance.csv   Acc / macro P, R, F1 / AUC / ECE
        tables/per_class_metrics.csv     per-class precision, recall and F1
        tables/top_confusions.csv        largest off-diagonal confusion rates
        tables/complexity.csv            params / GFLOPs / FPS  (Table 13)
        tables/training_history.csv      per-epoch loss, accuracy, learning rate
        figures/training_accuracy.png    figures/training_loss.png
        figures/per_class_f1.png         figures/confusion_matrix.png
        figures/roc_curves.png           figures/reliability_diagram.png
        plantdnet_results.json

Usage
-----
    python baseline_plantdnet.py
    python baseline_plantdnet.py --quick
    python baseline_plantdnet.py --epochs 20 --cpu
    python baseline_plantdnet.py --pretrained          # ImageNet-1k initialisation, as in the paper
"""

from __future__ import annotations

import argparse

import data_utils as D
import nn_utils as N

RESULT_NAME = "plantdnet"


def main() -> None:
    ap = argparse.ArgumentParser(description="PlantDNet baseline (Thapa et al. [18])")
    D.cli_common(ap)
    args = ap.parse_args()
    N.run_baseline("plantdnet", args, result_name=RESULT_NAME)


if __name__ == "__main__":
    main()
