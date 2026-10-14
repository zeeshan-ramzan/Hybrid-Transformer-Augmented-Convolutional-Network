# HTA-ConvNet — code for every algorithm in the manuscript

"Hybrid Transformer-Augmented Convolutional Network for Robust Plant Disease Recognition from Leaf Images"

One Python file per algorithm/experiment (`scripts/`), shared code in `common/`.
Files marked **[NEW]** implement the experiments added in the revision (Reviewer 3).

## 1. Installation

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Optional:
* **Focal-T** (not available in timm): `git clone https://github.com/microsoft/Focal-Transformer`, download the
  Focal-T ImageNet-1k checkpoint listed in that repository's README, then
  `export FOCAL_REPO=/path/Focal-Transformer` and `export FOCAL_T_CKPT=/path/focal-tiny.pth`
  (on Windows: `set FOCAL_REPO=...`). The official repo also needs `pip install yacs`.
* **SURF** for the Ahmad et al. re-implementation needs an OpenCV build with the non-free module; otherwise SIFT
  is used automatically and this is recorded in `results.json`.

## 2. Data layout

```
data/PlantVillage/color/<38 class folders>/*.JPG         # https://github.com/spMohanty/PlantVillage-Dataset (raw/color)
data/PlantVillage/segmented/<38 class folders>/*.jpg      # (raw/segmented) - background-removed test only
data/PlantDoc-Dataset/{train,test}/<class>/*.jpg          # https://github.com/pratikkayal/PlantDoc-Dataset
data/plant-pathology-2020/train.csv, images/*.jpg         # Kaggle FGVC7 "plant-pathology-2020-fgvc7"
```

All commands are run from the project root. Every training script accepts `--help`.
Defaults reproduce the manuscript protocol (224x224, batch 32, Adam, LR 1e-4, 5-epoch warm-up + cosine to 5e-6,
50 epochs, patience 10, label smoothing 0.1, w_c = 1/sqrt(N_c), mixup 0.2, weighted sampler, AMP, seed 42).
Outputs go to `runs/<name>/` (`results.json`, `best.pt`, `test_predictions.npz`).

## 3. Files

| File | Algorithm / experiment | Manuscript element |
|---|---|---|
| `00_prepare_splits.py` | stratified 70/15/15 split, seed 42 | Data Partitioning, Table split |
| `01_near_duplicate_screening.py` | MD5 + pHash + embedding near-duplicate groups, group-aware split | Leakage control, Table dedup |
| `02_label_audit.py` | out-of-fold label audit sheet | Label provenance |
| `10_hta_convnet.py` | **proposed HTA-ConvNet / DBSA** (all variants via flags) | Methods, Tables main/ablation |
| `11`–`18` | AlexNet, VGG-16, ResNet-50, InceptionV3, DenseNet-121, MobileNetV2, EfficientNet-B4, ViT-B/16 | Tables baselines/main/cross/complexity |
| `20_swin_t.py` **[NEW]** | Swin-T | R3.4 |
| `21_focal_t.py` **[NEW]** | Focal-T (official implementation) | R3.4 |
| `22_coatnet_0.py` **[NEW]** | CoAtNet-0 (timm `coatnet_0_rw_224`) | R3.4, R3.7 |
| `23_cvt_13.py` **[NEW]** | CvT-13 (HuggingFace `microsoft/cvt-13`) | R3.4, R3.7 |
| `24_convnext_t.py` **[NEW]** | ConvNeXt-T | R3.4 |
| `25_edgenext_s.py` **[NEW]** | EdgeNeXt-S | R3.4 |
| `26_plantxvit.py` **[NEW]** | PlantXViT re-implementation (from scratch) | R3.7 |
| `30_ahmad_surf_colour_glcm_svm.py` | Ahmad et al. 2021 two-stage SURF / colour+GLCM SVM | Tables main/cross/classical |
| `31`–`34` | colour-hist+kNN, LBP+SVM, HOG+RF, colour+GLCM+HOG+SVM | Table classical |
| `40_ablation_incremental.py` | DWSC-only, MHSA-only, DBSA@4, DBSA@3,4, full | Table ablation (a) |
| `41_ablation_fusion.py` | G=0.5, addition, concat+1x1, gate without residual | Table ablation (b) |
| `42_ablation_module_replacement.py` | SE, CBAM, MobileViT block | Table ablation (c) |
| `43_sensitivity_heads_blocks_resolution.py` | k = 1,2,4,16; extra stage 1/5 blocks; 96–256 px | Fig. heads, Table resolution |
| `44_effattn_window.py` **[NEW]** | Stage-2 window attention 7x7 | Table effattn (R3.1) |
| `45_effattn_sra4.py` **[NEW]** | Stage-2 spatial-reduction attention R=4 | Table effattn (R3.1) |
| `46_effattn_sra8.py` **[NEW]** | Stage-2 spatial-reduction attention R=8 | Table effattn (R3.1) |
| `47_effattn_linear.py` **[NEW]** | Stage-2 linear attention | Table effattn (R3.1) |
| `50_cross_validation_5fold.py` | 5-fold CV + out-of-fold predictions | Table cv |
| `51_multi_seed.py` | seeds 42–51, mean/SD/t-CI | Table seeds |
| `52_statistical_tests.py` | McNemar, Wilson, bootstrap | Statistical significance |
| `53_cross_dataset_zero_shot.py` | PlantDoc / PP2020 zero-shot, per-crop, ECE, **[NEW]** lenient multi-disease | Tables cross, crossdetail (R3.2/R3.3) |
| `54_background_removed_eval.py` | segmented-image test | Table dedup |
| `55_adapt_adabn.py` **[NEW]** | AdaBN | Table adapt (R3.3) |
| `56_adapt_dann.py` **[NEW]** | DANN | Table adapt (R3.3) |
| `57_adapt_finetune.py` **[NEW]** | few-shot (10/class) and full fine-tuning | Table adapt (R3.3) |
| `58_fusion_stability.py` **[NEW]** | update norm, local Lipschitz, noise, divergence | Table stability (R3.5) |
| `59_ambiguity_analysis.py` **[NEW]** | annotation sheet, kappa, margin/entropy/top-2 | Table ambig (R3.6) |
| `60_selective_prediction.py` **[NEW]** | temperature scaling, MSP / MC-dropout reject option, AURC | Table selective (R3.6) |
| `61_gradcam.py` | Grad-CAM | Grad-CAM section |
| `62_gate_analysis.py` | gate statistics per stage / class | Gate activation patterns |
| `63_profiling.py` | analytical + measured FLOPs, memory, latency, throughput | Tables dbsa_cost/complexity/profiling/effattn |
| `64_calibration_roc_confusion.py` | ECE, ROC/AUC, per-class table, confusion, error types | Figs. roc, f1hist; Tables auc, perclass |
| `65_edge_onnx_int8.py` | ONNX + INT8, CPU / Raspberry Pi throughput | Deployment, Table resolution |

## 4. Typical order

```bash
python scripts/00_prepare_splits.py
python scripts/01_near_duplicate_screening.py
python scripts/10_hta_convnet.py                                   # proposed model
python scripts/17_efficientnet_b4.py --lr-grid 5e-5,1e-4,3e-4      # every baseline: same protocol
...
python scripts/53_cross_dataset_zero_shot.py --runs runs/hta_convnet runs/efficientnet_b4 ... \
       --plantdoc-root data/PlantDoc-Dataset --pp2020-root data/plant-pathology-2020
python scripts/52_statistical_tests.py --reference runs/hta_convnet --others runs/efficientnet_b4 runs/swin_t ...
```

`run_new_experiments.py` (Windows/Linux) or `run_new_experiments.sh` (Linux) runs only the experiments added in the revision (requires the trained
`runs/hta_convnet`, `runs/efficientnet_b4` and the fusion variants of `41_ablation_fusion.py`).

## 5. Notes

* `--sdpa` in `10_hta_convnet.py` uses PyTorch memory-efficient attention kernels (same mathematics, much lower
  memory); leave it off to reproduce the memory numbers of the manuscript.
* Debug quickly on a subset with `--epochs 1 --max-train-batches 20 --img-size 64`.
* CoAtNet-0 uses timm's `coatnet_0_rw_224.sw_in1k` (timm re-implementation; 26.7M parameters with a 38-class head).
* PlantXViT hyper-parameters (patch size, depth, width) are exposed in `common/plantxvit.py`; check them against
  the original paper before reporting.
