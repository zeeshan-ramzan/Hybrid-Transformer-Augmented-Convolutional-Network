"""PROPOSED METHOD: HTA-ConvNet (EfficientNet-B4 + DBSA blocks after Stages 2, 3, 4).

All architectural variants of the manuscript are selectable from the command line, so this single script
also trains the ablation configurations (the dedicated ablation / efficient-attention scripts call it).

Examples
    # main model (Table 'main', 99.61% target)
    python scripts/10_hta_convnet.py --data-root data/PlantVillage/color
    # group-aware split (Table 'dedup')
    python scripts/10_hta_convnet.py --split-csv splits/group_aware_split.csv --name hta_groupaware
    # number of heads (Fig. 'heads') / stages (Sensitivity to Number of DBSA Blocks) / resolution
    python scripts/10_hta_convnet.py --heads 16 --name hta_k16
    python scripts/10_hta_convnet.py --stages 1,2,3,4 --name hta_stage1234
    python scripts/10_hta_convnet.py --img-size 128 --name hta_res128
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import pv_classes
from common.dbsa import HTAConvNet
from common.trainer import run_experiment
from common.utils import base_parser


def add_hta_args(p):
    p.add_argument("--name", default="hta_convnet")
    p.add_argument("--stages", default="2,3,4", help="stages that receive a block, e.g. 2,3,4 or 4")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--attention", default="dense", choices=["dense", "window", "sra4", "sra8", "linear"])
    p.add_argument("--stage2-attention", default="", choices=["", "dense", "window", "sra4", "sra8", "linear"],
                   help="override the attention type of the Stage-2 block only (Table 'effattn')")
    p.add_argument("--fusion", default="gate", choices=["gate", "avg", "add", "concat", "gate_nores"])
    p.add_argument("--branches", default="both", choices=["both", "conv", "attn"])
    p.add_argument("--module", default="dbsa", choices=["dbsa", "se", "cbam", "mobilevit"])
    p.add_argument("--sdpa", action="store_true",
                   help="use memory-efficient scaled_dot_product_attention kernels (results unchanged)")
    return p


def hta_config(args):
    stages = tuple(int(s) for s in args.stages.split(",") if s)
    stage_att = {2: args.stage2_attention} if args.stage2_attention else {}
    return dict(stages=stages, heads=args.heads, attention=args.attention, stage_attention=stage_att,
                fusion=args.fusion, branches=args.branches, module=args.module, dropout=args.dropout,
                use_sdpa=args.sdpa)


def main(argv=None):
    p = add_hta_args(base_parser("HTA-ConvNet (proposed)"))
    args = p.parse_args(argv)
    n_classes = len(pv_classes(args.data_root))
    cfg = hta_config(args)
    run_experiment(lambda: HTAConvNet(num_classes=n_classes, pretrained=not args.no_pretrained, **cfg), args, args.name,
                   extra={"hta_config": cfg, "arch": "hta_convnet"})


if __name__ == "__main__":
    main()
