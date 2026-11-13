from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from algorithm.evaluation import evaluate
from algorithm.main import score_dataset
from algorithm.pipeline import ALL_VARIANTS
from algorithm.settings import DEFAULT_CONFIG, ROOT, load_config


LEAVE_ONE_OUT = {
    "all_off": "off",
    "without_C": "VTE",
    "without_V": "CTE",
    "without_T": "CVE",
    "without_E": "CVT",
    "full": "CVTE",
}
FACTORIAL_FIXED_E = ("E", "CE", "VE", "TE", "CVE", "CTE", "VTE", "CVTE")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("score", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ablation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    feature_dir = args.features or ROOT / "video-features" / args.dataset
    if args.phase == "score":
        result = score_dataset(args.dataset, feature_dir, args.output,
                               torch.device(args.device), load_config(args.config),
                               ALL_VARIANTS)
    else:
        result = evaluate(args.dataset, feature_dir, args.output, ALL_VARIANTS)
        result["leave_one_out"] = {
            label: result["metrics"][variant] for label, variant in LEAVE_ONE_OUT.items()}
        result["factorial_fixed_e"] = {
            variant: result["metrics"][variant] for variant in FACTORIAL_FIXED_E}
        path = args.output / args.dataset / "evaluation.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    cli()
