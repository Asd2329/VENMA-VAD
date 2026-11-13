from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data_io import DATA, EXPECTED_WINDOWS, feature_paths, load_text_bank, load_video, video_key
from .evaluation import evaluate
from .pipeline import score_variants
from .settings import DEFAULT_CONFIG, ROOT, load_config


def score_dataset(dataset: str, feature_dir: Path, output: Path,
                  device: torch.device, config: dict,
                  variants: tuple[str, ...] = ("CVTE",)) -> dict:
    paths = feature_paths(dataset, feature_dir)
    folder = output / dataset
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output is not empty: {folder}")
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    bank = load_text_bank(dataset, device)
    windows = 0
    with torch.inference_mode():
        for index, feature_path in enumerate(paths, 1):
            video = load_video(feature_path, device)
            scores = score_variants(video, bank, config, variants)
            output_path = folder / "videos" / f"{video_key(dataset, feature_path)}.scores.npz"
            np.savez_compressed(output_path, **scores)
            windows += len(video)
            if index % 100 == 0 or index == len(paths):
                print(f"{dataset}: {index}/{len(paths)}", flush=True)
    if windows != EXPECTED_WINDOWS[dataset][0]:
        raise ValueError(f"Expected {EXPECTED_WINDOWS[dataset][0]} windows, found {windows}")
    text_archive = DATA / f"{dataset}_text_bank_cpu.npz"
    report = {
        "dataset": dataset,
        "videos": len(paths),
        "windows": windows,
        "variants": list(variants),
        "feature_dir": str(feature_dir),
        "text_bank_sha256": hashlib.sha256(text_archive.read_bytes()).hexdigest(),
        "settings": config,
    }
    (folder / "run_config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("score", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "main")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    feature_dir = args.features or ROOT / "video-features" / args.dataset
    if args.phase == "score":
        result = score_dataset(args.dataset, feature_dir, args.output,
                               torch.device(args.device), load_config(args.config))
    else:
        result = evaluate(args.dataset, feature_dir, args.output, ("CVTE",))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    cli()
