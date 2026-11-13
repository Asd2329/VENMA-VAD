from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .settings import ROOT


DATA = ROOT / "data"
EXPECTED_VIDEOS = {"ucf": 290, "xd": 800}
EXPECTED_WINDOWS = {"ucf": (67894, 5226), "xd": (141650, 32962)}


def video_key(dataset: str, path: Path) -> str:
    stem = path.stem
    return stem if dataset == "ucf" else stem.split("_label_", 1)[0]


def feature_paths(dataset: str, directory: Path) -> list[Path]:
    if dataset not in EXPECTED_VIDEOS:
        raise ValueError(f"Unknown dataset: {dataset}")
    names = (DATA / f"{dataset}_feature_list.txt").read_text(encoding="utf-8").splitlines()
    if len(names) != EXPECTED_VIDEOS[dataset] or len(set(names)) != len(names):
        raise ValueError(f"Expected {EXPECTED_VIDEOS[dataset]} unique feature names")
    indexed = {}
    for path in directory.rglob("*.npy"):
        if path.name in indexed:
            raise ValueError(f"Duplicate feature name: {path.name}")
        indexed[path.name] = path
    if set(indexed) != set(names):
        raise ValueError(f"Missing or unexpected features in {directory}")
    paths = [indexed[name] for name in names]
    keys = [video_key(dataset, path) for path in paths]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate feature identifiers in {directory}")
    return paths


def load_video(path: Path, device: torch.device) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"Invalid video features: {path}")
    return F.normalize(torch.from_numpy(array).float().to(device), dim=1)


def _description_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_text_bank(dataset: str, device: torch.device):
    description = DATA / f"{dataset}_descriptions.json"
    archive = DATA / f"{dataset}_text_bank_cpu.npz"
    document = json.loads(description.read_text(encoding="utf-8"))
    with np.load(archive, allow_pickle=False) as saved:
        if str(saved["descriptions_sha256"].item()) != _description_hash(description):
            raise ValueError(f"Text bank does not match {description.name}")
        normal, anomaly, means = (
            torch.from_numpy(np.asarray(saved[name], dtype=np.float32)).to(device)
            for name in ("normal", "anomaly", "anomaly_means"))
        sizes = list(map(int, saved["sizes"])) if "sizes" in saved.files else None
    if dataset == "ucf":
        effective = document["effective_prompts"]
        normal_texts = effective["normal"]
        anomaly_groups = effective["anomalies"]
        if normal_texts != document["prompt_config"]["normal_immunity_texts"]:
            raise ValueError("UCF prompt order does not match the final description file")
        if [group["class"] for group in anomaly_groups] != sorted(document["content"]["anomalies"]):
            raise ValueError("UCF anomaly classes are out of order")
        expected_sizes = [len(group["texts"]) for group in anomaly_groups]
    else:
        normal_texts = [text for group in document["prompt_config"]["scenario_normals"].values()
                        for text in group]
        classes = document["prompt_config"]["class_overrides"]
        expected_sizes = [len(document["content"]["anomalies"][classes[code]])
                          for code in sorted(classes)]
    if sizes is None:
        sizes = expected_sizes
    if (len(normal) != len(normal_texts) or len(means) != len(sizes)
            or sum(sizes) != len(anomaly) or sizes != expected_sizes):
        raise ValueError("Text embedding shapes do not match the prompt descriptions")
    return normal, anomaly, means, sizes
