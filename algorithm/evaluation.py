from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_io import DATA, EXPECTED_WINDOWS, feature_paths, video_key


def _annotation_key(token: str) -> str:
    name = Path(token).name
    if name.lower().endswith((".mp4", ".npy")):
        name = name.rsplit(".", 1)[0]
    return name.split("_label_", 1)[0]


def _annotations(dataset: str) -> dict[str, list[tuple[int, int]]]:
    path = DATA / f"{dataset}_annotations.txt"
    records = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if dataset == "ucf":
            key = Path(fields[0]).stem
            intervals = [(int(fields[i]), int(fields[i + 1]))
                         for i in range(2, len(fields) - 1, 2)]
            records[key] = [(start, end) for start, end in intervals if start != -1]
        else:
            if (len(fields) - 1) % 2:
                raise ValueError(f"Invalid annotation intervals for {fields[0]}")
            key = _annotation_key(fields[0])
            intervals = [(int(fields[i]), int(fields[i + 1]))
                         for i in range(1, len(fields), 2)]
            records.setdefault(key, []).extend(
                (start, end) for start, end in intervals if (start, end) != (-1, -1))
    return records


def evaluate(dataset: str, feature_dir: Path, output: Path,
             variants: tuple[str, ...]) -> dict:
    paths = feature_paths(dataset, feature_dir)
    folder = output / dataset
    score_paths = [folder / "videos" / f"{video_key(dataset, path)}.scores.npz"
                   for path in paths]
    missing = [path for path in score_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} score files; evaluation did not start")
    annotations = _annotations(dataset)
    labels = []
    scores = {variant: [] for variant in variants}
    for feature_path, score_path in zip(paths, score_paths):
        with np.load(score_path, allow_pickle=False) as saved:
            length = len(saved[variants[0]])
            truth = np.zeros(length, dtype=np.uint8)
            for start, end in annotations.get(video_key(dataset, feature_path), ()):
                truth[max(0, start // 16):min(length, end // 16)] = 1
            for variant in variants:
                value = np.asarray(saved[variant], dtype=np.float64)
                if len(value) != length or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {variant} scores in {score_path}")
                scores[variant].append(value)
        labels.append(truth)
    truth = np.concatenate(labels)
    if (len(truth), int(truth.sum())) != EXPECTED_WINDOWS[dataset]:
        raise ValueError("Feature and annotation window counts do not match the full test set")
    metrics = {}
    for variant in variants:
        values = np.concatenate(scores[variant])
        normal = values[truth == 0]
        abnormal = values[truth == 1]
        metrics[variant] = {
            "auc": float(roc_auc_score(truth, values)),
            "ap": float(average_precision_score(truth, values)),
            "snr": float((abnormal.mean() - normal.mean()) / (normal.std() + 1e-9)),
        }
    report = {
        "dataset": dataset,
        "videos": len(paths),
        "windows": len(truth),
        "positive_windows": int(truth.sum()),
        "metrics": metrics,
        "evaluation": "all mapped windows pooled across test videos",
        "frame_to_window": "start//16:end//16",
    }
    (folder / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
