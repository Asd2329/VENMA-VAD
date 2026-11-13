from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata


def _window_count(length: int, config: dict) -> int:
    settings = config["visual"]
    return min(length, max(settings["min_windows"],
                           min(settings["max_windows"],
                               math.ceil(settings["fraction"] * length))))


def _rank_threshold(tail_probability: float, slope: float) -> float:
    high = float(expit(slope))
    low = float(expit(-slope))
    value = float(expit(slope * (2 * tail_probability - 1)))
    return .5 + .5 * (value - low) / (high - low)


def _geometric_direction(video: np.ndarray, scores: np.ndarray,
                         selected: np.ndarray) -> np.ndarray:
    points = video[selected].astype(np.float64)
    values = scores[selected]
    median = np.median(values)
    scale = max(1.4826 * float(np.median(np.abs(values - median))), .001)
    weights = np.exp(np.clip(-(values - values.min()) / scale, -50, 0))
    weights /= weights.sum()
    center = np.sum(weights[:, None] * points, axis=0)
    for _ in range(80):
        distances = np.linalg.norm(points - center, axis=1)
        adjusted = weights / np.maximum(distances, 1e-8)
        updated = np.sum(adjusted[:, None] * points, axis=0) / adjusted.sum()
        if np.linalg.norm(updated - center) <= 1e-6:
            center = updated
            break
        center = updated
    return center / np.linalg.norm(center)


def visual_direction(video: torch.Tensor, initial_score: np.ndarray,
                     config: dict) -> np.ndarray:
    length = len(initial_score)
    count = _window_count(length, config)
    ranks = (rankdata(initial_score, method="average") - .5) / length
    block_means = np.convolve(ranks, np.ones(count) / count, mode="valid")
    prefix_rank = float(block_means[0])
    tail = float(np.count_nonzero(block_means >= prefix_rank) / len(block_means))
    threshold = _rank_threshold(tail, config["visual"]["rank_slope"])
    selected = (np.argsort(initial_score, kind="stable")[:count]
                if prefix_rank > threshold else np.arange(count))
    return _geometric_direction(video.detach().cpu().numpy(), initial_score, selected)


def fused_center(text_center: torch.Tensor, direction: np.ndarray,
                 config: dict) -> torch.Tensor:
    weight = config["visual"]["center_weight"]
    visual = torch.from_numpy(direction.astype(np.float32)).to(text_center.device)
    return (1 - weight) * text_center + weight * torch.linalg.norm(text_center) * visual
