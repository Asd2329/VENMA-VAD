from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _candidate_windows(video: torch.Tensor, scores: np.ndarray,
                       anomaly_means: torch.Tensor, config: dict):
    settings = config["prompt"]
    category_vectors = F.normalize(anomaly_means, p=2, dim=1)
    semantic = (video @ category_vectors.T).max(dim=1).values.cpu().numpy().astype(np.float64)
    length = len(scores)
    prefix = min(settings["candidate_prefix"], length)
    score_median = float(np.median(scores[:prefix]))
    score_scale = max(1.4826 * float(np.median(np.abs(scores[:prefix] - score_median))), 1e-6)
    semantic_median = float(np.median(semantic[:prefix]))
    semantic_scale = max(
        1.4826 * float(np.median(np.abs(semantic[:prefix] - semantic_median))), 1e-6)
    score_cut = score_median + settings["score_mad_multiplier"] * score_scale
    semantic_cut = semantic_median + settings["semantic_mad_multiplier"] * semantic_scale
    active = (scores > score_cut) & (semantic > semantic_cut)
    spans = []
    start = None
    for index in range(length + 1):
        inside = index < length and bool(active[index])
        if inside and start is None:
            start = index
        if not inside and start is not None:
            if index - start >= settings["min_event_windows"]:
                spans.append((start, index))
            start = None
    eligible = [index for start, end in spans for index in range(start, end)]
    strength = ((scores - score_cut) / score_scale
                + (semantic - semantic_cut) / semantic_scale)
    ranking = sorted(eligible, key=lambda index: (-float(strength[index]), index))
    available = np.ones(length, dtype=bool)
    selected = []
    maximum = max(1, math.ceil(length * settings["candidate_fraction"]))
    spacing = settings["candidate_spacing"]
    for index in ranking:
        if not available[index]:
            continue
        selected.append(index)
        available[max(0, index - spacing):min(length, index + spacing + 1)] = False
        if len(selected) >= maximum:
            break
    selected.sort()
    if len(eligible) / length > settings["candidate_coverage"]:
        selected = []
    return selected, spans


def _normal_reference(video: torch.Tensor, scores: np.ndarray,
                      candidates: list[int], spans: list[tuple[int, int]],
                      config: dict):
    settings = config["prompt"]
    length = len(scores)
    count = settings["normal_reference_count"]
    prefix = np.arange(min(length, count), dtype=np.int64)
    low_count = min(length, max(settings["low_min"],
                                min(settings["low_max"],
                                    math.ceil(settings["low_fraction"] * length))))
    low = np.argsort(scores, kind="stable")[:low_count].astype(np.int64)
    event_start = min((start for start, _ in spans),
                      default=min(candidates) if candidates else None)
    pre_event = None
    if event_start is not None and event_start - settings["pre_event_gap"] >= count:
        earlier = np.arange(event_start - settings["pre_event_gap"], dtype=np.int64)
        pre_event = earlier[np.argsort(scores[earlier], kind="stable")[:count]]
    lag = min(length - 1, settings["drift_lag"])
    drift = (float(torch.quantile((video[lag:] * video[:-lag]).sum(dim=1), .5).item())
             if lag else 1.)
    if drift >= settings["drift_cut"]:
        return prefix, "prefix"
    if pre_event is not None:
        return pre_event, "pre_event"
    return low, "low_score"


def select_normal(video: torch.Tensor, initial_score: np.ndarray,
                  anomaly_means: torch.Tensor, normal: torch.Tensor,
                  config: dict):
    candidates, spans = _candidate_windows(video, initial_score, anomaly_means, config)
    reference, reference_type = _normal_reference(video, initial_score,
                                                   candidates, spans, config)
    removed = ()
    if candidates:
        event_response = (video[candidates] @ normal.T).median(dim=0).values
        normal_response = (video[reference] @ normal.T).median(dim=0).values
        difference = event_response - normal_response
        count = math.ceil(config["prompt"]["normal_trim_fraction"] * len(normal))
        ranking = sorted(range(len(normal)),
                         key=lambda index: (-float(difference[index]), index))
        proposed = ranking[:count]
        if all(float(difference[index]) > 0 for index in proposed):
            removed = tuple(sorted(proposed))
    retained = [index for index in range(len(normal)) if index not in removed]
    return normal[retained], {
        "reference_type": reference_type,
        "removed_normal_indices": list(removed),
    }
