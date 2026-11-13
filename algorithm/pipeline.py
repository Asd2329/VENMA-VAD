from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from .covariance import category_axes
from .evidence import calibrated_score, residual_mean
from .prompt import select_normal
from .visual import fused_center, visual_direction


MODULES = "CVTE"
ALL_VARIANTS = (
    "off", "C", "V", "T", "E", "CV", "CT", "CE", "VT", "VE", "TE",
    "CVT", "CVE", "CTE", "VTE", "CVTE",
)


def _project(video: torch.Tensor, center: torch.Tensor,
             axes: torch.Tensor, config: dict) -> np.ndarray:
    logits = (video - center) @ axes
    weights = F.softmax(logits / config["projection_temperature"], dim=1)
    raw = (logits * weights).sum(dim=1)
    return gaussian_filter1d(raw.detach().cpu().numpy(), config["smoothing_sigma"])


def score_variants(video: torch.Tensor, text_bank: tuple,
                   config: dict, variants: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not variants or any(code not in ALL_VARIANTS for code in variants):
        raise ValueError("Unknown or empty module configuration")
    normal, anomaly, means, sizes = text_bank
    requested = {(code, "C" in code, "V" in code, "T" in code, "E" in code)
                 for code in variants}
    cache = {}
    for conditioned in sorted({item[1] for item in requested}):
        branches = [item for item in requested if item[1] == conditioned]
        needs_reference = any(item[2] or item[3] for item in branches)
        initial = None
        if needs_reference:
            first_center, first_axes, _ = category_axes(
                normal, anomaly, means, sizes, video, config, conditioned)
            initial = _project(video, first_center, first_axes, config)
        selected = (select_normal(video, initial, means, normal, config)[0]
                    if any(item[3] for item in branches) else None)
        direction = (visual_direction(video, initial, config)
                     if any(item[2] for item in branches) else None)
        for use_prompt in sorted({item[3] for item in branches}):
            bank = selected if use_prompt else normal
            center, axes, _ = category_axes(
                bank, anomaly, means, sizes, video, config, conditioned)
            for use_visual in sorted({item[2] for item in branches if item[3] == use_prompt}):
                scoring_center = fused_center(center, direction, config) if use_visual else center
                cache[(conditioned, use_prompt, use_visual)] = _project(
                    video, scoring_center, axes, config)
    residual = (residual_mean(video, normal, config)
                if any(item[4] for item in requested) else None)
    result = {}
    for code, conditioned, use_visual, use_prompt, use_evidence in requested:
        score = cache[(conditioned, use_prompt, use_visual)]
        result[code] = (calibrated_score(score, residual, config)
                        if use_evidence else score)
    return {code: result[code] for code in variants}
