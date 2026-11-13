from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def residual_mean(video: torch.Tensor, normal: torch.Tensor,
                  config: dict) -> float:
    normal_vectors = F.normalize(normal, dim=1).detach().cpu().numpy().astype(np.float64)
    video_vectors = F.normalize(video, dim=1).detach().cpu().numpy().astype(np.float64)
    similarity = normal_vectors @ normal_vectors.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    coefficient = similarity[np.arange(len(normal_vectors)), nearest]
    residual = video_vectors @ (
        normal_vectors - coefficient[:, None] * normal_vectors[nearest]).T
    contrast = residual.max(axis=1) - residual.mean(axis=1)
    return float(gaussian_filter1d(contrast, config["smoothing_sigma"]).mean())


def calibrated_score(score: np.ndarray, residual: float,
                     config: dict) -> np.ndarray:
    return score - config["evidence"]["weight"] * residual
