from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _text_factors(normal: torch.Tensor, anomaly: torch.Tensor,
                  class_sizes: list[int]):
    vectors = torch.cat((normal, anomaly), dim=0).double()
    groups = [len(normal), *class_sizes]
    if sum(groups) != len(vectors):
        raise ValueError("Anomaly class sizes do not match the text bank")
    mean = vectors.mean(dim=0)
    pooled = (vectors - mean) / math.sqrt(len(vectors) - 1)
    parts = vectors.split(groups, dim=0)
    within = torch.cat(
        [(part - part.mean(dim=0)) / math.sqrt(len(parts) * len(part))
         for part in parts], dim=0)
    between = torch.stack(
        [(part.mean(dim=0) - mean) * math.sqrt(len(part) / (len(vectors) - 1))
         for part in parts], dim=0)
    traces = [float(factor.square().sum()) for factor in (pooled, within, between)]
    if min(traces) <= 1e-10:
        raise ValueError("Degenerate text covariance")
    within = within * math.sqrt(traces[0] / traces[1])
    between = between * math.sqrt(traces[0] / traces[2])
    return within, between, traces


def _axes_from_factors(normal: torch.Tensor, means: torch.Tensor,
                       factors: list[torch.Tensor], ridge: float):
    matrix = torch.cat(factors, dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix @ matrix.T)
    retained = eigenvalues > 1e-10
    eigenvalues = eigenvalues[retained]
    eigenvectors = eigenvectors[:, retained]
    center = normal.double().mean(dim=0)
    directions = (means.double() - center).T
    whitened = directions * ridge ** -0.5
    if len(eigenvalues):
        basis = matrix.T @ (eigenvectors / torch.sqrt(eigenvalues)[None, :])
        whitened = whitened + basis @ (
            (torch.rsqrt(ridge + eigenvalues) - ridge ** -0.5)[:, None]
            * (basis.T @ directions))
    return center.float(), F.normalize(whitened.float(), dim=0)


def _pooled_axes(normal: torch.Tensor, anomaly: torch.Tensor,
                 means: torch.Tensor, ridge: float):
    vectors = torch.cat((normal, anomaly), dim=0)
    centered = vectors - vectors.mean(dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(
        centered @ centered.T / (len(vectors) - 1))
    retained = eigenvalues > 1e-7
    eigenvalues = eigenvalues[retained]
    eigenvectors = eigenvectors[:, retained]
    center = normal.mean(dim=0)
    directions = (means - center).T
    whitened = directions * ridge ** -0.5
    if len(eigenvalues):
        basis = centered.T @ eigenvectors / torch.sqrt((len(vectors) - 1) * eigenvalues)
        adjustment = torch.rsqrt(ridge + eigenvalues) - ridge ** -0.5
        whitened = whitened + basis @ (adjustment[:, None] * (basis.T @ directions))
    return center, F.normalize(whitened, p=2, dim=0)


def category_axes(normal: torch.Tensor, anomaly: torch.Tensor,
                  means: torch.Tensor, class_sizes: list[int],
                  video: torch.Tensor, config: dict, conditioned: bool):
    ridge = config["ridge"]
    if not conditioned:
        center, axes = _pooled_axes(normal, anomaly, means, ridge)
        return center, axes, None
    within, between, traces = _text_factors(normal, anomaly, class_sizes)
    centered_video = video.double() - video.double().mean(dim=0)
    within_energy = float(((centered_video @ within.T) ** 2).sum(dim=1).mean())
    between_energy = float(((centered_video @ between.T) ** 2).sum(dim=1).mean())
    settings = config["covariance"]
    signal = (math.log(traces[1] / traces[2])
              + settings["video_coefficient"]
              * math.log((within_energy + 1e-20) / (between_energy + 1e-20)))
    logit = settings["slope"] * signal
    weight = (1 / (1 + math.exp(-logit)) if logit >= 0
              else math.exp(logit) / (1 + math.exp(logit)))
    center, axes = _axes_from_factors(
        normal, means,
        [math.sqrt(1 - weight) * within, math.sqrt(weight) * between], ridge)
    return center, axes, weight
