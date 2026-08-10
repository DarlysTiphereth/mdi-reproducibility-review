"""MDI formulas and normalization, kept independent from data generation."""

from __future__ import annotations

import numpy as np


def minmax_0_100(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Normalization input must be finite")
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return 100.0 * (values - lo) / (hi - lo)


def socioeconomic_pressure(population: np.ndarray, area_km2: np.ndarray,
                           income: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P = population density x inverse income, followed by 0-100 scaling."""
    population = np.asarray(population, dtype=float)
    area_km2 = np.asarray(area_km2, dtype=float)
    income = np.asarray(income, dtype=float)
    if np.any(area_km2 <= 0) or np.any(income <= 0):
        raise ValueError("Area and income must be strictly positive")
    raw = (population / area_km2) * (1.0 / income)
    return raw, minmax_0_100(raw)


def routing_inefficiency(network_distance_m: np.ndarray,
                         geodesic_distance_m: np.ndarray) -> np.ndarray:
    """I = d_network / d_geodesic."""
    network_distance_m = np.asarray(network_distance_m, dtype=float)
    geodesic_distance_m = np.asarray(geodesic_distance_m, dtype=float)
    if np.any(geodesic_distance_m <= 0):
        raise ValueError("Geodesic distance must be strictly positive")
    ratio = network_distance_m / geodesic_distance_m
    if np.any(ratio < 1.0 - 1e-8):
        raise ValueError("Network distance cannot be shorter than geodesic distance")
    return ratio


def normalized_headway_entropy(headways: np.ndarray) -> float:
    """Normalized Shannon entropy of positive headway shares.

    Equal headways have maximum normalized entropy (1). The announced MDI
    supply formula uses an irregularity penalty, so H_irregularity is defined as
    one minus this entropy. This avoids discretization bins and is fully
    deterministic.
    """
    h = np.asarray(headways, dtype=float)
    h = h[np.isfinite(h) & (h > 0)]
    if h.size <= 1:
        return 1.0
    p = h / h.sum()
    entropy = -float(np.sum(p * np.log(p))) / float(np.log(h.size))
    return float(np.clip(entropy, 0.0, 1.0))


def audited_supply(frequency_per_hour: float, headways: np.ndarray) -> tuple[float, float]:
    """O = Frequency * (1 - H_irregularity).

    H_irregularity = 1 - normalized Shannon entropy of headway shares.
    Returns (O, H_irregularity).
    """
    if frequency_per_hour <= 0:
        raise ValueError("Frequency must be strictly positive")
    h_norm = normalized_headway_entropy(headways)
    h_irregularity = 1.0 - h_norm
    supply = float(frequency_per_hour * (1.0 - h_irregularity))
    return supply, h_irregularity


def mobility_desert_index(p_scaled: np.ndarray, inefficiency: np.ndarray,
                          supply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MDI_raw = P * I / O, followed by 0-100 scaling."""
    p_scaled = np.asarray(p_scaled, dtype=float)
    inefficiency = np.asarray(inefficiency, dtype=float)
    supply = np.asarray(supply, dtype=float)
    if np.any(supply <= 0):
        raise ValueError("Audited supply must be strictly positive")
    raw = p_scaled * inefficiency / supply
    return raw, minmax_0_100(raw)

