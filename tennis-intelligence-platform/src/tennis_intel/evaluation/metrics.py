"""
metrics.py — evaluation metrics for probability-quality assessment, consistent with the
methodology from the project owner's prior World Cup project (bootstrap confidence
intervals, calibration analysis, not just accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import brier_score_loss


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(sk_log_loss(y_true, y_prob, labels=[0, 1]))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def bootstrap_metric(
    y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_bootstrap: int = 1000,
    ci: float = 0.95, random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap confidence interval for any metric_fn(y_true, y_prob) -> float, by
    resampling match indices with replacement. Same technique used in the World Cup
    project for reporting metric uncertainty rather than a single point estimate."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_prob)

    boot_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_prob[idx])

    alpha = (1 - ci) / 2
    lower = float(np.quantile(boot_values, alpha))
    upper = float(np.quantile(boot_values, 1 - alpha))
    return BootstrapResult(point_estimate=point_estimate, ci_lower=lower, ci_upper=upper, n_bootstrap=n_bootstrap)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: for each predicted-probability bucket, the observed win
    rate should be close to the bucket's midpoint for a well-calibrated model. This is the
    check that matters more than accuracy for a win-probability product (per the project's
    own blueprint, Section 9)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bucket = np.digitize(y_prob, bins) - 1
    bucket = np.clip(bucket, 0, n_bins - 1)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bucket": bucket})
    table = df.groupby("bucket").agg(
        n=("y_true", "size"),
        mean_predicted=("y_prob", "mean"),
        observed_win_rate=("y_true", "mean"),
    ).reset_index()
    table["calibration_gap"] = table["observed_win_rate"] - table["mean_predicted"]
    return table


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE: the weighted average absolute calibration gap across probability buckets, weighted
    by how many predictions fall in each bucket. Lower is better (0 = perfectly calibrated).
    Standard metric for evaluating whether a model's probabilities are trustworthy, not just
    whether its point predictions are accurate.
    """
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    total_n = table["n"].sum()
    if total_n == 0:
        return float("nan")
    weighted_gap = (table["n"] * table["calibration_gap"].abs()).sum() / total_n
    return float(weighted_gap)


def sharpness(y_prob: np.ndarray) -> float:
    """
    Mean absolute distance from 0.5 — how decisive/confident the predictions are, distinct
    from whether they're CORRECT (that's calibration's job). A model that always predicts
    0.5 is perfectly uninformative (sharpness=0) even if perfectly calibrated in aggregate.
    For a live win-probability engine, sharpness should generally increase as a match
    progresses (more information available -> more confident, correct predictions).
    """
    return float(np.mean(np.abs(np.asarray(y_prob) - 0.5)))


@dataclass
class PairedBootstrapResult:
    metric_name: str
    point_estimate_diff: float  # metric(model_a) - metric(model_b); negative = A better for loss metrics
    ci_lower: float
    ci_upper: float
    zero_in_ci: bool
    n_bootstrap: int


def paired_bootstrap_diff(
    y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray, metric_fn,
    metric_name: str = "metric", n_bootstrap: int = 1000, ci: float = 0.95,
    random_state: int = 42,
) -> PairedBootstrapResult:
    """
    PAIRED bootstrap for comparing two models on the SAME evaluation points: resamples
    point indices with replacement and recomputes metric(A) - metric(B) on each resample,
    so per-point difficulty is shared between the two models within every resample (much
    tighter CI than two independent bootstraps). Same resampling style as bootstrap_metric.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    n = len(y_true)

    point_diff = metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b)

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = metric_fn(y_true[idx], y_prob_a[idx]) - metric_fn(y_true[idx], y_prob_b[idx])

    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    return PairedBootstrapResult(
        metric_name=metric_name, point_estimate_diff=float(point_diff),
        ci_lower=lo, ci_upper=hi, zero_in_ci=(lo <= 0.0 <= hi), n_bootstrap=n_bootstrap,
    )