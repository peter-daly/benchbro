from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from time import perf_counter


def confidence_interval_95(samples: list[float]) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(samples)
    if len(samples) == 1:
        return mean, mean, 0.0
    standard_error = statistics.stdev(samples) / math.sqrt(len(samples))
    margin = 1.96 * standard_error
    return mean - margin, mean + margin, margin


def outlier_indexes(samples: list[float]) -> set[int]:
    if len(samples) < 4:
        return set()
    q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    iqr = q3 - q1
    low = q1 - (1.5 * iqr)
    high = q3 + (1.5 * iqr)
    return {
        index for index, sample in enumerate(samples) if sample < low or sample > high
    }


def time_metrics(
    samples: list[float], noise_threshold_pct: float
) -> tuple[dict[str, float], int, bool]:
    if not samples:
        raise ValueError("At least one timing sample is required")
    sorted_samples = sorted(samples)
    mean_s = statistics.fmean(sorted_samples)
    median_s = statistics.median(sorted_samples)
    if len(sorted_samples) == 1:
        p80_s = p90_s = p95_s = p99_s = sorted_samples[0]
        q1_s = q3_s = sorted_samples[0]
    else:
        percentiles = statistics.quantiles(sorted_samples, n=100, method="inclusive")
        p80_s = percentiles[79]
        p90_s = percentiles[89]
        p95_s = percentiles[94]
        p99_s = percentiles[98]
        q1_s, _, q3_s = statistics.quantiles(sorted_samples, n=4, method="inclusive")
    iqr_s = q3_s - q1_s
    variance_s2 = statistics.pvariance(samples) if len(samples) > 1 else 0.0
    stddev_s = math.sqrt(variance_s2)
    cv_pct = (stddev_s / mean_s * 100.0) if mean_s > 0 else 0.0
    ci_low, ci_high, margin = confidence_interval_95(samples)
    relative_margin_pct = (margin / mean_s * 100.0) if mean_s > 0 else 0.0
    outliers = outlier_indexes(samples)
    is_noisy = cv_pct > noise_threshold_pct or bool(outliers)
    return (
        {
            "mean_s": mean_s,
            "median_s": median_s,
            "p80_s": p80_s,
            "p90_s": p90_s,
            "iqr_s": iqr_s,
            "p95_s": p95_s,
            "q1_s": q1_s,
            "q3_s": q3_s,
            "p99_s": p99_s,
            "stddev_s": stddev_s,
            "variance_s2": variance_s2,
            "standard_error_s": margin / 1.96,
            "ci95_low_s": ci_low,
            "ci95_high_s": ci_high,
            "relative_margin_pct": relative_margin_pct,
            "cv_pct": cv_pct,
            "ops_per_sec": (1.0 / mean_s) if mean_s > 0 else 0.0,
            "sample_count": float(len(samples)),
            "outlier_count": float(len(outliers)),
        },
        len(outliers),
        is_noisy,
    )


def should_stop_adaptive(
    samples: list[float],
    *,
    min_repeats: int,
    min_time_s: float,
    max_time_s: float | None,
    target_relative_margin_pct: float,
    elapsed_s: float,
) -> bool:
    if len(samples) < min_repeats or elapsed_s < min_time_s:
        return False
    if max_time_s is not None and elapsed_s >= max_time_s:
        return True
    _, _, margin = confidence_interval_95(samples)
    mean = statistics.fmean(samples)
    relative_margin = (margin / mean * 100.0) if mean > 0 else 0.0
    return relative_margin <= target_relative_margin_pct


def collect_timing_samples(
    measure_repeat: Callable[[], float],
    *,
    repeats: int,
    adaptive: bool,
    min_repeats: int,
    min_time_s: float,
    max_time_s: float | None,
    target_relative_margin_pct: float,
) -> list[float]:
    samples: list[float] = []
    started = perf_counter()
    for _ in range(repeats):
        samples.append(measure_repeat())
        if adaptive and should_stop_adaptive(
            samples,
            min_repeats=min_repeats,
            min_time_s=min_time_s,
            max_time_s=max_time_s,
            target_relative_margin_pct=target_relative_margin_pct,
            elapsed_s=perf_counter() - started,
        ):
            break
    return samples
