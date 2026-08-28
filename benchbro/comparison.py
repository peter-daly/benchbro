from __future__ import annotations

import math
from statistics import NormalDist

from benchbro.models import (
    BenchmarkRun,
    MetricType,
    Regression,
    default_comparison_metric,
)


def comparison_environment_mismatches(
    baseline: BenchmarkRun, current: BenchmarkRun
) -> dict[str, tuple[str, str]]:
    """Return environment fields that make two runs unsafe to compare."""

    baseline_environment = {
        "python_version": baseline.python_version,
        "platform": baseline.platform,
        **baseline.environment,
    }
    current_environment = {
        "python_version": current.python_version,
        "platform": current.platform,
        **current.environment,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for key in ("python_implementation", "system", "machine"):
        baseline_value = baseline_environment.get(key)
        current_value = current_environment.get(key)
        if baseline_value and current_value and baseline_value != current_value:
            mismatches[key] = (baseline_value, current_value)

    baseline_python = baseline_environment.get("python_version", "")
    current_python = current_environment.get("python_version", "")
    baseline_family = ".".join(baseline_python.split(".")[:2])
    current_family = ".".join(current_python.split(".")[:2])
    if baseline_family and current_family and baseline_family != current_family:
        mismatches["python_version"] = (baseline_python, current_python)
    return mismatches


def _comparison_metric_candidates(
    metric_type: MetricType, preferred: str | None
) -> list[str]:
    candidates = [preferred] if preferred else []
    default_metric = default_comparison_metric(metric_type)
    if default_metric not in candidates:
        candidates.append(default_metric)
    if metric_type == "time" and "mean_s" not in candidates:
        candidates.append("mean_s")
    return candidates


def _regression_confidence_pct(
    *,
    metric_name: str,
    baseline_value: float,
    current_value: float,
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    threshold_pct: float,
    inverse: bool,
) -> float:
    baseline_samples = baseline_metrics.get("sample_count")
    current_samples = current_metrics.get("sample_count")
    if (
        baseline_samples is not None
        and current_samples is not None
        and (baseline_samples < 2 or current_samples < 2)
    ):
        return 0.0
    standard_error_name = {
        "peak_alloc_bytes": "peak_alloc_standard_error_bytes",
        "net_alloc_bytes": "net_alloc_standard_error_bytes",
    }.get(metric_name, "standard_error_s")
    baseline_se = baseline_metrics.get(standard_error_name)
    current_se = current_metrics.get(standard_error_name)
    if baseline_se is None or current_se is None:
        return 100.0
    combined_se = math.sqrt((baseline_se**2) + (current_se**2))
    boundary = baseline_value * (
        1.0 - (threshold_pct / 100.0) if inverse else 1.0 + (threshold_pct / 100.0)
    )
    margin = boundary - current_value if inverse else current_value - boundary
    if combined_se == 0:
        return 100.0 if margin > 0 else 0.0
    return NormalDist().cdf(margin / combined_se) * 100.0


def compare_runs(
    baseline: BenchmarkRun,
    current: BenchmarkRun,
    *,
    allow_environment_mismatch: bool = False,
    confidence_threshold_pct: float = 95.0,
) -> list[Regression]:
    """Compare matching results and classify their performance change.

    Args:
        baseline: Reference run.
        current: Candidate run.
        allow_environment_mismatch: Compare even when runtime or machine fields
            differ.
        confidence_threshold_pct: Minimum confidence required for a regression.

    Returns:
        One classification for each current result that has a baseline match and
        a usable comparison metric.

    Raises:
        ValueError: If environments differ and the override is not enabled.
    """

    mismatches = comparison_environment_mismatches(baseline, current)
    if mismatches and not allow_environment_mismatch:
        details = ", ".join(
            f"{key}: {before!r} -> {after!r}"
            for key, (before, after) in mismatches.items()
        )
        raise ValueError(
            "Benchmark environments are incompatible for comparison "
            f"({details}). Pass allow_environment_mismatch=True to override."
        )

    baseline_map = {
        (item.case_name, item.benchmark_name, item.metric_type): item
        for item in baseline.benchmarks
    }
    regressions: list[Regression] = []
    for current_result in current.benchmarks:
        baseline_result = baseline_map.get(
            (
                current_result.case_name,
                current_result.benchmark_name,
                current_result.metric_type,
            )
        )
        if baseline_result is None:
            continue
        metric_name = None
        baseline_value = None
        current_value = None
        for candidate in _comparison_metric_candidates(
            current_result.metric_type, current_result.comparison_metric
        ):
            if (
                candidate in baseline_result.metrics
                and candidate in current_result.metrics
            ):
                metric_name = candidate
                baseline_value = baseline_result.metrics[candidate]
                current_value = current_result.metrics[candidate]
                break
        if (
            metric_name is None
            or baseline_value is None
            or current_value is None
            or baseline_value == 0
        ):
            continue

        percent_change = ((current_value - baseline_value) / baseline_value) * 100.0
        warning_threshold = current_result.warning_threshold_pct
        threshold = current_result.regression_threshold_pct
        inverse = metric_name == "ops_per_sec"
        crosses_error = (
            (percent_change + threshold) < -1e-12
            if inverse
            else (percent_change - threshold) > 1e-12
        )
        crosses_warning = (
            (percent_change + warning_threshold) < -1e-12
            if inverse
            else (percent_change - warning_threshold) > 1e-12
        )
        confidence = (
            _regression_confidence_pct(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                baseline_metrics=baseline_result.metrics,
                current_metrics=current_result.metrics,
                threshold_pct=threshold,
                inverse=inverse,
            )
            if crosses_error
            else 100.0
        )
        is_regression = crosses_error and confidence >= confidence_threshold_pct
        is_likely = crosses_error and not is_regression and confidence >= 80.0
        is_warning = (crosses_warning or is_likely) and not is_regression
        if is_regression:
            status = "regression"
        elif is_likely:
            status = "likely_regression"
        elif crosses_error:
            status = "inconclusive"
        elif is_warning:
            status = "warning"
        elif current_result.is_noisy:
            status = "noisy"
        elif (percent_change > 0) == inverse and abs(percent_change) > 1e-12:
            status = "improvement"
        else:
            status = "stable"
        regressions.append(
            Regression(
                case_name=current_result.case_name,
                benchmark_name=current_result.benchmark_name,
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                percent_change=percent_change,
                warning_threshold_pct=warning_threshold,
                threshold_pct=threshold,
                is_warning=is_warning,
                is_regression=is_regression,
                status=status,
                confidence_pct=confidence,
            )
        )
    return regressions
