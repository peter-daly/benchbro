from __future__ import annotations

import asyncio
import gc
import csv
import inspect
import json
import os
import platform
import statistics
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Literal

MetricType = Literal["time", "memory"]
GcControl = Literal["inherit", "disable_during_measure"]

TIME_COMPARISON_METRICS = {
    "mean_s",
    "median_s",
    "iqr_s",
    "p95_s",
    "stddev_s",
    "ops_per_sec",
}
MEMORY_COMPARISON_METRICS = {
    "peak_alloc_bytes",
    "net_alloc_bytes",
    "peak_alloc_bytes_max",
}


@dataclass
class BenchmarkSettings:
    warmup_iterations: int = 5
    min_iterations: int = 50
    repeats: int = 5
    gc_control: GcControl = "disable_during_measure"


@dataclass
class BenchmarkCase:
    case_name: str
    benchmark_name: str
    func: Callable[..., Any]
    case_type: str = "cpu"
    metric_type: MetricType = "time"
    tags: tuple[str, ...] = ()
    input_func: Callable[[], Any] | None = None
    comparison_metric: str | None = None
    regression_threshold_pct: float = 100.0
    warning_threshold_pct: float = 50.0
    settings: BenchmarkSettings = field(default_factory=BenchmarkSettings)


@dataclass
class BenchmarkResult:
    case_name: str
    benchmark_name: str
    case_type: str
    metric_type: MetricType
    gc_control: GcControl
    regression_threshold_pct: float
    iterations: int
    repeats: int
    metrics: dict[str, float]
    environment: dict[str, str] = field(default_factory=dict)
    warning_threshold_pct: float = 50.0
    comparison_metric: str | None = None
    time_samples_s: list[float] = field(default_factory=list)


@dataclass
class BenchmarkRun:
    started_at: str
    finished_at: str
    python_version: str
    platform: str
    environment: dict[str, str]
    benchmarks: list[BenchmarkResult] = field(default_factory=list)


@dataclass
class Regression:
    case_name: str
    benchmark_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    percent_change: float
    warning_threshold_pct: float
    threshold_pct: float
    is_warning: bool
    is_regression: bool


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._cases: list[BenchmarkCase] = []

    def register(self, case: BenchmarkCase) -> BenchmarkCase:
        self._cases.append(case)
        return case

    def all_cases(self) -> list[BenchmarkCase]:
        return list(self._cases)

    def clear(self) -> None:
        self._cases.clear()


def _run_awaitable(value: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "benchbro cannot execute async benchmarks from within an active event loop"
    )


def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return _run_awaitable(value)
    return value


def _call_case(case: BenchmarkCase) -> Any:
    if case.input_func is None:
        result = case.func()
        return _resolve_maybe_awaitable(result)
    input_value = _resolve_maybe_awaitable(case.input_func())
    result = case.func(input_value)
    return _resolve_maybe_awaitable(result)


def _time_metrics(case: BenchmarkCase) -> tuple[dict[str, float], list[float]]:
    samples: list[float] = []
    for _ in range(case.settings.repeats):
        start = perf_counter_ns()
        for _ in range(case.settings.min_iterations):
            _call_case(case)
        elapsed_s = (perf_counter_ns() - start) / 1_000_000_000
        samples.append(elapsed_s / case.settings.min_iterations)

    sorted_samples = sorted(samples)
    last_index = len(sorted_samples) - 1
    mean_s = statistics.fmean(sorted_samples)
    median_s = statistics.median(sorted_samples)
    p80_s = sorted_samples[int(last_index * 0.80)]
    p90_s = sorted_samples[int(last_index * 0.90)]
    p95_s = sorted_samples[int(last_index * 0.95)]
    p99_s = sorted_samples[int(last_index * 0.99)]
    q1_s = sorted_samples[int(last_index * 0.25)]
    q3_s = sorted_samples[int(last_index * 0.75)]
    iqr_s = q3_s - q1_s
    stddev_s = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    ops_per_sec = (1.0 / mean_s) if mean_s > 0 else 0.0

    return {
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
        "ops_per_sec": ops_per_sec,
    }, samples


def _memory_metrics(case: BenchmarkCase) -> dict[str, float]:
    peaks: list[float] = []
    nets: list[float] = []
    for _ in range(case.settings.repeats):
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        for _ in range(case.settings.min_iterations):
            _call_case(case)
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peaks.append(float(peak))
        nets.append(float(after_current - before_current))

    return {
        "peak_alloc_bytes": statistics.fmean(peaks),
        "net_alloc_bytes": statistics.fmean(nets),
        "peak_alloc_bytes_max": max(peaks) if peaks else 0.0,
    }


def default_comparison_metric(metric_type: MetricType) -> str:
    if metric_type == "time":
        return "median_s"
    return "peak_alloc_bytes"


def is_valid_comparison_metric(metric_type: MetricType, metric_name: str) -> bool:
    if metric_type == "time":
        return metric_name in TIME_COMPARISON_METRICS
    return metric_name in MEMORY_COMPARISON_METRICS


def _with_overrides(
    original: BenchmarkCase,
    repeats: int | None,
    warmup: int | None,
    min_iterations: int | None,
) -> BenchmarkCase:
    settings = BenchmarkSettings(
        warmup_iterations=warmup
        if warmup is not None
        else original.settings.warmup_iterations,
        min_iterations=min_iterations
        if min_iterations is not None
        else original.settings.min_iterations,
        repeats=repeats if repeats is not None else original.settings.repeats,
        gc_control=original.settings.gc_control,
    )
    return BenchmarkCase(
        case_name=original.case_name,
        benchmark_name=original.benchmark_name,
        func=original.func,
        case_type=original.case_type,
        metric_type=original.metric_type,
        tags=original.tags,
        input_func=original.input_func,
        comparison_metric=original.comparison_metric,
        regression_threshold_pct=original.regression_threshold_pct,
        warning_threshold_pct=original.warning_threshold_pct,
        settings=settings,
    )


def run_cases(
    cases: list[BenchmarkCase],
    repeats: int | None = None,
    warmup: int | None = None,
    min_iterations: int | None = None,
) -> BenchmarkRun:
    started = datetime.now(timezone.utc).isoformat()
    results: list[BenchmarkResult] = []
    environment = _collect_environment_metadata()

    for original in cases:
        case = _with_overrides(
            original, repeats=repeats, warmup=warmup, min_iterations=min_iterations
        )

        for _ in range(case.settings.warmup_iterations):
            _call_case(case)

        with _gc_control_context(case.settings.gc_control):
            if case.metric_type == "time":
                metrics, time_samples_s = _time_metrics(case)
            else:
                metrics = _memory_metrics(case)
                time_samples_s = []

        results.append(
            BenchmarkResult(
                case_name=case.case_name,
                benchmark_name=case.benchmark_name,
                case_type=case.case_type,
                metric_type=case.metric_type,
                gc_control=case.settings.gc_control,
                comparison_metric=case.comparison_metric,
                regression_threshold_pct=case.regression_threshold_pct,
                environment=environment.copy(),
                iterations=case.settings.min_iterations,
                repeats=case.settings.repeats,
                metrics=metrics,
                warning_threshold_pct=case.warning_threshold_pct,
                time_samples_s=time_samples_s,
            )
        )

    finished = datetime.now(timezone.utc).isoformat()
    return BenchmarkRun(
        started_at=started,
        finished_at=finished,
        python_version=environment["python_version"],
        platform=environment["platform"],
        environment=environment,
        benchmarks=results,
    )


@contextmanager
def _gc_control_context(mode: GcControl):
    if mode == "inherit":
        yield
        return

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def filter_cases(
    cases: list[BenchmarkCase],
    names: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[BenchmarkCase]:
    selected = cases
    if names:
        wanted = set(names)
        selected = [
            case
            for case in selected
            if case.case_name in wanted or case.benchmark_name in wanted
        ]
    if tags:
        wanted_tags = set(tags)
        selected = [case for case in selected if wanted_tags.intersection(case.tags)]
    return selected


def _comparison_metric_candidates(
    metric_type: MetricType, preferred: str | None
) -> list[str]:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    default_metric = default_comparison_metric(metric_type)
    if default_metric not in candidates:
        candidates.append(default_metric)
    if metric_type == "time" and "mean_s" not in candidates:
        # legacy fallback for older baselines that may not have median_s
        candidates.append("mean_s")
    return candidates


def compare_runs(
    baseline: BenchmarkRun,
    current: BenchmarkRun,
) -> list[Regression]:
    baseline_map = {
        (r.case_name, r.benchmark_name, r.metric_type): r for r in baseline.benchmarks
    }
    regressions: list[Regression] = []

    for cur in current.benchmarks:
        base = baseline_map.get((cur.case_name, cur.benchmark_name, cur.metric_type))
        if base is None:
            continue

        preferred_metric = cur.comparison_metric
        metric_name: str | None = None
        base_value: float | None = None
        cur_value: float | None = None
        for candidate in _comparison_metric_candidates(
            cur.metric_type, preferred_metric
        ):
            if candidate in base.metrics and candidate in cur.metrics:
                metric_name = candidate
                base_value = base.metrics[candidate]
                cur_value = cur.metrics[candidate]
                break
        if metric_name is None or base_value is None or cur_value is None:
            continue
        if base_value == 0:
            continue

        percent_change = ((cur_value - base_value) / base_value) * 100.0
        warning_threshold_pct = cur.warning_threshold_pct
        threshold_pct = cur.regression_threshold_pct
        if metric_name == "ops_per_sec":
            # For throughput metrics, lower values are regressions.
            is_regression = (percent_change + threshold_pct) < -1e-12
            is_warning = (
                percent_change + warning_threshold_pct
            ) < -1e-12 and not is_regression
        else:
            # Keep strict threshold semantics while avoiding float rounding artifacts at equality boundaries.
            is_regression = (percent_change - threshold_pct) > 1e-12
            is_warning = (
                percent_change - warning_threshold_pct
            ) > 1e-12 and not is_regression
        regressions.append(
            Regression(
                case_name=cur.case_name,
                benchmark_name=cur.benchmark_name,
                metric_name=metric_name,
                baseline_value=base_value,
                current_value=cur_value,
                percent_change=percent_change,
                warning_threshold_pct=warning_threshold_pct,
                threshold_pct=threshold_pct,
                is_warning=is_warning,
                is_regression=is_regression,
            )
        )

    return regressions


def write_json(path: str | Path, run: BenchmarkRun) -> None:
    payload = asdict(run)
    for benchmark_payload in payload.get("benchmarks", []):
        benchmark_payload.pop("time_samples_s", None)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> BenchmarkRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    run_environment = payload.get("environment", {})
    benchmarks = [
        BenchmarkResult(
            case_name=item["case_name"],
            benchmark_name=item["benchmark_name"],
            case_type=item["case_type"],
            metric_type=item["metric_type"],
            gc_control=item.get("gc_control", "disable_during_measure"),
            comparison_metric=item.get("comparison_metric"),
            regression_threshold_pct=item.get("regression_threshold_pct", 100.0),
            environment=item.get("environment", run_environment),
            iterations=item["iterations"],
            repeats=item["repeats"],
            metrics=item["metrics"],
            warning_threshold_pct=item.get(
                "warning_threshold_pct", item.get("regression_warning_pct", 50.0)
            ),
            time_samples_s=item.get("time_samples_s", []),
        )
        for item in payload.get("benchmarks", [])
    ]
    return BenchmarkRun(
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
        python_version=payload["python_version"],
        platform=payload["platform"],
        environment=run_environment,
        benchmarks=benchmarks,
    )


def write_csv(path: str | Path, run: BenchmarkRun) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["case", "benchmark", "metric_type", "gc_control", "metric_name", "value"]
        )
        for result in run.benchmarks:
            for metric_name, value in result.metrics.items():
                writer.writerow(
                    [
                        result.case_name,
                        result.benchmark_name,
                        result.metric_type,
                        result.gc_control,
                        metric_name,
                        value,
                    ]
                )


def write_markdown(path: str | Path, run: BenchmarkRun) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Benchbro Results",
        "",
        f"- started_at: `{run.started_at}`",
        f"- finished_at: `{run.finished_at}`",
        f"- python_version: `{run.python_version}`",
        f"- platform: `{run.platform}`",
        "",
        "| case | benchmark | metric_type | mean_s | median_s | iqr_s | p95_s | stddev_s | ops_per_sec | peak_alloc_bytes | net_alloc_bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in run.benchmarks:
        metrics = result.metrics
        lines.append(
            "| "
            f"{result.case_name} | "
            f"{result.benchmark_name} | "
            f"{result.metric_type} | "
            f"{_format_markdown_metric(metrics.get('mean_s'))} | "
            f"{_format_markdown_metric(metrics.get('median_s'))} | "
            f"{_format_markdown_metric(metrics.get('iqr_s'))} | "
            f"{_format_markdown_metric(metrics.get('p95_s'))} | "
            f"{_format_markdown_metric(metrics.get('stddev_s'))} | "
            f"{_format_markdown_metric(metrics.get('ops_per_sec'))} | "
            f"{_format_markdown_metric(metrics.get('peak_alloc_bytes'))} | "
            f"{_format_markdown_metric(metrics.get('net_alloc_bytes'))} |"
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_environment_metadata() -> dict[str, str]:
    uname = platform.uname()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count": str(os.cpu_count() or 0),
        "gc_enabled_at_start": str(gc.isenabled()),
        "gc_threshold": ",".join(str(value) for value in gc.get_threshold()),
    }


def _format_markdown_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"
