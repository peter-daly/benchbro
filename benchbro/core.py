from __future__ import annotations

import gc
import csv
import json
import os
import platform
import statistics
import sys
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Literal

MetricType = Literal["time", "memory"]
GcControl = Literal["inherit", "disable_during_measure"]


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
    settings: BenchmarkSettings = field(default_factory=BenchmarkSettings)


@dataclass
class BenchmarkResult:
    case_name: str
    benchmark_name: str
    case_type: str
    metric_type: MetricType
    iterations: int
    repeats: int
    metrics: dict[str, float]


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


def _call_case(case: BenchmarkCase) -> Any:
    if case.input_func is None:
        return case.func()
    value = case.input_func()
    return case.func(value)


def _time_metrics(case: BenchmarkCase) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(case.settings.repeats):
        start = perf_counter_ns()
        for _ in range(case.settings.min_iterations):
            _call_case(case)
        elapsed_s = (perf_counter_ns() - start) / 1_000_000_000
        samples.append(elapsed_s / case.settings.min_iterations)

    mean_s = statistics.fmean(samples)
    median_s = statistics.median(samples)
    p95_s = sorted(samples)[int((len(samples) - 1) * 0.95)]
    stddev_s = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    ops_per_sec = (1.0 / mean_s) if mean_s > 0 else 0.0

    return {
        "mean_s": mean_s,
        "median_s": median_s,
        "p95_s": p95_s,
        "stddev_s": stddev_s,
        "ops_per_sec": ops_per_sec,
    }


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


def _with_overrides(
    original: BenchmarkCase,
    repeats: int | None,
    warmup: int | None,
    min_iterations: int | None,
) -> BenchmarkCase:
    settings = BenchmarkSettings(
        warmup_iterations=warmup if warmup is not None else original.settings.warmup_iterations,
        min_iterations=min_iterations if min_iterations is not None else original.settings.min_iterations,
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
        case = _with_overrides(original, repeats=repeats, warmup=warmup, min_iterations=min_iterations)

        for _ in range(case.settings.warmup_iterations):
            _call_case(case)

        with _gc_control_context(case.settings.gc_control):
            metrics = _time_metrics(case) if case.metric_type == "time" else _memory_metrics(case)

        results.append(
            BenchmarkResult(
                case_name=case.case_name,
                benchmark_name=case.benchmark_name,
                case_type=case.case_type,
                metric_type=case.metric_type,
                iterations=case.settings.min_iterations,
                repeats=case.settings.repeats,
                metrics=metrics,
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
            case for case in selected if case.case_name in wanted or case.benchmark_name in wanted
        ]
    if tags:
        wanted_tags = set(tags)
        selected = [case for case in selected if wanted_tags.intersection(case.tags)]
    return selected


def _primary_metric(result: BenchmarkResult) -> tuple[str, float]:
    if result.metric_type == "time":
        return "mean_s", result.metrics["mean_s"]
    return "peak_alloc_bytes", result.metrics["peak_alloc_bytes"]


def compare_runs(
    baseline: BenchmarkRun,
    current: BenchmarkRun,
    regression_threshold_pct: float = 5.0,
) -> list[Regression]:
    baseline_map = {(r.case_name, r.benchmark_name, r.metric_type): r for r in baseline.benchmarks}
    regressions: list[Regression] = []

    for cur in current.benchmarks:
        base = baseline_map.get((cur.case_name, cur.benchmark_name, cur.metric_type))
        if base is None:
            continue

        metric_name, base_value = _primary_metric(base)
        _, cur_value = _primary_metric(cur)
        if base_value == 0:
            continue

        percent_change = ((cur_value - base_value) / base_value) * 100.0
        is_regression = percent_change > regression_threshold_pct
        regressions.append(
            Regression(
                case_name=cur.case_name,
                benchmark_name=cur.benchmark_name,
                metric_name=metric_name,
                baseline_value=base_value,
                current_value=cur_value,
                percent_change=percent_change,
                is_regression=is_regression,
            )
        )

    return regressions


def write_json(path: str | Path, run: BenchmarkRun) -> None:
    payload = asdict(run)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> BenchmarkRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    benchmarks = [BenchmarkResult(**item) for item in payload.get("benchmarks", [])]
    return BenchmarkRun(
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
        python_version=payload["python_version"],
        platform=payload["platform"],
        environment=payload.get("environment", {}),
        benchmarks=benchmarks,
    )


def write_csv(path: str | Path, run: BenchmarkRun) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["case", "benchmark", "metric_type", "metric_name", "value"])
        for result in run.benchmarks:
            for metric_name, value in result.metrics.items():
                writer.writerow(
                    [
                        result.case_name,
                        result.benchmark_name,
                        result.metric_type,
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
        "| case | benchmark | metric_type | mean_s | median_s | p95_s | stddev_s | ops_per_sec | peak_alloc_bytes | net_alloc_bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count": str(os.cpu_count() or 0),
    }


def _format_markdown_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"
