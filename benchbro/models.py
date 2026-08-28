from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2

MetricType = Literal["time", "memory"]
GcControl = Literal["inherit", "disable_during_measure"]
SystemScope = Literal["iteration", "benchmark", "session", "function", "module"]
CanonicalSystemScope = Literal["iteration", "benchmark", "session"]
BenchmarkKind = Literal["python", "subprocess"]
CommandStream = Literal["devnull", "inherit"]
IsolationMode = Literal["in_process", "process"]
TimingMode = Literal["include", "exclude"]

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


def normalize_system_scope(scope: SystemScope) -> CanonicalSystemScope:
    aliases: dict[str, CanonicalSystemScope] = {
        "iteration": "iteration",
        "benchmark": "benchmark",
        "session": "session",
        "function": "benchmark",
        "module": "session",
    }
    try:
        return aliases[scope]
    except KeyError as exc:
        raise ValueError(f"Invalid system scope '{scope}'") from exc


def _validate_iteration_count(name: str, value: int, *, allow_zero: bool) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "a non-negative integer" if allow_zero else "a positive integer"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_positive_number_or_none(name: str, value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number or None")


def default_comparison_metric(metric_type: MetricType) -> str:
    if metric_type == "time":
        return "median_s"
    return "peak_alloc_bytes"


def is_valid_comparison_metric(metric_type: MetricType, metric_name: str) -> bool:
    if metric_type == "time":
        return metric_name in TIME_COMPARISON_METRICS
    return metric_name in MEMORY_COMPARISON_METRICS


def validate_case_configuration(
    *,
    case_name: str,
    benchmark_name: str | None = None,
    case_type: str,
    metric_type: str,
    gc_control: str,
    comparison_metric: str | None,
    regression_threshold_pct: float,
    warning_threshold_pct: float,
) -> None:
    if not isinstance(case_name, str) or not case_name.strip():
        raise ValueError("Case name must be a non-empty string")
    if benchmark_name is not None and (
        not isinstance(benchmark_name, str) or not benchmark_name.strip()
    ):
        raise ValueError("Benchmark name must be a non-empty string")
    if not isinstance(case_type, str) or not case_type.strip():
        raise ValueError("case_type must be a non-empty string")
    if metric_type not in ("time", "memory"):
        raise ValueError("metric_type must be 'time' or 'memory'")
    if gc_control not in ("inherit", "disable_during_measure"):
        raise ValueError("gc_control must be 'inherit' or 'disable_during_measure'")
    if comparison_metric is not None and not is_valid_comparison_metric(
        metric_type, comparison_metric
    ):
        raise ValueError(
            f"Invalid comparison metric '{comparison_metric}' for metric_type '{metric_type}'"
        )
    _validate_non_negative_number("regression_threshold_pct", regression_threshold_pct)
    _validate_non_negative_number("warning_threshold_pct", warning_threshold_pct)


@dataclass
class BenchmarkSettings:
    """Validated measurement, lifecycle, isolation, and profiling settings."""

    warmup_iterations: int = 5
    min_iterations: int = 50
    repeats: int = 20
    gc_control: GcControl = "disable_during_measure"
    adaptive: bool = False
    min_repeats: int = 5
    min_time_s: float = 0.0
    max_time_s: float | None = None
    target_relative_margin_pct: float = 2.0
    noise_threshold_pct: float = 10.0
    setup_timing: TimingMode = "include"
    teardown_timing: TimingMode = "exclude"
    timeout_s: float | None = None
    isolation: IsolationMode = "in_process"
    cpu_affinity: tuple[int, ...] | None = None
    stabilization_delay_s: float = 0.0
    profiler: str | None = None
    profile_output: str | None = None

    def __post_init__(self) -> None:
        _validate_iteration_count(
            "warmup_iterations", self.warmup_iterations, allow_zero=True
        )
        _validate_iteration_count(
            "min_iterations", self.min_iterations, allow_zero=False
        )
        _validate_iteration_count("repeats", self.repeats, allow_zero=False)
        _validate_iteration_count("min_repeats", self.min_repeats, allow_zero=False)
        if self.min_repeats > self.repeats:
            raise ValueError("min_repeats cannot be greater than repeats")
        if self.gc_control not in ("inherit", "disable_during_measure"):
            raise ValueError("gc_control must be 'inherit' or 'disable_during_measure'")
        if not isinstance(self.adaptive, bool):
            raise ValueError("adaptive must be a bool")
        _validate_non_negative_number("min_time_s", self.min_time_s)
        _validate_positive_number_or_none("max_time_s", self.max_time_s)
        if self.max_time_s is not None and self.max_time_s < self.min_time_s:
            raise ValueError("max_time_s cannot be less than min_time_s")
        _validate_non_negative_number(
            "target_relative_margin_pct", self.target_relative_margin_pct
        )
        _validate_non_negative_number("noise_threshold_pct", self.noise_threshold_pct)
        if self.setup_timing not in ("include", "exclude"):
            raise ValueError("setup_timing must be 'include' or 'exclude'")
        if self.teardown_timing not in ("include", "exclude"):
            raise ValueError("teardown_timing must be 'include' or 'exclude'")
        _validate_positive_number_or_none("timeout_s", self.timeout_s)
        if self.isolation not in ("in_process", "process"):
            raise ValueError("isolation must be 'in_process' or 'process'")
        if self.cpu_affinity is not None:
            if not self.cpu_affinity or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                for cpu in self.cpu_affinity
            ):
                raise ValueError(
                    "cpu_affinity must contain one or more non-negative integers"
                )
            self.cpu_affinity = tuple(dict.fromkeys(self.cpu_affinity))
        _validate_non_negative_number(
            "stabilization_delay_s", self.stabilization_delay_s
        )
        if self.profiler is not None and (
            not isinstance(self.profiler, str) or not self.profiler.strip()
        ):
            raise ValueError("profiler must be a non-empty string or None")
        if self.profile_output is not None and (
            not isinstance(self.profile_output, str) or not self.profile_output.strip()
        ):
            raise ValueError("profile_output must be a non-empty string or None")


@dataclass
class BenchmarkCase:
    case_name: str
    benchmark_name: str
    func: Callable[..., Any]
    kind: BenchmarkKind = "python"
    case_type: str = "cpu"
    metric_type: MetricType = "time"
    tags: tuple[str, ...] = ()
    input_func: Callable[[], Any] | None = None
    input_name: str | None = None
    comparison_metric: str | None = None
    regression_threshold_pct: float = 100.0
    warning_threshold_pct: float = 50.0
    settings: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("python", "subprocess"):
            raise ValueError("kind must be 'python' or 'subprocess'")
        if self.kind == "subprocess" and self.metric_type != "time":
            raise ValueError("Subprocess benchmarks only support metric_type 'time'")
        if not all(isinstance(tag, str) and tag.strip() for tag in self.tags):
            raise ValueError("tags must contain only non-empty strings")
        if not isinstance(self.parameters, dict) or not all(
            isinstance(name, str) and name.strip() for name in self.parameters
        ):
            raise ValueError("parameters must use non-empty string names")
        validate_case_configuration(
            case_name=self.case_name,
            benchmark_name=self.benchmark_name,
            case_type=self.case_type,
            metric_type=self.metric_type,
            gc_control=self.settings.gc_control,
            comparison_metric=self.comparison_metric,
            regression_threshold_pct=self.regression_threshold_pct,
            warning_threshold_pct=self.warning_threshold_pct,
        )


@dataclass
class SystemDefinition:
    name: str
    func: Callable[..., Any]
    scope: SystemScope = "function"
    is_async: bool = False

    def __post_init__(self) -> None:
        normalize_system_scope(self.scope)


@dataclass
class CommandSpec:
    """Describe one command invocation for a subprocess benchmark.

    Attributes:
        argv: Executable and arguments. Commands never pass through a shell.
        cwd: Optional child working directory.
        env: Environment values merged into the child environment.
        timeout_s: Optional hard timeout in seconds.
        expected_exit_codes: Exit codes considered successful.
        stdin_bytes: Optional fixed standard input.
        stdout: Inherit standard output or redirect it to the null device.
        stderr: Inherit standard error or redirect it to the null device.
    """

    argv: tuple[str, ...] | list[str]
    cwd: str | Path | None = None
    env: dict[str, str] | None = None
    timeout_s: float | None = None
    expected_exit_codes: tuple[int, ...] | list[int] = (0,)
    stdin_bytes: bytes | None = None
    stdout: CommandStream = "devnull"
    stderr: CommandStream = "devnull"

    def __post_init__(self) -> None:
        if not isinstance(self.argv, (list, tuple)) or not self.argv:
            raise ValueError("CommandSpec.argv must be a non-empty list or tuple")
        if not all(isinstance(arg, str) for arg in self.argv):
            raise ValueError("CommandSpec.argv must contain only strings")
        self.argv = tuple(self.argv)
        if self.cwd is not None and not isinstance(self.cwd, (str, Path)):
            raise ValueError("CommandSpec.cwd must be a str, Path, or None")
        if self.env is not None and (
            not isinstance(self.env, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in self.env.items()
            )
        ):
            raise ValueError("CommandSpec.env must contain only string keys and values")
        _validate_positive_number_or_none("CommandSpec.timeout_s", self.timeout_s)
        if not isinstance(self.expected_exit_codes, (list, tuple)):
            raise ValueError(
                "CommandSpec.expected_exit_codes must be a non-empty list or tuple of ints"
            )
        if not self.expected_exit_codes:
            raise ValueError("CommandSpec.expected_exit_codes cannot be empty")
        if not all(isinstance(code, int) for code in self.expected_exit_codes):
            raise ValueError("CommandSpec.expected_exit_codes must contain only ints")
        self.expected_exit_codes = tuple(self.expected_exit_codes)
        if self.stdin_bytes is not None and not isinstance(self.stdin_bytes, bytes):
            raise ValueError("CommandSpec.stdin_bytes must be bytes or None")
        if self.stdout not in ("devnull", "inherit"):
            raise ValueError("CommandSpec.stdout must be 'devnull' or 'inherit'")
        if self.stderr not in ("devnull", "inherit"):
            raise ValueError("CommandSpec.stderr must be 'devnull' or 'inherit'")


@dataclass
class BenchmarkResult:
    """Measurements and effective metadata for one registered benchmark."""

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
    parameters: dict[str, str] = field(default_factory=dict)
    parameter_id: str | None = None
    is_noisy: bool = False
    outlier_count: int = 0
    profile_path: str | None = None
    duration_s: float = 0.0


@dataclass
class BenchmarkRun:
    """A complete schema-versioned collection of benchmark results."""

    started_at: str
    finished_at: str
    python_version: str
    platform: str
    environment: dict[str, str]
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""


@dataclass
class Regression:
    """Classification produced by comparing one result with its baseline."""

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
    status: str = "stable"
    confidence_pct: float = 100.0


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._cases: list[BenchmarkCase] = []

    def register(self, case: BenchmarkCase) -> BenchmarkCase:
        key = (case.case_name, case.benchmark_name, case.metric_type)
        if any(
            (item.case_name, item.benchmark_name, item.metric_type) == key
            for item in self._cases
        ):
            raise ValueError(
                f"Benchmark '{case.case_name}::{case.benchmark_name}' is already registered"
            )
        self._cases.append(case)
        return case

    def all_cases(self) -> list[BenchmarkCase]:
        return list(self._cases)

    def clear(self) -> None:
        self._cases.clear()


class SystemRegistry:
    def __init__(self) -> None:
        self._systems: dict[str, SystemDefinition] = {}

    def register(self, system: SystemDefinition) -> SystemDefinition:
        if system.name in self._systems:
            raise ValueError(f"System '{system.name}' is already registered")
        self._systems[system.name] = system
        return system

    def all(self) -> dict[str, SystemDefinition]:
        return dict(self._systems)

    def clear(self) -> None:
        self._systems.clear()
