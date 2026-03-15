from __future__ import annotations

import asyncio
import csv
import gc
import inspect
import json
import os
import platform
import statistics
import subprocess
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Literal

MetricType = Literal["time", "memory"]
GcControl = Literal["inherit", "disable_during_measure"]
SystemScope = Literal["function", "module"]
BenchmarkKind = Literal["python", "subprocess"]
CommandStream = Literal["devnull", "inherit"]

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


@dataclass
class SystemDefinition:
    name: str
    func: Callable[..., Any]
    scope: SystemScope = "function"
    is_async: bool = False


@dataclass
class CommandSpec:
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
        normalized_argv: list[str] = []
        for arg in self.argv:
            if not isinstance(arg, str):
                raise ValueError("CommandSpec.argv must contain only strings")
            normalized_argv.append(arg)
        self.argv = tuple(normalized_argv)

        if self.cwd is not None and not isinstance(self.cwd, (str, Path)):
            raise ValueError("CommandSpec.cwd must be a str, Path, or None")

        if self.env is not None:
            if not isinstance(self.env, dict):
                raise ValueError("CommandSpec.env must be a dict[str, str] or None")
            for key, value in self.env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(
                        "CommandSpec.env must contain only string keys and values"
                    )

        if self.timeout_s is not None:
            if not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
                raise ValueError(
                    "CommandSpec.timeout_s must be a positive number or None"
                )

        if not isinstance(self.expected_exit_codes, (list, tuple)):
            raise ValueError(
                "CommandSpec.expected_exit_codes must be a non-empty list or tuple of ints"
            )
        normalized_codes: list[int] = []
        for code in self.expected_exit_codes:
            if not isinstance(code, int):
                raise ValueError(
                    "CommandSpec.expected_exit_codes must contain only ints"
                )
            normalized_codes.append(code)
        if not normalized_codes:
            raise ValueError("CommandSpec.expected_exit_codes cannot be empty")
        self.expected_exit_codes = tuple(normalized_codes)

        if self.stdin_bytes is not None and not isinstance(self.stdin_bytes, bytes):
            raise ValueError("CommandSpec.stdin_bytes must be bytes or None")

        if self.stdout not in ("devnull", "inherit"):
            raise ValueError("CommandSpec.stdout must be 'devnull' or 'inherit'")
        if self.stderr not in ("devnull", "inherit"):
            raise ValueError("CommandSpec.stderr must be 'devnull' or 'inherit'")


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


@dataclass
class _ScopeState:
    cache: dict[str, Any] = field(default_factory=dict)
    finalizers: list[Callable[[], Any]] = field(default_factory=list)


@dataclass
class _InvocationContext:
    case: BenchmarkCase
    systems: dict[str, SystemDefinition]
    module_state: _ScopeState
    function_state: _ScopeState
    active_stack: list[str] = field(default_factory=list)


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


def _validate_injected_callable(
    func: Callable[..., Any],
    *,
    label: str,
    allow_empty: bool,
) -> inspect.Signature:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    if not allow_empty and parameters:
        raise ValueError(f"{label} cannot declare dependencies")
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"{label} cannot use positional-only parameters")
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(f"{label} cannot use *args")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(f"{label} cannot use **kwargs")
    return signature


def _resolve_input(case: BenchmarkCase) -> Any:
    if case.input_func is None:
        raise ValueError(f"Benchmark '{case.benchmark_name}' does not define an input")
    return _resolve_maybe_awaitable(case.input_func())


def _consume_generator_teardown(generator: Any, system_name: str) -> None:
    try:
        next(generator)
    except StopIteration:
        return
    raise RuntimeError(f"System '{system_name}' yielded more than once")


async def _consume_async_generator_teardown(generator: Any, system_name: str) -> None:
    try:
        await anext(generator)
    except StopAsyncIteration:
        return
    raise RuntimeError(f"System '{system_name}' yielded more than once")


def _resolve_system_result(
    system_name: str,
    result: Any,
    scope_state: _ScopeState,
) -> Any:
    if inspect.isasyncgen(result):
        try:
            value = _run_awaitable(anext(result))
        except StopAsyncIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"System '{system_name}' must yield exactly one value"
            ) from exc
        scope_state.finalizers.append(
            lambda result=result, system_name=system_name: (
                _consume_async_generator_teardown(result, system_name)
            )
        )
        return value
    if inspect.isgenerator(result):
        try:
            value = next(result)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"System '{system_name}' must yield exactly one value"
            ) from exc
        scope_state.finalizers.append(
            lambda result=result, system_name=system_name: _consume_generator_teardown(
                result, system_name
            )
        )
        return value
    return _resolve_maybe_awaitable(result)


def _stdio_target(mode: CommandStream) -> Any:
    if mode == "devnull":
        return subprocess.DEVNULL
    return None


def _resolve_callable_kwargs(
    func: Callable[..., Any],
    context: _InvocationContext,
    *,
    owner_name: str,
    owner_kind: str,
    owner_scope: SystemScope | None = None,
    owner_is_async: bool | None = None,
) -> dict[str, Any]:
    signature = _validate_injected_callable(
        func,
        label=f"{owner_kind} '{owner_name}'",
        allow_empty=True,
    )
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if (
            context.case.input_name is not None
            and parameter.name == context.case.input_name
            and parameter.name in context.systems
        ):
            raise ValueError(
                f"Dependency name '{parameter.name}' is ambiguous between the case input and a system"
            )
        if (
            context.case.input_name is not None
            and parameter.name == context.case.input_name
        ):
            if owner_kind == "system":
                raise ValueError(
                    f"System '{owner_name}' cannot depend on input '{parameter.name}'"
                )
            kwargs[parameter.name] = _resolve_input(context.case)
            continue
        if parameter.name in context.systems:
            dependency = context.systems[parameter.name]
            if owner_scope == "module" and dependency.scope == "function":
                raise ValueError(
                    f"Module-scoped system '{owner_name}' cannot depend on function-scoped system '{parameter.name}'"
                )
            if owner_is_async is False and dependency.is_async:
                raise ValueError(
                    f"Sync {owner_kind} '{owner_name}' cannot depend on async system '{parameter.name}'"
                )
            kwargs[parameter.name] = _resolve_system(parameter.name, context)
            continue
        if parameter.default is not inspect.Signature.empty:
            continue
        raise ValueError(
            f"{owner_kind.capitalize()} '{owner_name}' has unknown dependency '{parameter.name}'"
        )
    return kwargs


def _resolve_system(system_name: str, context: _InvocationContext) -> Any:
    system = context.systems.get(system_name)
    if system is None:
        raise ValueError(f"Unknown system '{system_name}'")

    scope_state = (
        context.module_state if system.scope == "module" else context.function_state
    )
    if system_name in scope_state.cache:
        return scope_state.cache[system_name]

    if system_name in context.active_stack:
        cycle = " -> ".join([*context.active_stack, system_name])
        raise ValueError(f"System dependency cycle detected: {cycle}")

    context.active_stack.append(system_name)
    try:
        kwargs = _resolve_callable_kwargs(
            system.func,
            context,
            owner_name=system_name,
            owner_kind="system",
            owner_scope=system.scope,
            owner_is_async=system.is_async,
        )
        value = _resolve_system_result(system_name, system.func(**kwargs), scope_state)
        scope_state.cache[system_name] = value
        return value
    finally:
        context.active_stack.pop()


def _invoke_python_case(case: BenchmarkCase, context: _InvocationContext) -> Any:
    kwargs = _resolve_callable_kwargs(
        case.func,
        context,
        owner_name=case.benchmark_name,
        owner_kind="benchmark",
        owner_is_async=(
            inspect.iscoroutinefunction(case.func)
            or inspect.isasyncgenfunction(case.func)
        ),
    )
    return _resolve_maybe_awaitable(case.func(**kwargs))


def _build_command_spec(
    case: BenchmarkCase, context: _InvocationContext
) -> CommandSpec:
    kwargs = _resolve_callable_kwargs(
        case.func,
        context,
        owner_name=case.benchmark_name,
        owner_kind="subprocess benchmark",
        owner_is_async=False,
    )
    result = case.func(**kwargs)
    if inspect.isawaitable(result):
        raise ValueError(
            f"Subprocess benchmark '{case.benchmark_name}' cannot be async; return CommandSpec directly"
        )
    if not isinstance(result, CommandSpec):
        raise ValueError(
            f"Subprocess benchmark '{case.benchmark_name}' must return CommandSpec"
        )
    return result


def _invoke_command_case(case: BenchmarkCase, context: _InvocationContext) -> None:
    spec = _build_command_spec(case, context)
    env = None
    if spec.env is not None:
        env = os.environ.copy()
        env.update(spec.env)
    cwd = str(spec.cwd) if spec.cwd is not None else None
    try:
        completed = subprocess.run(
            spec.argv,
            cwd=cwd,
            env=env,
            input=spec.stdin_bytes,
            stdout=_stdio_target(spec.stdout),
            stderr=_stdio_target(spec.stderr),
            timeout=spec.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Subprocess benchmark '{case.benchmark_name}' timed out after {spec.timeout_s} seconds"
        ) from exc

    if completed.returncode not in spec.expected_exit_codes:
        raise RuntimeError(
            f"Subprocess benchmark '{case.benchmark_name}' exited with code {completed.returncode}"
        )


def _invoke_case(case: BenchmarkCase, context: _InvocationContext) -> Any:
    if case.kind == "subprocess":
        return _invoke_command_case(case, context)
    return _invoke_python_case(case, context)


def _time_metrics(
    invoke: Callable[[], Any],
    case: BenchmarkCase,
) -> tuple[dict[str, float], list[float]]:
    samples: list[float] = []
    for _ in range(case.settings.repeats):
        start = perf_counter_ns()
        for _ in range(case.settings.min_iterations):
            invoke()
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


def _memory_metrics(invoke: Callable[[], Any], case: BenchmarkCase) -> dict[str, float]:
    peaks: list[float] = []
    nets: list[float] = []
    for _ in range(case.settings.repeats):
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        for _ in range(case.settings.min_iterations):
            invoke()
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
        kind=original.kind,
        case_type=original.case_type,
        metric_type=original.metric_type,
        tags=original.tags,
        input_func=original.input_func,
        input_name=original.input_name,
        comparison_metric=original.comparison_metric,
        regression_threshold_pct=original.regression_threshold_pct,
        warning_threshold_pct=original.warning_threshold_pct,
        settings=settings,
    )


def _finalize_scope(state: _ScopeState) -> None:
    error: BaseException | None = None
    while state.finalizers:
        finalizer = state.finalizers.pop()
        try:
            _resolve_maybe_awaitable(finalizer())
        except BaseException as exc:  # pragma: no cover - defensive
            if error is None:
                error = exc
    state.cache.clear()
    if error is not None:
        raise error


def run_cases(
    cases: list[BenchmarkCase],
    repeats: int | None = None,
    warmup: int | None = None,
    min_iterations: int | None = None,
) -> BenchmarkRun:
    from benchbro.api import get_system_registry

    started = datetime.now(timezone.utc).isoformat()
    results: list[BenchmarkResult] = []
    environment = _collect_environment_metadata()
    module_state = _ScopeState()
    systems = get_system_registry().all()

    try:
        for original in cases:
            case = _with_overrides(
                original, repeats=repeats, warmup=warmup, min_iterations=min_iterations
            )
            if case.kind == "subprocess" and case.metric_type != "time":
                raise ValueError(
                    f"Subprocess benchmark '{case.benchmark_name}' only supports metric_type 'time'"
                )
            function_state = _ScopeState()
            context = _InvocationContext(
                case=case,
                systems=systems,
                module_state=module_state,
                function_state=function_state,
            )
            try:
                for _ in range(case.settings.warmup_iterations):
                    _invoke_case(case, context)

                with _gc_control_context(case.settings.gc_control):
                    if case.metric_type == "time":
                        metrics, time_samples_s = _time_metrics(
                            lambda: _invoke_case(case, context),
                            case,
                        )
                    else:
                        metrics = _memory_metrics(
                            lambda: _invoke_case(case, context),
                            case,
                        )
                        time_samples_s = []
            finally:
                _finalize_scope(function_state)

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
    finally:
        _finalize_scope(module_state)

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
