from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from benchbro.lifecycle import validate_injected_callable
from benchbro.models import (
    BenchmarkCase,
    BenchmarkKind,
    BenchmarkRegistry,
    BenchmarkSettings,
    GcControl,
    IsolationMode,
    MetricType,
    SystemDefinition,
    SystemRegistry,
    SystemScope,
    TimingMode,
    is_valid_comparison_metric,
    normalize_system_scope,
    validate_case_configuration,
)

_registry = BenchmarkRegistry()
_system_registry = SystemRegistry()
F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    values: tuple[Any, ...]
    ids: tuple[str, ...]


def _parameter_specs(func: Callable[..., Any]) -> list[_ParameterSpec]:
    return list(getattr(func, "__benchbro_parameter_specs__", []))


def _validate_metric(metric_type: MetricType, metric: str | None) -> None:
    if metric is not None and not is_valid_comparison_metric(metric_type, metric):
        raise ValueError(
            f"Invalid comparison metric '{metric}' for metric_type '{metric_type}'"
        )


@dataclass
class Case:
    """Configure a group of benchmarks with shared measurement settings.

    Attributes:
        name: Stable case name used in reports and baseline matching.
        case_type: Free-form workload category, such as ``"cpu"``.
        metric_type: Measure elapsed time or Python allocations.
        tags: Labels used by command-line filtering.
        warmup_iterations: Unmeasured invocations before sampling.
        min_iterations: Invocations combined into each repeat sample.
        repeats: Fixed repeat count or adaptive maximum.
        comparison_metric: Metric used for baseline comparison. The metric-type
            default is used when this is ``None``.
        adaptive: Stop sampling when the precision target is reached.
        isolation: Run in this process or a dedicated worker process.
    """

    name: str
    case_type: str = "cpu"
    metric_type: MetricType = "time"
    tags: list[str] = field(default_factory=list)
    warmup_iterations: int = 5
    min_iterations: int = 50
    repeats: int = 20
    gc_control: GcControl = "disable_during_measure"
    comparison_metric: str | None = None
    regression_threshold_pct: float = 100.0
    warning_threshold_pct: float = 50.0
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
    _input_func: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in self.tags
        ):
            raise ValueError("tags must be a list of non-empty strings")
        self._base_settings()
        validate_case_configuration(
            case_name=self.name,
            case_type=self.case_type,
            metric_type=self.metric_type,
            gc_control=self.gc_control,
            comparison_metric=self.comparison_metric,
            regression_threshold_pct=self.regression_threshold_pct,
            warning_threshold_pct=self.warning_threshold_pct,
        )

    def _base_settings(self) -> BenchmarkSettings:
        return BenchmarkSettings(
            warmup_iterations=self.warmup_iterations,
            min_iterations=self.min_iterations,
            repeats=self.repeats,
            gc_control=self.gc_control,
            adaptive=self.adaptive,
            min_repeats=min(self.min_repeats, self.repeats),
            min_time_s=self.min_time_s,
            max_time_s=self.max_time_s,
            target_relative_margin_pct=self.target_relative_margin_pct,
            noise_threshold_pct=self.noise_threshold_pct,
            setup_timing=self.setup_timing,
            teardown_timing=self.teardown_timing,
            timeout_s=self.timeout_s,
            isolation=self.isolation,
            cpu_affinity=self.cpu_affinity,
            stabilization_delay_s=self.stabilization_delay_s,
            profiler=self.profiler,
            profile_output=self.profile_output,
        )

    def input(self) -> Callable[[F], F]:
        """Register the callable that supplies input for this case's benchmarks.

        Returns:
            A decorator that preserves the decorated callable.
        """

        def decorator(func: F) -> F:
            validate_injected_callable(
                func,
                label=f"Input '{getattr(func, '__name__', 'input')}'",
                allow_empty=False,
            )
            self._input_func = func
            return func

        return decorator

    def parametrize(
        self,
        name: str,
        values: Iterable[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> Callable[[F], F]:
        """Add named parameter values to a benchmark.

        Multiple parameter decorators form a Cartesian product. The generated
        identifier is recorded in result artifacts and benchmark names.

        Args:
            name: Function parameter to inject.
            values: Values to register.
            ids: Optional stable display identifier for each value.

        Returns:
            A decorator that records the parameter specification.
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Parameter name must be a non-empty string")
        collected = tuple(values)
        if not collected:
            raise ValueError("Parameter values cannot be empty")
        normalized_ids = (
            tuple(str(value) for value in collected) if ids is None else tuple(ids)
        )
        if len(normalized_ids) != len(collected) or not all(
            isinstance(item, str) and item.strip() for item in normalized_ids
        ):
            raise ValueError(
                "Parameter ids must contain one non-empty string per value"
            )

        def decorator(func: F) -> F:
            specs = _parameter_specs(func)
            if any(spec.name == name for spec in specs):
                raise ValueError(f"Parameter '{name}' is already defined")
            specs.append(_ParameterSpec(name, collected, normalized_ids))
            func.__dict__["__benchbro_parameter_specs__"] = specs
            return func

        return decorator

    def _effective_settings(self, **overrides: Any) -> BenchmarkSettings:
        values = {key: value for key, value in overrides.items() if value is not None}
        base = self._base_settings()
        effective_repeats = values.get("repeats", base.repeats)
        if base.min_repeats > effective_repeats:
            values["min_repeats"] = effective_repeats
        return replace(base, **values)

    def _register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None,
        kind: BenchmarkKind,
        comparison_metric: str | None,
        regression_threshold_pct: float | None,
        warning_threshold_pct: float | None,
        settings_overrides: dict[str, Any],
    ) -> None:
        benchmark_name = name or getattr(func, "__name__", "benchmark")
        metric = (
            self.comparison_metric if comparison_metric is None else comparison_metric
        )
        _validate_metric(self.metric_type, metric)
        threshold = (
            self.regression_threshold_pct
            if regression_threshold_pct is None
            else regression_threshold_pct
        )
        warning = (
            self.warning_threshold_pct
            if warning_threshold_pct is None
            else warning_threshold_pct
        )
        specs = _parameter_specs(func)
        combinations = (
            itertools.product(*(range(len(spec.values)) for spec in specs))
            if specs
            else [()]
        )
        for indexes in combinations:
            parameters = {
                spec.name: spec.values[index] for spec, index in zip(specs, indexes)
            }
            parameter_id = (
                "-".join(spec.ids[index] for spec, index in zip(specs, indexes)) or None
            )
            qualified_name = (
                f"{benchmark_name}[{parameter_id}]" if parameter_id else benchmark_name
            )
            _registry.register(
                BenchmarkCase(
                    case_name=self.name,
                    benchmark_name=qualified_name,
                    func=func,
                    kind=kind,
                    case_type=self.case_type,
                    metric_type=self.metric_type,
                    tags=tuple(self.tags),
                    input_func=self._input_func,
                    input_name=(
                        getattr(self._input_func, "__name__", None)
                        if self._input_func is not None
                        else None
                    ),
                    comparison_metric=metric,
                    regression_threshold_pct=threshold,
                    warning_threshold_pct=warning,
                    settings=self._effective_settings(**settings_overrides),
                    parameters=parameters,
                    parameter_id=parameter_id,
                )
            )

    def benchmark(
        self,
        name: str | None = None,
        comparison_metric: str | None = None,
        regression_threshold_pct: float | None = None,
        warning_threshold_pct: float | None = None,
        *,
        warmup_iterations: int | None = None,
        min_iterations: int | None = None,
        repeats: int | None = None,
        adaptive: bool | None = None,
        timeout_s: float | None = None,
        isolation: IsolationMode | None = None,
        setup_timing: TimingMode | None = None,
        teardown_timing: TimingMode | None = None,
        profiler: str | None = None,
        profile_output: str | None = None,
    ) -> Callable[[F], F]:
        """Register a Python callable as a benchmark.

        Optional arguments override the corresponding case settings for this
        callable only. Synchronous and asynchronous callables are supported.

        Returns:
            A decorator that registers and preserves the callable.
        """

        def decorator(func: F) -> F:
            validate_injected_callable(
                func,
                label=f"Benchmark '{name or getattr(func, '__name__', 'benchmark')}'",
                allow_empty=True,
            )
            self._register(
                func,
                name=name,
                kind="python",
                comparison_metric=comparison_metric,
                regression_threshold_pct=regression_threshold_pct,
                warning_threshold_pct=warning_threshold_pct,
                settings_overrides={
                    "warmup_iterations": warmup_iterations,
                    "min_iterations": min_iterations,
                    "repeats": repeats,
                    "adaptive": adaptive,
                    "timeout_s": timeout_s,
                    "isolation": isolation,
                    "setup_timing": setup_timing,
                    "teardown_timing": teardown_timing,
                    "profiler": profiler,
                    "profile_output": profile_output,
                },
            )
            return func

        return decorator

    def subprocess(
        self,
        name: str | None = None,
        comparison_metric: str | None = None,
        regression_threshold_pct: float | None = None,
        warning_threshold_pct: float | None = None,
        *,
        warmup_iterations: int | None = None,
        min_iterations: int | None = None,
        repeats: int | None = None,
        adaptive: bool | None = None,
        timeout_s: float | None = None,
        isolation: IsolationMode | None = None,
        setup_timing: TimingMode | None = None,
        teardown_timing: TimingMode | None = None,
        profiler: str | None = None,
        profile_output: str | None = None,
    ) -> Callable[[F], F]:
        """Register a callable that returns a :class:`CommandSpec`.

        Subprocess benchmarks support time metrics only. The command callable
        must be synchronous and returns an explicit argument vector rather than a
        shell command string.

        Returns:
            A decorator that registers and preserves the callable.
        """

        if self.metric_type != "time":
            raise ValueError("Subprocess benchmarks only support metric_type 'time'")

        def decorator(func: F) -> F:
            benchmark_name = name or getattr(func, "__name__", "subprocess")
            validate_injected_callable(
                func,
                label=f"Subprocess benchmark '{benchmark_name}'",
                allow_empty=True,
            )
            if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
                raise ValueError(
                    f"Subprocess benchmark '{benchmark_name}' cannot be async"
                )
            self._register(
                func,
                name=name,
                kind="subprocess",
                comparison_metric=comparison_metric,
                regression_threshold_pct=regression_threshold_pct,
                warning_threshold_pct=warning_threshold_pct,
                settings_overrides={
                    "warmup_iterations": warmup_iterations,
                    "min_iterations": min_iterations,
                    "repeats": repeats,
                    "adaptive": adaptive,
                    "timeout_s": timeout_s,
                    "isolation": isolation,
                    "setup_timing": setup_timing,
                    "teardown_timing": teardown_timing,
                    "profiler": profiler,
                    "profile_output": profile_output,
                },
            )
            return func

        return decorator


def system(*, scope: SystemScope = "function") -> Callable[[F], F]:
    """Register an injectable dependency with a managed lifecycle.

    Systems can be synchronous, asynchronous, generators, or async generators,
    and can depend on other systems by parameter name.

    Args:
        scope: ``"iteration"``, ``"benchmark"``, or ``"session"``. The legacy
            aliases ``"function"`` and ``"module"`` are also accepted.

    Returns:
        A decorator that registers and preserves the callable.
    """

    normalize_system_scope(scope)

    def decorator(func: F) -> F:
        name = getattr(func, "__name__", "system")
        validate_injected_callable(func, label=f"System '{name}'", allow_empty=True)
        _system_registry.register(
            SystemDefinition(
                name=name,
                func=func,
                scope=scope,
                is_async=(
                    inspect.iscoroutinefunction(func)
                    or inspect.isasyncgenfunction(func)
                ),
            )
        )
        return func

    return decorator


def get_registry() -> BenchmarkRegistry:
    return _registry


def get_system_registry() -> SystemRegistry:
    return _system_registry


def list_cases() -> list[BenchmarkCase]:
    """Return all benchmark cases currently registered by decorators."""

    return _registry.all_cases()


def clear_registry() -> None:
    """Remove all registered benchmarks and systems."""

    _registry.clear()
    _system_registry.clear()
