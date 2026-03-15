from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from benchbro.core import (
    BenchmarkCase,
    BenchmarkRegistry,
    BenchmarkSettings,
    GcControl,
    MetricType,
    SystemDefinition,
    SystemRegistry,
    SystemScope,
    is_valid_comparison_metric,
)

_registry = BenchmarkRegistry()
_system_registry = SystemRegistry()
F = TypeVar("F", bound=Callable[..., Any])


def _validate_callable_signature(
    func: Callable[..., Any],
    *,
    label: str,
    allow_dependencies: bool,
) -> None:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    if not allow_dependencies and parameters:
        raise ValueError(f"{label} cannot declare dependencies")
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"{label} cannot use positional-only parameters")
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(f"{label} cannot use *args")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(f"{label} cannot use **kwargs")


@dataclass
class Case:
    name: str
    case_type: str = "cpu"
    metric_type: MetricType = "time"
    tags: list[str] = field(default_factory=list)
    warmup_iterations: int = 5
    min_iterations: int = 50
    repeats: int = 5
    gc_control: GcControl = "disable_during_measure"
    comparison_metric: str | None = None
    regression_threshold_pct: float = 100.0
    warning_threshold_pct: float = 50.0
    _input_func: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    def input(self) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            _validate_callable_signature(
                func,
                label=f"Input '{getattr(func, '__name__', 'input')}'",
                allow_dependencies=False,
            )
            self._input_func = func
            return func

        return decorator

    def benchmark(
        self,
        name: str | None = None,
        comparison_metric: str | None = None,
        regression_threshold_pct: float | None = None,
        warning_threshold_pct: float | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            _validate_callable_signature(
                func,
                label=f"Benchmark '{name or getattr(func, '__name__', 'benchmark')}'",
                allow_dependencies=True,
            )
            benchmark_name = name or getattr(func, "__name__", "benchmark")
            effective_comparison_metric = (
                self.comparison_metric
                if comparison_metric is None
                else comparison_metric
            )
            if (
                effective_comparison_metric is not None
                and not is_valid_comparison_metric(
                    self.metric_type, effective_comparison_metric
                )
            ):
                raise ValueError(
                    f"Invalid comparison metric '{effective_comparison_metric}' for metric_type '{self.metric_type}'"
                )
            effective_threshold = (
                self.regression_threshold_pct
                if regression_threshold_pct is None
                else regression_threshold_pct
            )
            effective_warning = (
                self.warning_threshold_pct
                if warning_threshold_pct is None
                else warning_threshold_pct
            )
            _registry.register(
                BenchmarkCase(
                    case_name=self.name,
                    benchmark_name=benchmark_name,
                    func=func,
                    case_type=self.case_type,
                    metric_type=self.metric_type,
                    tags=tuple(self.tags),
                    input_func=self._input_func,
                    input_name=(
                        getattr(self._input_func, "__name__", None)
                        if self._input_func is not None
                        else None
                    ),
                    comparison_metric=effective_comparison_metric,
                    regression_threshold_pct=effective_threshold,
                    warning_threshold_pct=effective_warning,
                    settings=BenchmarkSettings(
                        warmup_iterations=self.warmup_iterations,
                        min_iterations=self.min_iterations,
                        repeats=self.repeats,
                        gc_control=self.gc_control,
                    ),
                )
            )
            return func

        return decorator

    def subprocess(
        self,
        name: str | None = None,
        comparison_metric: str | None = None,
        regression_threshold_pct: float | None = None,
        warning_threshold_pct: float | None = None,
    ) -> Callable[[F], F]:
        if self.metric_type != "time":
            raise ValueError("Subprocess benchmarks only support metric_type 'time'")

        def decorator(func: F) -> F:
            benchmark_name = name or getattr(func, "__name__", "subprocess")
            _validate_callable_signature(
                func,
                label=f"Subprocess benchmark '{benchmark_name}'",
                allow_dependencies=True,
            )
            if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
                raise ValueError(
                    f"Subprocess benchmark '{benchmark_name}' cannot be async"
                )

            effective_comparison_metric = (
                self.comparison_metric
                if comparison_metric is None
                else comparison_metric
            )
            if (
                effective_comparison_metric is not None
                and not is_valid_comparison_metric(
                    self.metric_type, effective_comparison_metric
                )
            ):
                raise ValueError(
                    f"Invalid comparison metric '{effective_comparison_metric}' for metric_type '{self.metric_type}'"
                )
            effective_threshold = (
                self.regression_threshold_pct
                if regression_threshold_pct is None
                else regression_threshold_pct
            )
            effective_warning = (
                self.warning_threshold_pct
                if warning_threshold_pct is None
                else warning_threshold_pct
            )
            _registry.register(
                BenchmarkCase(
                    case_name=self.name,
                    benchmark_name=benchmark_name,
                    func=func,
                    kind="subprocess",
                    case_type=self.case_type,
                    metric_type=self.metric_type,
                    tags=tuple(self.tags),
                    input_func=self._input_func,
                    input_name=(
                        getattr(self._input_func, "__name__", None)
                        if self._input_func is not None
                        else None
                    ),
                    comparison_metric=effective_comparison_metric,
                    regression_threshold_pct=effective_threshold,
                    warning_threshold_pct=effective_warning,
                    settings=BenchmarkSettings(
                        warmup_iterations=self.warmup_iterations,
                        min_iterations=self.min_iterations,
                        repeats=self.repeats,
                        gc_control=self.gc_control,
                    ),
                )
            )
            return func

        return decorator


def system(
    *,
    scope: SystemScope = "function",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if scope not in ("function", "module"):
        raise ValueError(f"Invalid system scope '{scope}'")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        system_name = getattr(func, "__name__", "system")
        _validate_callable_signature(
            func,
            label=f"System '{system_name}'",
            allow_dependencies=True,
        )
        _system_registry.register(
            SystemDefinition(
                name=system_name,
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
    return _registry.all_cases()


def clear_registry() -> None:
    _registry.clear()
    _system_registry.clear()
