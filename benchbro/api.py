from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from benchbro.core import BenchmarkCase, BenchmarkRegistry, BenchmarkSettings, GcControl, MetricType

_registry = BenchmarkRegistry()


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
    regression_threshold_pct: float = 100.0
    warning_threshold_pct: float = 50.0
    _input_func: Callable[[], Any] | None = field(default=None, init=False, repr=False)

    def input(self) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
        def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
            self._input_func = func
            return func

        return decorator

    def benchmark(
        self,
        name: str | None = None,
        regression_threshold_pct: float | None = None,
        warning_threshold_pct: float | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            benchmark_name = name or getattr(func, "__name__", "benchmark")
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


def get_registry() -> BenchmarkRegistry:
    return _registry


def list_cases() -> list[BenchmarkCase]:
    return _registry.all_cases()


def clear_registry() -> None:
    _registry.clear()
