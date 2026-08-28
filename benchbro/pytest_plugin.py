from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from benchbro.models import BenchmarkCase, BenchmarkResult, BenchmarkSettings
from benchbro.runner import run_cases


class PytestBenchmarkRunner:
    """Measure a callable while letting pytest provide its arguments and fixtures."""

    def __call__(
        self,
        func: Callable[..., Any],
        *args: Any,
        name: str | None = None,
        repeats: int = 20,
        warmup: int = 5,
        min_iterations: int = 50,
        adaptive: bool = False,
        max_time_s: float | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        def invoke() -> Any:
            return func(*args, **kwargs)

        case = BenchmarkCase(
            case_name="pytest",
            benchmark_name=name or getattr(func, "__name__", "benchmark"),
            func=invoke,
            settings=BenchmarkSettings(
                repeats=repeats,
                warmup_iterations=warmup,
                min_iterations=min_iterations,
                adaptive=adaptive,
                min_repeats=min(5, repeats),
                max_time_s=max_time_s,
            ),
        )
        return run_cases([case]).benchmarks[0]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchbro: marks a test that performs a BenchBro measurement",
    )


@pytest.fixture
def benchbro_runner() -> PytestBenchmarkRunner:
    return PytestBenchmarkRunner()
