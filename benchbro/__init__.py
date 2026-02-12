from benchbro.api import Case, clear_registry, get_registry, list_cases
from benchbro.core import (
    BenchmarkCase,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSettings,
    GcControl,
    MetricType,
    Regression,
    compare_runs,
    filter_cases,
    read_json,
    run_cases,
    write_csv,
    write_json,
    write_markdown,
)


def main(argv: list[str] | None = None) -> int:
    from benchbro.cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "BenchmarkCase",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkSettings",
    "Case",
    "GcControl",
    "MetricType",
    "Regression",
    "clear_registry",
    "compare_runs",
    "filter_cases",
    "get_registry",
    "list_cases",
    "main",
    "read_json",
    "run_cases",
    "write_markdown",
    "write_csv",
    "write_json",
]
