from benchbro.api import Case, clear_registry, get_registry, list_cases, system
from benchbro.core import (
    SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSettings,
    CommandSpec,
    GcControl,
    IsolationMode,
    MetricType,
    Regression,
    SystemDefinition,
    SystemRegistry,
    SystemScope,
    TimingMode,
    compare_runs,
    comparison_environment_mismatches,
    filter_cases,
    read_json,
    run_cases,
    write_csv,
    write_json,
    write_markdown,
)
from benchbro.profiling import ProfilerHook, register_profiler


def main(argv: list[str] | None = None) -> int:
    from benchbro.cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkSettings",
    "Case",
    "CommandSpec",
    "GcControl",
    "IsolationMode",
    "MetricType",
    "ProfilerHook",
    "Regression",
    "SystemDefinition",
    "SystemRegistry",
    "SystemScope",
    "TimingMode",
    "clear_registry",
    "compare_runs",
    "comparison_environment_mismatches",
    "filter_cases",
    "get_registry",
    "list_cases",
    "main",
    "read_json",
    "register_profiler",
    "run_cases",
    "system",
    "write_csv",
    "write_json",
    "write_markdown",
]
