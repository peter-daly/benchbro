"""Backward-compatible public facade for BenchBro's focused modules."""

from __future__ import annotations

from benchbro.comparison import compare_runs, comparison_environment_mismatches
from benchbro.environment import collect_environment_metadata
from benchbro.models import (
    MEMORY_COMPARISON_METRICS,
    SCHEMA_VERSION,
    TIME_COMPARISON_METRICS,
    BenchmarkCase,
    BenchmarkKind,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSettings,
    CommandSpec,
    CommandStream,
    GcControl,
    IsolationMode,
    MetricType,
    Regression,
    SystemDefinition,
    SystemRegistry,
    SystemScope,
    TimingMode,
    default_comparison_metric,
    is_valid_comparison_metric,
    validate_case_configuration,
)
from benchbro.runner import run_cases
from benchbro.serialization import (
    migrate_payload,
    read_json,
    write_csv,
    write_json,
    write_markdown,
)

_validate_case_configuration = validate_case_configuration
_collect_environment_metadata = collect_environment_metadata


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


__all__ = [
    "MEMORY_COMPARISON_METRICS",
    "SCHEMA_VERSION",
    "TIME_COMPARISON_METRICS",
    "BenchmarkCase",
    "BenchmarkKind",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkSettings",
    "CommandSpec",
    "CommandStream",
    "GcControl",
    "IsolationMode",
    "MetricType",
    "Regression",
    "SystemDefinition",
    "SystemRegistry",
    "SystemScope",
    "TimingMode",
    "compare_runs",
    "comparison_environment_mismatches",
    "default_comparison_metric",
    "filter_cases",
    "is_valid_comparison_metric",
    "migrate_payload",
    "read_json",
    "run_cases",
    "write_csv",
    "write_json",
    "write_markdown",
]
