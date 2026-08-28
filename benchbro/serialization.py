from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchbro.models import SCHEMA_VERSION, BenchmarkResult, BenchmarkRun


def migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("schema_version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError("Invalid benchmark result schema_version")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark result schema_version {version}; "
            f"this BenchBro version supports up to {SCHEMA_VERSION}"
        )
    migrated = dict(payload)
    if version == 1:
        migrated["schema_version"] = 2
        migrated.setdefault("run_id", "")
        for item in migrated.get("benchmarks", []):
            item.setdefault("parameters", {})
            item.setdefault("parameter_id", None)
            item.setdefault("is_noisy", False)
            item.setdefault("outlier_count", 0)
            item.setdefault("profile_path", None)
            item.setdefault("duration_s", 0.0)
    return migrated


def write_json(path: str | Path, run: BenchmarkRun) -> None:
    """Write a complete schema-versioned run as JSON."""

    payload = asdict(run)
    payload["schema_version"] = SCHEMA_VERSION
    for benchmark_payload in payload.get("benchmarks", []):
        benchmark_payload.pop("time_samples_s", None)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: str | Path) -> BenchmarkRun:
    """Read a run, migrating supported older payloads to the current schema."""

    raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("Benchmark result must be a JSON object")
    payload = migrate_payload(raw_payload)
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
            parameters=item.get("parameters", {}),
            parameter_id=item.get("parameter_id"),
            is_noisy=item.get("is_noisy", False),
            outlier_count=item.get("outlier_count", 0),
            profile_path=item.get("profile_path"),
            duration_s=item.get("duration_s", 0.0),
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
        schema_version=payload["schema_version"],
        run_id=payload.get("run_id", ""),
    )


def write_csv(path: str | Path, run: BenchmarkRun) -> None:
    """Write one long-form CSV row per benchmark metric."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "case",
                "benchmark",
                "parameters",
                "metric_type",
                "gc_control",
                "noisy",
                "metric_name",
                "value",
            ]
        )
        for result in run.benchmarks:
            parameters = ", ".join(
                f"{key}={value}" for key, value in result.parameters.items()
            )
            for metric_name, value in result.metrics.items():
                writer.writerow(
                    [
                        result.case_name,
                        result.benchmark_name,
                        parameters,
                        result.metric_type,
                        result.gc_control,
                        result.is_noisy,
                        metric_name,
                        value,
                    ]
                )


def write_markdown(path: str | Path, run: BenchmarkRun) -> None:
    """Write a human-readable Markdown result summary."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchbro Results",
        "",
        f"- schema_version: `{run.schema_version}`",
        f"- run_id: `{run.run_id}`",
        f"- started_at: `{run.started_at}`",
        f"- finished_at: `{run.finished_at}`",
        f"- python_version: `{run.python_version}`",
        f"- platform: `{run.platform}`",
        "",
        "| case | benchmark | metric_type | parameters | mean_s | median_s | ci95 | cv_pct | p95_s | ops_per_sec | peak_alloc_bytes | net_alloc_bytes | status |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in run.benchmarks:
        metrics = result.metrics
        parameters = ", ".join(
            f"{key}={value}" for key, value in result.parameters.items()
        )
        ci95 = "-"
        if "ci95_low_s" in metrics and "ci95_high_s" in metrics:
            ci95 = (
                f"{_format_metric(metrics['ci95_low_s'])}–"
                f"{_format_metric(metrics['ci95_high_s'])}"
            )
        lines.append(
            "| "
            f"{_escape_cell(result.case_name)} | "
            f"{_escape_cell(result.benchmark_name)} | "
            f"{_escape_cell(result.metric_type)} | "
            f"{_escape_cell(parameters)} | "
            f"{_format_metric(metrics.get('mean_s'))} | "
            f"{_format_metric(metrics.get('median_s'))} | "
            f"{ci95} | "
            f"{_format_metric(metrics.get('cv_pct'))} | "
            f"{_format_metric(metrics.get('p95_s'))} | "
            f"{_format_metric(metrics.get('ops_per_sec'))} | "
            f"{_format_metric(metrics.get('peak_alloc_bytes'))} | "
            f"{_format_metric(metrics.get('net_alloc_bytes'))} | "
            f"{'noisy' if result.is_noisy else 'stable'} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
