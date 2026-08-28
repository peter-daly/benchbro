from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchbro.models import IsolationMode

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml

DEFAULT_FILE_PATTERNS = (
    "bench_*.py",
    "*_bench.py",
    "*benchmark.py",
    "*benchmarks.py",
)


@dataclass(frozen=True)
class BenchbroConfig:
    benchmark_paths: tuple[str, ...] = ("benchmarks",)
    file_patterns: tuple[str, ...] = DEFAULT_FILE_PATTERNS
    repeats: int | None = None
    warmup: int | None = None
    min_iterations: int | None = None
    adaptive: bool | None = None
    min_repeats: int | None = None
    min_time_s: float | None = None
    max_time_s: float | None = None
    target_relative_margin_pct: float | None = None
    noise_threshold_pct: float | None = None
    isolation: IsolationMode | None = None
    cpu_affinity: tuple[int, ...] | None = None
    stabilization_delay_s: float | None = None
    baseline: str = "local"
    save_history: bool = False
    output_json: str | None = None
    output_csv: str | None = None
    output_md: str | None = None


def _string_tuple(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = (
        (value,)
        if isinstance(value, str)
        else tuple(value)
        if isinstance(value, list)
        else ()
    )
    if not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{name} must be a string or non-empty list of strings")
    return values


def _positive_int_or_none(
    value: Any, name: str, *, allow_zero: bool = False
) -> int | None:
    if value is None:
        return None
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        label = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {label} integer")
    return value


def _positive_float_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _non_negative_float_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _read_toml(path: Path) -> dict[str, Any]:
    return _toml.loads(path.read_text(encoding="utf-8"))


def load_config(repo_root: Path, explicit_path: str | None = None) -> BenchbroConfig:
    pyproject_path = repo_root / "pyproject.toml"
    section: dict[str, Any] = {}
    if pyproject_path.is_file():
        payload = _read_toml(pyproject_path)
        tool_section = payload.get("tool", {}).get("benchbro", {})
        if isinstance(tool_section, dict):
            legacy = tool_section.get("ini_options", {})
            if isinstance(legacy, dict):
                section.update(legacy)
            section.update(
                {
                    key: value
                    for key, value in tool_section.items()
                    if key != "ini_options"
                }
            )

    standalone_path = (
        Path(explicit_path) if explicit_path else repo_root / "benchbro.toml"
    )
    if standalone_path.is_file():
        standalone = _read_toml(standalone_path)
        standalone_section = standalone.get("benchbro", standalone)
        if not isinstance(standalone_section, dict):
            raise ValueError(f"Invalid BenchBro configuration in {standalone_path}")
        section.update(standalone_section)
    elif explicit_path:
        raise ValueError(f"Configuration file does not exist: {standalone_path}")

    run = section.get("run", {})
    output = section.get("output", {})
    if not isinstance(run, dict) or not isinstance(output, dict):
        raise ValueError("BenchBro run/output configuration must be TOML tables")
    paths = _string_tuple(section.get("benchmark_paths"), "benchmark_paths")
    patterns = _string_tuple(
        section.get("file_pattern", section.get("file_patterns")), "file_pattern"
    )
    affinity_raw = run.get("cpu_affinity")
    affinity = None
    if affinity_raw is not None:
        if (
            not isinstance(affinity_raw, list)
            or not affinity_raw
            or not all(
                isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0
                for cpu in affinity_raw
            )
        ):
            raise ValueError("run.cpu_affinity must be a list of non-negative integers")
        affinity = tuple(affinity_raw)
    adaptive = run.get("adaptive")
    if adaptive is not None and not isinstance(adaptive, bool):
        raise ValueError("run.adaptive must be a bool")
    isolation = run.get("isolation")
    if isolation not in (None, "in_process", "process"):
        raise ValueError("run.isolation must be 'in_process' or 'process'")
    baseline = section.get("baseline", "local")
    if not isinstance(baseline, str) or not baseline.strip():
        raise ValueError("baseline must be a non-empty string")
    save_history = section.get("save_history", False)
    if not isinstance(save_history, bool):
        raise ValueError("save_history must be a bool")
    return BenchbroConfig(
        benchmark_paths=paths or ("benchmarks",),
        file_patterns=patterns or DEFAULT_FILE_PATTERNS,
        repeats=_positive_int_or_none(run.get("repeats"), "run.repeats"),
        warmup=_positive_int_or_none(run.get("warmup"), "run.warmup", allow_zero=True),
        min_iterations=_positive_int_or_none(
            run.get("min_iterations"), "run.min_iterations"
        ),
        adaptive=adaptive,
        min_repeats=_positive_int_or_none(run.get("min_repeats"), "run.min_repeats"),
        min_time_s=_non_negative_float_or_none(run.get("min_time_s"), "run.min_time_s"),
        max_time_s=_positive_float_or_none(run.get("max_time_s"), "run.max_time_s"),
        target_relative_margin_pct=_non_negative_float_or_none(
            run.get("target_relative_margin_pct"),
            "run.target_relative_margin_pct",
        ),
        noise_threshold_pct=_non_negative_float_or_none(
            run.get("noise_threshold_pct"), "run.noise_threshold_pct"
        ),
        isolation=cast(IsolationMode | None, isolation),
        cpu_affinity=affinity,
        stabilization_delay_s=_non_negative_float_or_none(
            run.get("stabilization_delay_s"), "run.stabilization_delay_s"
        ),
        baseline=baseline,
        save_history=save_history,
        output_json=output.get("json"),
        output_csv=output.get("csv"),
        output_md=output.get("markdown"),
    )
