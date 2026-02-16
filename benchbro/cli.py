from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import math
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchbro.api import get_registry
from benchbro.core import (
    compare_runs,
    filter_cases,
    read_json,
    run_cases,
    write_csv,
    write_json,
    write_markdown,
)

BENCH_FILE_PATTERNS = (
    "bench_*.py",
    "*_bench.py",
    "*benchmark.py",
    "*benchmarks.py",
)

_CONSOLE = Console()
LOCAL_BASELINE_JSON_PATH = Path(".benchbro/baseline.local.json")
CI_BASELINE_JSON_PATH = Path(".benchbro/baseline.ci.json")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _print_results(run) -> None:
    env_table = Table(title="Environment")
    env_table.add_column("Key", style="cyan", no_wrap=True)
    env_table.add_column("Value", style="bold")

    for key in sorted(run.environment):
        env_table.add_row(key, run.environment[key])
    _CONSOLE.print(env_table)

    by_type = defaultdict(list)
    for result in run.benchmarks:
        by_type[result.metric_type].append(result)

    for metric_type in sorted(by_type):
        table = Table(title=f"Benchbro Results: {metric_type}")
        table.add_column("Case", style="cyan", no_wrap=True)
        table.add_column("Benchmark", style="bold")
        if metric_type == "time":
            table.add_column("Mean (s)", justify="right")
            table.add_column("Median (s)", justify="right")
            table.add_column("IQR (s)", justify="right")
            table.add_column("P95 (s)", justify="right")
            table.add_column("StdDev (s)", justify="right")
            table.add_column("Ops/s", justify="right")
        else:
            table.add_column("Peak (B)", justify="right")
            table.add_column("Net (B)", justify="right")
            table.add_column("Peak Max (B)", justify="right")

        for result in by_type[metric_type]:
            metrics = result.metrics
            if metric_type == "time":
                table.add_row(
                    result.case_name,
                    result.benchmark_name,
                    _format_metric(metrics.get("mean_s")),
                    _format_metric(metrics.get("median_s")),
                    _format_metric(metrics.get("iqr_s")),
                    _format_metric(metrics.get("p95_s")),
                    _format_metric(metrics.get("stddev_s")),
                    _format_metric(metrics.get("ops_per_sec")),
                )
            else:
                table.add_row(
                    result.case_name,
                    result.benchmark_name,
                    _format_metric(metrics.get("peak_alloc_bytes")),
                    _format_metric(metrics.get("net_alloc_bytes")),
                    _format_metric(metrics.get("peak_alloc_bytes_max")),
                )
        _CONSOLE.print(table)


def _print_regressions(regressions) -> int:
    if not regressions:
        _CONSOLE.print("[yellow]No comparable benchmarks found.[/yellow]")
        return 0

    failing = [r for r in regressions if r.is_regression]
    table = Table(title="Benchmark Comparison")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Benchmark", style="bold")
    table.add_column("Metric", style="magenta")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Thresholds", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Status", justify="center")

    for item in regressions:
        if item.is_regression:
            status = "[red]REGRESSION[/red]"
            change_style = "red"
        elif item.is_warning:
            status = "[yellow]WARNING[/yellow]"
            change_style = "yellow"
        else:
            status = "[green]OK[/green]"
            change_style = "green"
        table.add_row(
            item.case_name,
            item.benchmark_name,
            item.metric_name,
            f"{item.baseline_value:.6g}",
            f"{item.current_value:.6g}",
            f"[yellow]{item.warning_threshold_pct:.2f}%[/yellow][white] / [/white][red]{item.threshold_pct:.2f}%[/red]",
            f"[{change_style}]{item.percent_change:+.2f}%[/{change_style}]",
            status,
        )
    _CONSOLE.print(table)
    return 2 if failing else 0


def _build_histogram(
    samples: list[float], bins: int = 10
) -> list[tuple[float, float, int]]:
    if not samples:
        return []

    if len(samples) == 1:
        value = samples[0]
        return [(value, value, 1)]

    low = min(samples)
    high = max(samples)
    if math.isclose(low, high):
        return [(low, high, len(samples))]

    width = (high - low) / bins
    counts = [0] * bins
    for sample in samples:
        if sample == high:
            index = bins - 1
        else:
            index = int((sample - low) / width)
            index = max(0, min(index, bins - 1))
        counts[index] += 1

    ranges: list[tuple[float, float, int]] = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        bin_low = low + (width * i)
        bin_high = low + (width * (i + 1))
        ranges.append((bin_low, bin_high, count))
    return ranges


def _print_histograms(run) -> None:
    time_results = [
        r for r in run.benchmarks if r.metric_type == "time" and r.time_samples_s
    ]
    if not time_results:
        _CONSOLE.print(
            "[yellow]No time samples available for histogram output.[/yellow]"
        )
        return

    for result in time_results:
        histogram = _build_histogram(result.time_samples_s, bins=10)
        if not histogram:
            continue
        max_count = max(count for _, _, count in histogram)
        table = Table(title=f"Histogram: {result.case_name}::{result.benchmark_name}")
        table.add_column("Range (s)", style="cyan")
        table.add_column("Distribution", style="bold")
        table.add_column("Count", justify="right")
        for low, high, count in histogram:
            bar_len = max(1, int((count / max_count) * 24))
            bar = "█" * bar_len
            table.add_row(f"{low:.6g} - {high:.6g}", bar, str(count))
        _CONSOLE.print(table)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchbro")
    parser.add_argument(
        "target",
        nargs="?",
        help="Benchmark target (module, .py file, or directory). Defaults to <repo>/benchmarks.",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--min-iterations", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-md")
    parser.add_argument("--new-baseline", action="store_true")
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print per-time-benchmark histograms in terminal output.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip baseline comparison; still backfill missing benchmark entries.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Use baseline.ci.json for baseline read/write/compare.",
    )
    return parser


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _baseline_output_path(use_ci: bool = False) -> Path:
    repo_root = _find_repo_root(Path.cwd())
    output_dir = repo_root / ".benchbro"
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_ci:
        return output_dir / CI_BASELINE_JSON_PATH.name
    return output_dir / LOCAL_BASELINE_JSON_PATH.name


def _matches_bench_pattern(path: Path, file_patterns: tuple[str, ...]) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in file_patterns)


def _load_ini_options(repo_root: Path) -> tuple[list[str], tuple[str, ...]]:
    benchmark_paths = ["benchmarks"]
    file_patterns: tuple[str, ...] = BENCH_FILE_PATTERNS
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        return benchmark_paths, file_patterns

    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    section = payload.get("tool", {}).get("benchbro", {}).get("ini_options", {})

    configured_paths = section.get("benchmark_paths")
    if isinstance(configured_paths, str):
        benchmark_paths = [configured_paths]
    elif isinstance(configured_paths, list):
        filtered = [
            item for item in configured_paths if isinstance(item, str) and item.strip()
        ]
        if filtered:
            benchmark_paths = filtered

    configured_patterns = section.get("file_pattern")
    if isinstance(configured_patterns, str):
        file_patterns = (configured_patterns,)
    elif isinstance(configured_patterns, list):
        filtered_patterns = tuple(
            item
            for item in configured_patterns
            if isinstance(item, str) and item.strip()
        )
        if filtered_patterns:
            file_patterns = filtered_patterns

    return benchmark_paths, file_patterns


def _discover_python_files(
    target_path: Path, file_patterns: tuple[str, ...], include_all_python: bool = False
) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError(f"Expected a Python file, got: {target_path}")
        return [target_path.resolve()]

    if not target_path.is_dir():
        raise ValueError(f"Not a file or directory: {target_path}")

    return [
        item.resolve()
        for item in sorted(target_path.rglob("*.py"))
        if item.name != "__init__.py"
        and (include_all_python or _matches_bench_pattern(item, file_patterns))
    ]


def _import_file(path: Path, index: int) -> None:
    module_name = f"benchbro_discovered_{path.stem}_{index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _import_target(
    target: str, file_patterns: tuple[str, ...], include_all_python: bool = False
) -> None:
    target_path = Path(target)
    if not target_path.exists():
        importlib.import_module(target)
        return

    for index, file_path in enumerate(
        _discover_python_files(
            target_path,
            file_patterns=file_patterns,
            include_all_python=include_all_python,
        )
    ):
        _import_file(file_path, index)


def _benchmark_key(result) -> tuple[str, str, str]:
    return (result.case_name, result.benchmark_name, result.metric_type)


def _merge_missing_benchmarks_into_baseline(
    baseline_run, current_run, baseline_path: Path
) -> int:
    existing = {_benchmark_key(item) for item in baseline_run.benchmarks}
    missing = [
        item for item in current_run.benchmarks if _benchmark_key(item) not in existing
    ]
    if not missing:
        return 0

    baseline_run.benchmarks.extend(missing)
    write_json(baseline_path, baseline_run)
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _find_repo_root(Path.cwd())

    registry = get_registry()
    registry.clear()
    if args.target is None:
        benchmark_paths, file_patterns = _load_ini_options(repo_root)
        targets = [(repo_root / path) for path in benchmark_paths]
        existing_targets = [path for path in targets if path.exists()]
        if not existing_targets:
            print(
                "No target provided and configured benchmark paths do not exist: "
                + ", ".join(str(path) for path in targets),
                file=sys.stderr,
            )
            return 1
        for target in existing_targets:
            _import_target(
                str(target), file_patterns=file_patterns, include_all_python=False
            )
    else:
        _import_target(
            args.target,
            file_patterns=BENCH_FILE_PATTERNS,
            include_all_python=False,
        )

    cases = registry.all_cases()
    selected = filter_cases(cases, names=args.case or None, tags=args.tag or None)
    run = run_cases(
        selected,
        repeats=args.repeats,
        warmup=args.warmup,
        min_iterations=args.min_iterations,
    )
    _print_results(run)
    if args.histogram:
        _print_histograms(run)

    if args.output_json:
        write_json(args.output_json, run)
    if args.output_csv:
        write_csv(args.output_csv, run)
    if args.output_md:
        write_markdown(args.output_md, run)

    baseline_path = _baseline_output_path(use_ci=args.ci)
    if args.new_baseline:
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved new baseline:[/green] {baseline_path}")
        return 0

    if not baseline_path.exists():
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved baseline:[/green] {baseline_path}")
        return 0

    if args.no_compare:
        baseline = read_json(baseline_path)
        merged_count = _merge_missing_benchmarks_into_baseline(
            baseline, run, baseline_path
        )
        if merged_count:
            _CONSOLE.print(
                f"[cyan]Merged {merged_count} new benchmark(s) into baseline:[/cyan] {baseline_path}"
            )
        _CONSOLE.print("[yellow]Comparison skipped (--no-compare).[/yellow]")
        return 0

    baseline = read_json(baseline_path)
    regressions = compare_runs(baseline, run)

    merged_count = _merge_missing_benchmarks_into_baseline(baseline, run, baseline_path)
    if merged_count:
        _CONSOLE.print(
            f"[cyan]Merged {merged_count} new benchmark(s) into baseline:[/cyan] {baseline_path}"
        )

    return _print_regressions(regressions)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
