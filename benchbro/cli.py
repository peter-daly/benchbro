from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchbro.api import get_registry
from benchbro.core import compare_runs, filter_cases, read_json, run_cases, write_csv, write_json, write_markdown

BENCH_FILE_PATTERNS = (
    "bench_*.py",
    "*_bench.py",
    "*benchmark.py",
    "*benchmarks.py",
)

_CONSOLE = Console()
BASELINE_JSON_PATH = Path(".benchbro/baseline.json")


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
        table.add_column("Mean (s)", justify="right")
        table.add_column("Median (s)", justify="right")
        table.add_column("P95 (s)", justify="right")
        table.add_column("StdDev (s)", justify="right")
        table.add_column("Ops/s", justify="right")
        table.add_column("Peak (B)", justify="right")
        table.add_column("Net (B)", justify="right")

        for result in by_type[metric_type]:
            metrics = result.metrics
            table.add_row(
                result.case_name,
                result.benchmark_name,
                _format_metric(metrics.get("mean_s")),
                _format_metric(metrics.get("median_s")),
                _format_metric(metrics.get("p95_s")),
                _format_metric(metrics.get("stddev_s")),
                _format_metric(metrics.get("ops_per_sec")),
                _format_metric(metrics.get("peak_alloc_bytes")),
                _format_metric(metrics.get("net_alloc_bytes")),
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
    table.add_column("Threshold %", justify="right", style="yellow")
    table.add_column("Change %", justify="right")
    table.add_column("Status", justify="center")

    for item in regressions:
        status = "[red]REGRESSION[/red]" if item.is_regression else "[green]OK[/green]"
        change_style = "red" if item.is_regression else "green"
        table.add_row(
            item.case_name,
            item.benchmark_name,
            item.metric_name,
            f"{item.baseline_value:.6g}",
            f"{item.current_value:.6g}",
            f"[yellow]{item.threshold_pct:.2f}%[/yellow]",
            f"[{change_style}]{item.percent_change:+.2f}%[/{change_style}]",
            status,
        )
    _CONSOLE.print(table)
    return 2 if failing else 0


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
    return parser


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _baseline_output_path() -> Path:
    repo_root = _find_repo_root(Path.cwd())
    output_dir = repo_root / ".benchbro"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / BASELINE_JSON_PATH.name


def _matches_bench_pattern(path: Path) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in BENCH_FILE_PATTERNS)


def _discover_python_files(target_path: Path, include_all_python: bool = False) -> list[Path]:
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
        and (include_all_python or _matches_bench_pattern(item))
    ]


def _import_file(path: Path, index: int) -> None:
    module_name = f"benchbro_discovered_{path.stem}_{index}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _import_target(target: str, include_all_python: bool = False) -> None:
    target_path = Path(target)
    if not target_path.exists():
        importlib.import_module(target)
        return

    for index, file_path in enumerate(
        _discover_python_files(target_path, include_all_python=include_all_python)
    ):
        _import_file(file_path, index)


def _benchmark_key(result) -> tuple[str, str, str]:
    return (result.case_name, result.benchmark_name, result.metric_type)


def _merge_missing_benchmarks_into_baseline(baseline_run, current_run, baseline_path: Path) -> int:
    existing = {_benchmark_key(item) for item in baseline_run.benchmarks}
    missing = [item for item in current_run.benchmarks if _benchmark_key(item) not in existing]
    if not missing:
        return 0

    baseline_run.benchmarks.extend(missing)
    write_json(baseline_path, baseline_run)
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    registry = get_registry()
    registry.clear()
    if args.target is None:
        default_target = _find_repo_root(Path.cwd()) / "benchmarks"
        if not default_target.exists():
            print(
                f"No target provided and default path does not exist: {default_target}",
                file=sys.stderr,
            )
            return 1
        target = str(default_target)
    else:
        target = args.target

    _import_target(target, include_all_python=args.target is None)

    cases = registry.all_cases()
    selected = filter_cases(cases, names=args.case or None, tags=args.tag or None)
    run = run_cases(
        selected,
        repeats=args.repeats,
        warmup=args.warmup,
        min_iterations=args.min_iterations,
    )
    _print_results(run)

    if args.output_json:
        write_json(args.output_json, run)
    if args.output_csv:
        write_csv(args.output_csv, run)
    if args.output_md:
        write_markdown(args.output_md, run)

    baseline_path = _baseline_output_path()
    if args.new_baseline:
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved new baseline:[/green] {baseline_path}")
        return 0

    if not baseline_path.exists():
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved baseline:[/green] {baseline_path}")
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
