from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import math
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from benchbro.api import clear_registry, get_registry
from benchbro.comparison import compare_runs
from benchbro.config import DEFAULT_FILE_PATTERNS, BenchbroConfig, load_config
from benchbro.core import filter_cases
from benchbro.history import list_history, load_history, save_history
from benchbro.models import BenchmarkCase, BenchmarkRun
from benchbro.runner import run_cases
from benchbro.serialization import read_json, write_csv, write_json, write_markdown

BENCH_FILE_PATTERNS = DEFAULT_FILE_PATTERNS
LOCAL_BASELINE_JSON_PATH = Path(".benchbro/baseline.local.json")
CI_BASELINE_JSON_PATH = Path(".benchbro/baseline.ci.json")
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REGRESSION = 2
EXIT_EXECUTION = 3
_COMMANDS = {"run", "list", "compare", "baseline", "history", "init", "completion"}
_CONSOLE = Console()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _cpu_list(value: str) -> tuple[int, ...]:
    try:
        cpus = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated CPU indexes") from exc
    if not cpus or any(cpu < 0 for cpu in cpus):
        raise argparse.ArgumentTypeError("must be comma-separated non-negative indexes")
    return cpus


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def _print_results(run: BenchmarkRun) -> None:
    env_table = Table(title="Environment")
    env_table.add_column("Key", style="cyan", no_wrap=True)
    env_table.add_column("Value", style="bold")
    for key in sorted(run.environment):
        env_table.add_row(key, run.environment[key])
    _CONSOLE.print(env_table)

    by_type: dict[str, list[Any]] = defaultdict(list)
    for result in run.benchmarks:
        by_type[result.metric_type].append(result)
    for metric_type in sorted(by_type):
        table = Table(title=f"Benchbro Results: {metric_type}")
        table.add_column("Case", style="cyan", no_wrap=True)
        table.add_column("Benchmark", style="bold")
        if metric_type == "time":
            table.add_column("Median (s)", justify="right")
            table.add_column("95% CI (s)", justify="right")
            table.add_column("CV %", justify="right")
            table.add_column("P95 (s)", justify="right")
            table.add_column("Ops/s", justify="right")
            table.add_column("Samples", justify="right")
            table.add_column("Quality")
        else:
            table.add_column("Peak (B)", justify="right")
            table.add_column("Net (B)", justify="right")
            table.add_column("Peak Max (B)", justify="right")
            table.add_column("CV %", justify="right")
            table.add_column("Quality")
        for result in by_type[metric_type]:
            metrics = result.metrics
            quality = (
                "[yellow]NOISY[/yellow]" if result.is_noisy else "[green]stable[/green]"
            )
            if metric_type == "time":
                ci = (
                    f"{_format_metric(metrics.get('ci95_low_s'))}–"
                    f"{_format_metric(metrics.get('ci95_high_s'))}"
                )
                table.add_row(
                    result.case_name,
                    result.benchmark_name,
                    _format_metric(metrics.get("median_s")),
                    ci,
                    _format_metric(metrics.get("cv_pct")),
                    _format_metric(metrics.get("p95_s")),
                    _format_metric(metrics.get("ops_per_sec")),
                    str(result.repeats),
                    quality,
                )
            else:
                table.add_row(
                    result.case_name,
                    result.benchmark_name,
                    _format_metric(metrics.get("peak_alloc_bytes")),
                    _format_metric(metrics.get("net_alloc_bytes")),
                    _format_metric(metrics.get("peak_alloc_bytes_max")),
                    _format_metric(metrics.get("cv_pct")),
                    quality,
                )
        _CONSOLE.print(table)


def _print_regressions(regressions: list[Any]) -> int:
    if not regressions:
        _CONSOLE.print("[yellow]No comparable benchmarks found.[/yellow]")
        return EXIT_OK
    table = Table(title="Benchmark Comparison")
    table.add_column("Benchmark / values", style="cyan")
    table.add_column("Change", justify="right")
    table.add_column("Thresholds", justify="right", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    for item in regressions:
        styles = {
            "regression": ("red", "REGRESSION"),
            "likely_regression": ("yellow", "LIKELY"),
            "inconclusive": ("yellow", "INCONCLUSIVE"),
            "warning": ("yellow", "WARNING"),
            "noisy": ("yellow", "NOISY"),
            "improvement": ("green", "IMPROVED"),
            "stable": ("green", "OK"),
        }
        style, label = styles.get(item.status, ("white", item.status.upper()))
        table.add_row(
            (
                f"[bold]{item.case_name}::{item.benchmark_name}[/bold]\n"
                f"[magenta]{item.metric_name}[/magenta]: "
                f"{item.baseline_value:.6g} → {item.current_value:.6g}"
            ),
            (
                f"[{style}]{item.percent_change:+.2f}%[/{style}]\n"
                f"[dim]{item.confidence_pct:.1f}% confidence[/dim]"
            ),
            (
                f"[yellow]{item.warning_threshold_pct:.2f}%[/yellow] / "
                f"[red]{item.threshold_pct:.2f}%[/red]"
            ),
            f"[{style}]{label}[/{style}]",
        )
    _CONSOLE.print(table)
    return (
        EXIT_REGRESSION if any(item.is_regression for item in regressions) else EXIT_OK
    )


def _build_histogram(
    samples: list[float], bins: int = 10
) -> list[tuple[float, float, int]]:
    if not samples:
        return []
    if len(samples) == 1:
        return [(samples[0], samples[0], 1)]
    low, high = min(samples), max(samples)
    if math.isclose(low, high):
        return [(low, high, len(samples))]
    width = (high - low) / bins
    counts = [0] * bins
    for sample in samples:
        index = bins - 1 if sample == high else int((sample - low) / width)
        counts[max(0, min(index, bins - 1))] += 1
    return [
        (low + width * index, low + width * (index + 1), count)
        for index, count in enumerate(counts)
        if count
    ]


def _print_histograms(run: BenchmarkRun) -> None:
    results = [
        result
        for result in run.benchmarks
        if result.metric_type == "time" and result.time_samples_s
    ]
    if not results:
        _CONSOLE.print(
            "[yellow]No time samples available for histogram output.[/yellow]"
        )
        return
    for result in results:
        histogram = _build_histogram(result.time_samples_s)
        max_count = max(count for _, _, count in histogram)
        table = Table(title=f"Histogram: {result.case_name}::{result.benchmark_name}")
        table.add_column("Range (s)", style="cyan")
        table.add_column("Distribution", style="bold")
        table.add_column("Count", justify="right")
        for low, high, count in histogram:
            bar = "█" * max(1, int((count / max_count) * 24))
            table.add_row(f"{low:.6g} - {high:.6g}", bar, str(count))
        _CONSOLE.print(table)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", nargs="?", help="Module, Python file, or directory")
    parser.add_argument("--config")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--repeats", type=_positive_int)
    parser.add_argument("--warmup", type=_non_negative_int)
    parser.add_argument("--min-iterations", type=_positive_int)
    adaptive = parser.add_mutually_exclusive_group()
    adaptive.add_argument("--adaptive", action="store_true", default=None)
    adaptive.add_argument("--fixed", action="store_false", dest="adaptive")
    parser.add_argument("--min-repeats", type=_positive_int)
    parser.add_argument("--min-time", type=_non_negative_float)
    parser.add_argument("--max-time", type=_positive_float)
    parser.add_argument("--target-margin", type=_non_negative_float)
    parser.add_argument("--noise-threshold", type=_non_negative_float)
    parser.add_argument("--stabilization-delay", type=_non_negative_float)
    parser.add_argument("--isolation", choices=("in_process", "process"))
    parser.add_argument("--cpu-affinity", type=_cpu_list, metavar="CPU[,CPU...]")
    parser.add_argument("--timeout", type=_positive_float)
    parser.add_argument("--profile", metavar="PROFILER")
    parser.add_argument("--profile-output")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-md")
    parser.add_argument("--baseline", metavar="NAME")
    parser.add_argument("--new-baseline", action="store_true")
    parser.add_argument("--save-history", action="store_true", default=None)
    parser.add_argument("--histogram", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchbro",
        description="Run benchmarks (legacy target-first syntax is supported).",
        epilog="Commands: run, list, compare, baseline, history, init, completion",
    )
    _add_run_arguments(parser)
    return parser


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchbro")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Discover and run benchmarks")
    _add_run_arguments(run_parser)
    list_parser = commands.add_parser("list", help="List discovered benchmarks")
    list_parser.add_argument("target", nargs="?")
    list_parser.add_argument("--config")
    list_parser.add_argument("--case", action="append", default=[])
    list_parser.add_argument("--tag", action="append", default=[])
    list_parser.add_argument("--verbose", action="store_true")
    compare_parser = commands.add_parser("compare", help="Compare two result files")
    compare_parser.add_argument("baseline_file")
    compare_parser.add_argument("current_file")
    compare_parser.add_argument("--allow-environment-mismatch", action="store_true")
    baseline_parser = commands.add_parser("baseline", help="Manage named baselines")
    baseline_commands = baseline_parser.add_subparsers(
        dest="baseline_command", required=True
    )
    baseline_update = baseline_commands.add_parser("update")
    _add_run_arguments(baseline_update)
    baseline_list = baseline_commands.add_parser("list")
    baseline_list.add_argument("--config")
    history_parser = commands.add_parser(
        "history", help="Inspect local benchmark history"
    )
    history_commands = history_parser.add_subparsers(
        dest="history_command", required=True
    )
    history_list = history_commands.add_parser("list")
    history_list.add_argument("--limit", type=_positive_int, default=20)
    history_show = history_commands.add_parser("show")
    history_show.add_argument("selector")
    history_compare = history_commands.add_parser("compare")
    history_compare.add_argument("baseline")
    history_compare.add_argument("current")
    history_compare.add_argument("--allow-environment-mismatch", action="store_true")
    init_parser = commands.add_parser(
        "init", help="Create starter configuration and benchmark"
    )
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--force", action="store_true")
    completion_parser = commands.add_parser(
        "completion", help="Print a shell completion script"
    )
    completion_parser.add_argument("shell", choices=("bash", "zsh", "fish"))
    return parser


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _validate_baseline_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            "Baseline names may contain letters, numbers, '.', '_' and '-'"
        )
    return name


def _baseline_output_path(use_ci: bool = False, name: str | None = None) -> Path:
    repo_root = _find_repo_root(Path.cwd())
    output_dir = repo_root / ".benchbro"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = "ci" if use_ci else (name or "local")
    _validate_baseline_name(selected)
    if selected == "local":
        return output_dir / LOCAL_BASELINE_JSON_PATH.name
    if selected == "ci":
        return output_dir / CI_BASELINE_JSON_PATH.name
    return output_dir / "baselines" / f"{selected}.json"


def _matches_bench_pattern(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _load_ini_options(repo_root: Path) -> tuple[list[str], tuple[str, ...]]:
    config = load_config(repo_root)
    return list(config.benchmark_paths), config.file_patterns


def _discover_python_files(
    target_path: Path,
    file_patterns: tuple[str, ...],
    include_all_python: bool = False,
) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix != ".py":
            raise ValueError(f"Expected a Python file, got: {target_path}")
        return [target_path.resolve()]
    if not target_path.is_dir():
        raise ValueError(f"Not a file or directory: {target_path}")
    return [
        path.resolve()
        for path in sorted(target_path.rglob("*.py"))
        if path.name != "__init__.py"
        and (include_all_python or _matches_bench_pattern(path, file_patterns))
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
    target: str,
    file_patterns: tuple[str, ...],
    include_all_python: bool = False,
) -> None:
    target_path = Path(target)
    if not target_path.exists():
        if target in sys.modules:
            importlib.reload(sys.modules[target])
        else:
            importlib.import_module(target)
        return
    for index, path in enumerate(
        _discover_python_files(target_path, file_patterns, include_all_python)
    ):
        _import_file(path, index)


def _discover_cases(
    target: str | None,
    repo_root: Path,
    config: BenchbroConfig,
) -> list[BenchmarkCase]:
    clear_registry()
    if target is not None:
        _import_target(target, config.file_patterns)
    else:
        targets = [repo_root / path for path in config.benchmark_paths]
        existing = [path for path in targets if path.exists()]
        if not existing:
            raise ValueError(
                "No target provided and configured benchmark paths do not exist: "
                + ", ".join(str(path) for path in targets)
            )
        for path in existing:
            _import_target(str(path), config.file_patterns)
    return get_registry().all_cases()


def _benchmark_key(result: Any) -> tuple[str, str, str]:
    return result.case_name, result.benchmark_name, result.metric_type


def _missing_baseline_benchmarks(
    baseline: BenchmarkRun, current: BenchmarkRun
) -> list[Any]:
    existing = {_benchmark_key(item) for item in baseline.benchmarks}
    return [item for item in current.benchmarks if _benchmark_key(item) not in existing]


def _merge_missing_benchmarks_into_baseline(
    baseline: BenchmarkRun, current: BenchmarkRun, path: Path
) -> int:
    missing = _missing_baseline_benchmarks(baseline, current)
    if missing:
        baseline.benchmarks.extend(missing)
        write_json(path, baseline)
    return len(missing)


def _apply_case_overrides(
    cases: list[BenchmarkCase], args: argparse.Namespace
) -> list[BenchmarkCase]:
    updates = {
        "timeout_s": getattr(args, "timeout", None),
        "profiler": getattr(args, "profile", None),
        "profile_output": getattr(args, "profile_output", None),
    }
    updates = {key: value for key, value in updates.items() if value is not None}
    return [replace(case, settings=replace(case.settings, **updates)) for case in cases]


def _run_command(args: argparse.Namespace, *, force_baseline: bool = False) -> int:
    repo_root = _find_repo_root(Path.cwd())
    try:
        config = load_config(repo_root, getattr(args, "config", None))
        cases = _discover_cases(args.target, repo_root, config)
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    if not cases:
        print("No benchmarks were discovered.", file=sys.stderr)
        return EXIT_USAGE
    selected = filter_cases(cases, names=args.case or None, tags=args.tag or None)
    if not selected:
        print("No benchmarks matched the requested case/tag filters.", file=sys.stderr)
        return EXIT_USAGE
    selected = _apply_case_overrides(selected, args)
    repeats = args.repeats if args.repeats is not None else config.repeats
    warmup = args.warmup if args.warmup is not None else config.warmup
    iterations = (
        args.min_iterations
        if args.min_iterations is not None
        else config.min_iterations
    )
    adaptive = args.adaptive if args.adaptive is not None else config.adaptive
    min_repeats = (
        args.min_repeats if args.min_repeats is not None else config.min_repeats
    )
    min_time = args.min_time if args.min_time is not None else config.min_time_s
    max_time = args.max_time if args.max_time is not None else config.max_time_s
    target_margin = (
        args.target_margin
        if args.target_margin is not None
        else config.target_relative_margin_pct
    )
    noise_threshold = (
        args.noise_threshold
        if args.noise_threshold is not None
        else config.noise_threshold_pct
    )
    stabilization_delay = (
        args.stabilization_delay
        if args.stabilization_delay is not None
        else config.stabilization_delay_s
    )
    if adaptive is None and any(
        value is not None for value in (min_repeats, min_time, max_time, target_margin)
    ):
        adaptive = True
    isolation = args.isolation if args.isolation is not None else config.isolation
    affinity = (
        args.cpu_affinity if args.cpu_affinity is not None else config.cpu_affinity
    )
    try:
        run = run_cases(
            selected,
            repeats=repeats,
            warmup=warmup,
            min_iterations=iterations,
            adaptive=adaptive,
            min_repeats=min_repeats,
            min_time_s=min_time,
            max_time_s=max_time,
            target_relative_margin_pct=target_margin,
            noise_threshold_pct=noise_threshold,
            isolation=isolation,
            cpu_affinity=affinity,
            stabilization_delay_s=stabilization_delay,
        )
    except Exception as exc:  # noqa: BLE001
        _CONSOLE.print(f"[red]Benchmark execution failed:[/red] {exc}")
        return EXIT_EXECUTION
    _print_results(run)
    if args.histogram:
        _print_histograms(run)
    output_json = args.output_json or config.output_json
    output_csv = args.output_csv or config.output_csv
    output_md = args.output_md or config.output_md
    if output_json:
        write_json(output_json, run)
    if output_csv:
        write_csv(output_csv, run)
    if output_md:
        write_markdown(output_md, run)
    if args.save_history if args.save_history is not None else config.save_history:
        path = save_history(repo_root, run)
        _CONSOLE.print(f"[cyan]Saved history:[/cyan] {path}")

    baseline_name = "ci" if args.ci else (args.baseline or config.baseline)
    try:
        baseline_path = _baseline_output_path(name=baseline_name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    new_baseline = args.new_baseline or force_baseline
    if new_baseline:
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved new baseline:[/green] {baseline_path}")
        return EXIT_OK
    if not baseline_path.exists():
        if args.ci:
            _CONSOLE.print(
                "[red]CI baseline is missing.[/red] Create and commit it explicitly with --ci --new-baseline."
            )
            return EXIT_USAGE
        write_json(baseline_path, run)
        _CONSOLE.print(f"[green]Saved baseline:[/green] {baseline_path}")
        return EXIT_OK
    baseline = read_json(baseline_path)
    missing = _missing_baseline_benchmarks(baseline, run)
    if args.ci and missing:
        names = ", ".join(
            f"{item.case_name}::{item.benchmark_name}" for item in missing
        )
        _CONSOLE.print(f"[red]CI baseline is missing benchmark entries:[/red] {names}")
        _CONSOLE.print("Update it explicitly with [bold]--ci --new-baseline[/bold].")
        return EXIT_USAGE
    if args.no_compare:
        merged = (
            _merge_missing_benchmarks_into_baseline(baseline, run, baseline_path)
            if not args.ci
            else 0
        )
        if merged:
            _CONSOLE.print(
                f"[cyan]Merged {merged} new benchmark(s) into baseline:[/cyan] {baseline_path}"
            )
        _CONSOLE.print("[yellow]Comparison skipped (--no-compare).[/yellow]")
        return EXIT_OK
    try:
        regressions = compare_runs(
            baseline,
            run,
            allow_environment_mismatch=args.allow_environment_mismatch,
        )
    except ValueError as exc:
        _CONSOLE.print(f"[red]Cannot compare benchmark runs:[/red] {exc}")
        return EXIT_USAGE
    merged = (
        _merge_missing_benchmarks_into_baseline(baseline, run, baseline_path)
        if not args.ci
        else 0
    )
    if merged:
        _CONSOLE.print(
            f"[cyan]Merged {merged} new benchmark(s) into baseline:[/cyan] {baseline_path}"
        )
    return _print_regressions(regressions)


def _list_command(args: argparse.Namespace) -> int:
    repo_root = _find_repo_root(Path.cwd())
    try:
        config = load_config(repo_root, args.config)
        cases = _discover_cases(args.target, repo_root, config)
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    selected = filter_cases(cases, names=args.case or None, tags=args.tag or None)
    if not selected:
        print("No benchmarks matched.", file=sys.stderr)
        return EXIT_USAGE
    table = Table(title="Discovered Benchmarks")
    table.add_column("Case", style="cyan")
    table.add_column("Benchmark", style="bold")
    table.add_column("Metric")
    table.add_column("Tags")
    if args.verbose:
        table.add_column("Settings")
        table.add_column("Parameters")
    for case in selected:
        row = [
            case.case_name,
            case.benchmark_name,
            case.metric_type,
            ", ".join(case.tags),
        ]
        if args.verbose:
            row.extend(
                [
                    (
                        f"{case.settings.repeats}×{case.settings.min_iterations}; "
                        f"adaptive={case.settings.adaptive}; "
                        f"isolation={case.settings.isolation}"
                    ),
                    ", ".join(
                        f"{key}={value!r}" for key, value in case.parameters.items()
                    ),
                ]
            )
        table.add_row(*row)
    _CONSOLE.print(table)
    return EXIT_OK


def _compare_files(baseline_path: str, current_path: str, allow_mismatch: bool) -> int:
    try:
        regressions = compare_runs(
            read_json(baseline_path),
            read_json(current_path),
            allow_environment_mismatch=allow_mismatch,
        )
    except (OSError, ValueError, KeyError) as exc:
        _CONSOLE.print(f"[red]Cannot compare benchmark runs:[/red] {exc}")
        return EXIT_USAGE
    return _print_regressions(regressions)


def _baseline_list() -> int:
    root = _find_repo_root(Path.cwd())
    candidates = [root / LOCAL_BASELINE_JSON_PATH, root / CI_BASELINE_JSON_PATH]
    candidates.extend(sorted((root / ".benchbro" / "baselines").glob("*.json")))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        _CONSOLE.print("[yellow]No baselines found.[/yellow]")
        return EXIT_OK
    for path in existing:
        _CONSOLE.print(path.relative_to(root))
    return EXIT_OK


def _history_command(args: argparse.Namespace) -> int:
    root = _find_repo_root(Path.cwd())
    if args.history_command == "list":
        paths = list_history(root)[: args.limit]
        if not paths:
            _CONSOLE.print("[yellow]No saved benchmark history.[/yellow]")
        for path in paths:
            run = read_json(path)
            commit = run.environment.get("git_commit", "working-tree")[:12]
            _CONSOLE.print(f"{path.name}  {commit}  {len(run.benchmarks)} benchmark(s)")
        return EXIT_OK
    if args.history_command == "show":
        try:
            path, run = load_history(root, args.selector)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        _CONSOLE.print(f"[cyan]{path}[/cyan]")
        _print_results(run)
        return EXIT_OK
    try:
        _, baseline = load_history(root, args.baseline)
        _, current = load_history(root, args.current)
        regressions = compare_runs(
            baseline,
            current,
            allow_environment_mismatch=args.allow_environment_mismatch,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    return _print_regressions(regressions)


def _init_project(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    config_path = root / "benchbro.toml"
    benchmark_path = root / "benchmarks" / "bench_example.py"
    conflicts = [path for path in (config_path, benchmark_path) if path.exists()]
    if conflicts and not args.force:
        print(
            "Refusing to overwrite existing files: "
            + ", ".join(str(path) for path in conflicts),
            file=sys.stderr,
        )
        return EXIT_USAGE
    root.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        'benchmark_paths = ["benchmarks"]\n'
        'baseline = "local"\n'
        "save_history = true\n\n"
        "[run]\n"
        "repeats = 20\n"
        "warmup = 5\n"
        "min_iterations = 50\n"
        "adaptive = true\n"
        "min_repeats = 5\n"
        "max_time_s = 10.0\n",
        encoding="utf-8",
    )
    benchmark_path.write_text(
        "from benchbro import Case\n\n"
        'case = Case(name="example", warning_threshold_pct=5, regression_threshold_pct=10)\n\n'
        "@case.benchmark()\n"
        '@case.parametrize("size", [100, 1_000], ids=["small", "large"])\n'
        "def build_list(size: int) -> list[int]:\n"
        "    return list(range(size))\n",
        encoding="utf-8",
    )
    _CONSOLE.print(f"[green]Created:[/green] {config_path}")
    _CONSOLE.print(f"[green]Created:[/green] {benchmark_path}")
    return EXIT_OK


def _completion(shell: str) -> int:
    commands = "run list compare baseline history init completion"
    scripts = {
        "bash": f"complete -W '{commands}' benchbro",
        "zsh": f"compctl -k '({commands})' benchbro",
        "fish": "\n".join(
            f"complete -c benchbro -f -a {command}" for command in commands.split()
        ),
    }
    print(scripts[shell])
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _COMMANDS:
        args = _build_command_parser().parse_args(arguments)
        if args.command == "run":
            return _run_command(args)
        if args.command == "list":
            return _list_command(args)
        if args.command == "compare":
            return _compare_files(
                args.baseline_file,
                args.current_file,
                args.allow_environment_mismatch,
            )
        if args.command == "baseline":
            if args.baseline_command == "update":
                return _run_command(args, force_baseline=True)
            return _baseline_list()
        if args.command == "history":
            return _history_command(args)
        if args.command == "init":
            return _init_project(args)
        return _completion(args.shell)
    return _run_command(_build_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
