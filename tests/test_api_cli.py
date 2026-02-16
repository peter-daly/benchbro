from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import pytest

from benchbro import (
    BenchmarkResult,
    BenchmarkRun,
    Case,
    clear_registry,
    compare_runs,
    list_cases,
    read_json,
    run_cases,
    write_json,
)
from benchbro.cli import main


def test_case_groups_multiple_benchmarks() -> None:
    clear_registry()
    case = Case(
        name="hashing",
        case_type="cpu",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
    )

    @case.input()
    def input_data() -> bytes:
        return b"abc"

    @case.benchmark()
    def sha1(input_data: bytes) -> int:
        return len(input_data)

    @case.benchmark()
    def sha256(input_data: bytes) -> int:
        return len(input_data) * 2

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 2
    assert "python_version" in run.environment
    assert "platform" in run.environment
    assert {r.case_name for r in run.benchmarks} == {"hashing"}
    assert {r.benchmark_name for r in run.benchmarks} == {"sha1", "sha256"}
    assert {r.regression_threshold_pct for r in run.benchmarks} == {100.0}
    assert {r.warning_threshold_pct for r in run.benchmarks} == {50.0}


def test_case_threshold_used_when_benchmark_override_missing() -> None:
    clear_registry()
    case = Case(
        name="threshold_case",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        regression_threshold_pct=12.5,
    )

    @case.benchmark()
    def bench_default() -> int:
        return 1

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1
    assert run.benchmarks[0].regression_threshold_pct == 12.5
    assert run.benchmarks[0].warning_threshold_pct == 50.0


def test_benchmark_threshold_override_supersedes_case_threshold() -> None:
    clear_registry()
    case = Case(
        name="threshold_override_case",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        regression_threshold_pct=20.0,
    )

    @case.benchmark(regression_threshold_pct=7.5)
    def bench_override() -> int:
        return 1

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1
    assert run.benchmarks[0].regression_threshold_pct == 7.5
    assert run.benchmarks[0].warning_threshold_pct == 50.0


def test_case_warning_used_when_benchmark_warning_override_missing() -> None:
    clear_registry()
    case = Case(
        name="warning_case",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        warning_threshold_pct=12.0,
    )

    @case.benchmark()
    def bench_default_warning() -> int:
        return 1

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1
    assert run.benchmarks[0].warning_threshold_pct == 12.0


def test_benchmark_warning_override_supersedes_case_warning() -> None:
    clear_registry()
    case = Case(
        name="warning_override_case",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        warning_threshold_pct=20.0,
    )

    @case.benchmark(warning_threshold_pct=7.0)
    def bench_warning_override() -> int:
        return 1

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1
    assert run.benchmarks[0].warning_threshold_pct == 7.0


def test_case_memory_result_shape() -> None:
    clear_registry()
    case = Case(
        name="mem_case",
        case_type="memory",
        metric_type="memory",
        repeats=2,
        min_iterations=3,
        warmup_iterations=0,
    )

    @case.benchmark()
    def allocate() -> list[int]:
        return [1, 2, 3, 4, 5]

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1

    result = run.benchmarks[0]
    assert result.metric_type == "memory"
    assert "peak_alloc_bytes" in result.metrics
    assert "net_alloc_bytes" in result.metrics


def test_compare_runs_uses_current_result_threshold() -> None:
    baseline = BenchmarkRun(
        started_at="t0",
        finished_at="t1",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="cmp_case",
                benchmark_name="time_bench",
                case_type="cpu",
                metric_type="time",
                gc_control="disable_during_measure",
                regression_threshold_pct=100.0,
                iterations=1,
                repeats=1,
                metrics={"mean_s": 1.0},
                warning_threshold_pct=50.0,
            )
        ],
    )
    current = BenchmarkRun(
        started_at="t2",
        finished_at="t3",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="cmp_case",
                benchmark_name="time_bench",
                case_type="cpu",
                metric_type="time",
                gc_control="disable_during_measure",
                regression_threshold_pct=10.0,
                iterations=1,
                repeats=1,
                metrics={"mean_s": 1.2},
                warning_threshold_pct=5.0,
            )
        ],
    )
    regressions = compare_runs(baseline, current)
    assert len(regressions) == 1
    assert regressions[0].threshold_pct == 10.0
    assert regressions[0].warning_threshold_pct == 5.0
    assert regressions[0].is_regression
    assert not regressions[0].is_warning


def test_compare_runs_threshold_equality_is_not_regression() -> None:
    baseline = BenchmarkRun(
        started_at="t0",
        finished_at="t1",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="eq_case",
                benchmark_name="eq_bench",
                case_type="cpu",
                metric_type="time",
                gc_control="disable_during_measure",
                regression_threshold_pct=100.0,
                iterations=1,
                repeats=1,
                metrics={"mean_s": 1.0},
                warning_threshold_pct=50.0,
            )
        ],
    )
    current = BenchmarkRun(
        started_at="t2",
        finished_at="t3",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="eq_case",
                benchmark_name="eq_bench",
                case_type="cpu",
                metric_type="time",
                gc_control="disable_during_measure",
                regression_threshold_pct=10.0,
                iterations=1,
                repeats=1,
                metrics={"mean_s": 1.1},
                warning_threshold_pct=5.0,
            )
        ],
    )
    regressions = compare_runs(baseline, current)
    assert len(regressions) == 1
    assert regressions[0].percent_change == pytest.approx(10.0)
    assert not regressions[0].is_regression
    assert regressions[0].is_warning


def test_compare_runs_memory_uses_peak_alloc_threshold() -> None:
    baseline = BenchmarkRun(
        started_at="t0",
        finished_at="t1",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="mem_cmp",
                benchmark_name="alloc",
                case_type="memory",
                metric_type="memory",
                gc_control="disable_during_measure",
                regression_threshold_pct=100.0,
                iterations=1,
                repeats=1,
                metrics={"peak_alloc_bytes": 100.0},
                warning_threshold_pct=50.0,
            )
        ],
    )
    current = BenchmarkRun(
        started_at="t2",
        finished_at="t3",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="mem_cmp",
                benchmark_name="alloc",
                case_type="memory",
                metric_type="memory",
                gc_control="disable_during_measure",
                regression_threshold_pct=50.0,
                iterations=1,
                repeats=1,
                metrics={"peak_alloc_bytes": 130.0},
                warning_threshold_pct=25.0,
            )
        ],
    )
    regressions = compare_runs(baseline, current)
    assert len(regressions) == 1
    assert regressions[0].metric_name == "peak_alloc_bytes"
    assert not regressions[0].is_regression
    assert regressions[0].is_warning


def test_read_json_defaults_threshold_for_older_payload(tmp_path: Path) -> None:
    payload = {
        "started_at": "t0",
        "finished_at": "t1",
        "python_version": "3.13",
        "platform": "test",
        "environment": {},
        "benchmarks": [
            {
                "case_name": "old_case",
                "benchmark_name": "old_bench",
                "case_type": "cpu",
                "metric_type": "time",
                "gc_control": "disable_during_measure",
                "iterations": 1,
                "repeats": 1,
                "metrics": {"mean_s": 1.0},
            }
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = read_json(path)
    assert run.benchmarks[0].regression_threshold_pct == 100.0
    assert run.benchmarks[0].environment == {}
    assert run.benchmarks[0].warning_threshold_pct == 50.0


def test_read_json_backfills_benchmark_environment_from_run_environment(
    tmp_path: Path,
) -> None:
    payload = {
        "started_at": "t0",
        "finished_at": "t1",
        "python_version": "3.13",
        "platform": "test",
        "environment": {"python_version": "3.13", "platform": "test-platform"},
        "benchmarks": [
            {
                "case_name": "old_case",
                "benchmark_name": "old_bench",
                "case_type": "cpu",
                "metric_type": "time",
                "gc_control": "disable_during_measure",
                "iterations": 1,
                "repeats": 1,
                "metrics": {"mean_s": 1.0},
            }
        ],
    }
    path = tmp_path / "old_with_env.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = read_json(path)
    assert run.benchmarks[0].environment == payload["environment"]


def test_write_json_includes_threshold_field(tmp_path: Path) -> None:
    run = BenchmarkRun(
        started_at="t0",
        finished_at="t1",
        python_version="3.13",
        platform="test",
        environment={},
        benchmarks=[
            BenchmarkResult(
                case_name="case",
                benchmark_name="bench",
                case_type="cpu",
                metric_type="time",
                gc_control="disable_during_measure",
                regression_threshold_pct=33.0,
                iterations=1,
                repeats=1,
                metrics={"mean_s": 1.0},
                warning_threshold_pct=22.0,
            )
        ],
    )
    path = tmp_path / "new.json"
    write_json(path, run)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["benchmarks"][0]["regression_threshold_pct"] == 33.0
    assert payload["benchmarks"][0]["warning_threshold_pct"] == 22.0
    assert "environment" in payload["benchmarks"][0]


def test_cli_rejects_removed_fail_on_regression_flag(tmp_path: Path) -> None:
    clear_registry()
    module_path = tmp_path / "bench_flag.py"
    module_path.write_text(
        "from benchbro import Case\n"
        "case = Case(name='flag_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='flag_bench')\n"
        "def flag_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main([str(module_path), "--fail-on-regression", "5"])


def test_gc_control_disable_during_measure_restores_enabled_state() -> None:
    clear_registry()
    gc.enable()

    case = Case(
        name="gc_enabled_case",
        metric_type="time",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        gc_control="disable_during_measure",
    )

    @case.benchmark()
    def gc_enabled_bench() -> int:
        return 1

    run_cases(list_cases())
    assert gc.isenabled()


def test_gc_control_disable_during_measure_preserves_initial_disabled_state() -> None:
    clear_registry()
    gc.disable()
    try:
        case = Case(
            name="gc_disabled_case",
            metric_type="time",
            repeats=1,
            min_iterations=1,
            warmup_iterations=0,
            gc_control="disable_during_measure",
        )

        @case.benchmark()
        def gc_disabled_bench() -> int:
            return 1

        run_cases(list_cases())
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_cli_run_writes_json(tmp_path: Path) -> None:
    clear_registry()

    module_path = tmp_path / "bench_mod.py"
    module_path.write_text(
        "from benchbro import Case\n"
        "case = Case(name='mod_group', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='mod_case')\n"
        "def mod_case():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output_path = tmp_path / "out.json"
        markdown_path = tmp_path / "out.md"
        code = main(
            [
                "bench_mod",
                "--output-json",
                str(output_path),
                "--output-md",
                str(markdown_path),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert "environment" in payload
        assert "python_version" in payload["environment"]
        assert payload["benchmarks"][0]["metric_type"] == "time"
        assert payload["benchmarks"][0]["case_name"] == "mod_group"
        assert payload["benchmarks"][0]["benchmark_name"] == "mod_case"
        assert "metrics" in payload["benchmarks"][0]
        assert "environment" in payload["benchmarks"][0]
        assert (
            payload["benchmarks"][0]["environment"]["python_version"]
            == payload["environment"]["python_version"]
        )
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "# Benchbro Results" in markdown
        assert "| case | benchmark | metric_type |" in markdown
        assert "mod_group" in markdown
        assert "mod_case" in markdown
    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(tmp_path))


def test_cli_run_accepts_python_file_path(tmp_path: Path) -> None:
    clear_registry()

    module_path = tmp_path / "bench_file.py"
    module_path.write_text(
        "from benchbro import Case\n"
        "case = Case(name='file_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='file_bench')\n"
        "def file_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "out_file.json"
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        code = main(
            [
                str(module_path),
                "--output-json",
                str(output_path),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
    finally:
        os.chdir(original_cwd)
    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmarks"][0]["case_name"] == "file_case"
    assert payload["benchmarks"][0]["benchmark_name"] == "file_bench"


def test_cli_runs_without_subcommands(tmp_path: Path) -> None:
    clear_registry()

    module_path = tmp_path / "bench_shorthand.py"
    module_path.write_text(
        "from benchbro import Case\n"
        "case = Case(name='short_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='short_bench')\n"
        "def short_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "short.json"
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        code = main(
            [
                str(module_path),
                "--output-json",
                str(output_path),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
    finally:
        os.chdir(original_cwd)
    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmarks"][0]["case_name"] == "short_case"
    assert payload["benchmarks"][0]["benchmark_name"] == "short_bench"


def test_cli_run_accepts_directory_path(tmp_path: Path) -> None:
    clear_registry()

    benchmarks_dir = tmp_path / "benches"
    benchmarks_dir.mkdir()
    (benchmarks_dir / "math_benchmarks.py").write_text(
        "from benchbro import Case\n"
        "case = Case(name='dir_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='dir_bench')\n"
        "def dir_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (benchmarks_dir / "ignore_me.py").write_text("x = 1\n", encoding="utf-8")

    output_path = tmp_path / "out_dir.json"
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        code = main(
            [
                str(benchmarks_dir),
                "--output-json",
                str(output_path),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
    finally:
        os.chdir(original_cwd)
    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["benchmarks"]) == 1
    assert payload["benchmarks"][0]["case_name"] == "dir_case"
    assert payload["benchmarks"][0]["benchmark_name"] == "dir_bench"


def test_cli_run_writes_default_outputs_to_repo_root(tmp_path: Path) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    benchmark_file = repo_root / "sample_benchmarks.py"
    benchmark_file.write_text(
        "from benchbro import Case\n"
        "case = Case(name='default_output_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='default_output_bench')\n"
        "def default_output_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        code = main(
            [
                str(benchmark_file),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert code == 0

        default_json = repo_root / ".benchbro" / "current.json"
        default_csv = repo_root / ".benchbro" / "current.csv"
        baseline_json = repo_root / ".benchbro" / "baseline.json"
        assert not default_json.exists()
        assert not default_csv.exists()
        assert baseline_json.exists()
    finally:
        os.chdir(original_cwd)


def test_cli_run_creates_repo_baseline_json_when_missing(tmp_path: Path) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    benchmark_file = repo_root / "sample_benchmarks.py"
    benchmark_file.write_text(
        "from benchbro import Case\n"
        "case = Case(name='baseline_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='baseline_bench')\n"
        "def baseline_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        code = main(
            [
                str(benchmark_file),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert code == 0

        baseline_json = repo_root / ".benchbro" / "baseline.json"
        assert baseline_json.exists()
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        assert payload["benchmarks"][0]["case_name"] == "baseline_case"
        assert payload["benchmarks"][0]["benchmark_name"] == "baseline_bench"
        assert "environment" in payload
        assert "gc_enabled_at_start" in payload["environment"]
        assert "gc_threshold" in payload["environment"]
    finally:
        os.chdir(original_cwd)


def test_cli_run_uses_default_benchmarks_directory(tmp_path: Path) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    benchmarks_dir = repo_root / "benchmarks" / "nested"
    benchmarks_dir.mkdir(parents=True)

    benchmark_file = benchmarks_dir / "sample_benchmarks.py"
    benchmark_file.write_text(
        "from benchbro import Case\n"
        "case = Case(name='default_target_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='default_target_bench')\n"
        "def default_target_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    non_matching_file = benchmarks_dir / "custom_name.py"
    non_matching_file.write_text(
        "from benchbro import Case\n"
        "case = Case(name='default_target_case_extra', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='default_target_bench_extra')\n"
        "def default_target_bench_extra():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    out_json = repo_root / "artifacts.json"
    out_csv = repo_root / "artifacts.csv"
    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        code = main(
            [
                "--output-json",
                str(out_json),
                "--output-csv",
                str(out_csv),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert code == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert {item["case_name"] for item in payload["benchmarks"]} == {
            "default_target_case",
            "default_target_case_extra",
        }
        assert {item["benchmark_name"] for item in payload["benchmarks"]} == {
            "default_target_bench",
            "default_target_bench_extra",
        }
        assert out_csv.exists()
    finally:
        os.chdir(original_cwd)


def test_cli_run_merges_new_benchmark_into_existing_baseline(tmp_path: Path) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    first_module = repo_root / "bench_one.py"
    first_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='merge_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='one')\n"
        "def one():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    second_module = repo_root / "bench_two.py"
    second_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='merge_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='one')\n"
        "def one():\n"
        "    return 1\n"
        "@case.benchmark(name='two')\n"
        "def two():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        first_code = main(
            [
                str(first_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert first_code == 0

        second_code = main(
            [
                str(second_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert second_code == 0

        baseline_json = repo_root / ".benchbro" / "baseline.json"
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        names = {
            (item["case_name"], item["benchmark_name"])
            for item in payload["benchmarks"]
        }
        assert ("merge_case", "one") in names
        assert ("merge_case", "two") in names
    finally:
        os.chdir(original_cwd)


def test_cli_comparison_output_includes_threshold_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    baseline_module = repo_root / "bench_baseline.py"
    baseline_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='cmp_table_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='cmp_bench')\n"
        "def cmp_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    current_module = repo_root / "bench_current.py"
    current_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='cmp_table_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0, warning_threshold_pct=2.0, regression_threshold_pct=5.0)\n"
        "@case.benchmark(name='cmp_bench')\n"
        "def cmp_bench():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        first_code = main(
            [
                str(baseline_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert first_code == 0

        second_code = main(
            [
                str(current_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert second_code in (0, 2)
    finally:
        os.chdir(original_cwd)

    output = capsys.readouterr().out
    assert "Benchmark Comparison" in output
    assert "/" in output
    assert "5.00%" in output


def test_cli_shows_warning_status_between_warning_and_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    baseline_module = repo_root / "bench_warning_base.py"
    baseline_module.write_text(
        "from benchbro import Case\n"
        "import time\n"
        "case = Case(name='warn_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='warn_bench')\n"
        "def warn_bench():\n"
        "    time.sleep(0.002)\n",
        encoding="utf-8",
    )

    current_module = repo_root / "bench_warning_current.py"
    current_module.write_text(
        "from benchbro import Case\n"
        "import time\n"
        "case = Case(name='warn_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0, warning_threshold_pct=10.0, regression_threshold_pct=100.0)\n"
        "@case.benchmark(name='warn_bench')\n"
        "def warn_bench():\n"
        "    time.sleep(0.003)\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        first_code = main(
            [
                str(baseline_module),
                "--new-baseline",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert first_code == 0

        second_code = main(
            [
                str(current_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert second_code == 0
    finally:
        os.chdir(original_cwd)

    output = capsys.readouterr().out
    assert "WARN" in output


def test_cli_new_baseline_replaces_existing_baseline(tmp_path: Path) -> None:
    clear_registry()

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    first_module = repo_root / "bench_original.py"
    first_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='replace_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='original')\n"
        "def original():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    second_module = repo_root / "bench_new.py"
    second_module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='replace_case', metric_type='time', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='new_only')\n"
        "def new_only():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        first_code = main(
            [
                str(first_module),
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert first_code == 0

        second_code = main(
            [
                str(second_module),
                "--new-baseline",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--min-iterations",
                "1",
            ]
        )
        assert second_code == 0

        baseline_json = repo_root / ".benchbro" / "baseline.json"
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        names = {
            (item["case_name"], item["benchmark_name"])
            for item in payload["benchmarks"]
        }
        assert names == {("replace_case", "new_only")}
    finally:
        os.chdir(original_cwd)
