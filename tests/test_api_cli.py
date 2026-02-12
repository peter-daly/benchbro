from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

from benchbro import Case, clear_registry, list_cases, run_cases
from benchbro.cli import main


def test_case_groups_multiple_benchmarks() -> None:
    clear_registry()
    case = Case(name="hashing", case_type="cpu", metric_type="time", repeats=1, min_iterations=1, warmup_iterations=0)

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


def test_case_memory_result_shape() -> None:
    clear_registry()
    case = Case(name="mem_case", case_type="memory", metric_type="memory", repeats=2, min_iterations=3, warmup_iterations=0)

    @case.benchmark()
    def allocate() -> list[int]:
        return [1, 2, 3, 4, 5]

    run = run_cases(list_cases())
    assert len(run.benchmarks) == 1

    result = run.benchmarks[0]
    assert result.metric_type == "memory"
    assert "peak_alloc_bytes" in result.metrics
    assert "net_alloc_bytes" in result.metrics


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
    try:
        output_path = tmp_path / "out.json"
        markdown_path = tmp_path / "out.md"
        code = main(
            [
                "run",
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
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "# Benchbro Results" in markdown
        assert "| case | benchmark | metric_type |" in markdown
        assert "mod_group" in markdown
        assert "mod_case" in markdown
    finally:
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
    code = main(
        [
            "run",
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
    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmarks"][0]["case_name"] == "file_case"
    assert payload["benchmarks"][0]["benchmark_name"] == "file_bench"


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
    code = main(
        [
            "run",
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
        code = main(["run", str(benchmark_file), "--warmup", "0", "--repeats", "1", "--min-iterations", "1"])
        assert code == 0

        default_json = repo_root / ".benchbro" / "current.json"
        default_csv = repo_root / ".benchbro" / "current.csv"
        assert default_json.exists()
        assert default_csv.exists()

        payload = json.loads(default_json.read_text(encoding="utf-8"))
        assert payload["benchmarks"][0]["case_name"] == "default_output_case"
        assert payload["benchmarks"][0]["benchmark_name"] == "default_output_bench"
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
                "run",
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
