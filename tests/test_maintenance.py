from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

import benchbro
from benchbro import (
    BenchmarkResult,
    BenchmarkRun,
    Case,
    clear_registry,
    compare_runs,
    comparison_environment_mismatches,
    list_cases,
    run_cases,
    system,
    write_json,
    write_markdown,
)
from benchbro.cli import main


def _run(*, python_version: str = "3.13.1", machine: str = "arm64") -> BenchmarkRun:
    environment = {
        "python_version": python_version,
        "python_implementation": "CPython",
        "system": "Darwin",
        "machine": machine,
    }
    return BenchmarkRun(
        started_at="t0",
        finished_at="t1",
        python_version=python_version,
        platform=f"Darwin-{machine}",
        environment=environment,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "Case name"),
        ({"name": "case", "case_type": ""}, "case_type"),
        ({"name": "case", "metric_type": "latency"}, "metric_type"),
        ({"name": "case", "gc_control": "sometimes"}, "gc_control"),
        ({"name": "case", "warmup_iterations": -1}, "warmup_iterations"),
        ({"name": "case", "min_iterations": 0}, "min_iterations"),
        ({"name": "case", "repeats": 0}, "repeats"),
        ({"name": "case", "regression_threshold_pct": float("nan")}, "regression"),
        ({"name": "case", "warning_threshold_pct": -1}, "warning"),
        ({"name": "case", "tags": [""]}, "tags"),
    ],
)
def test_case_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Case(**kwargs)


def test_case_uses_enough_default_repeats_for_percentiles() -> None:
    assert Case(name="defaults").repeats == 20


def test_duplicate_benchmark_identity_is_rejected() -> None:
    clear_registry()
    case = Case(name="duplicate")

    @case.benchmark(name="same")
    def first() -> int:
        return 1

    with pytest.raises(ValueError, match="already registered"):

        @case.benchmark(name="same")
        def second() -> int:
            return 2


def test_async_lifecycle_uses_one_event_loop() -> None:
    clear_registry()
    case = Case(name="loop-affinity", warmup_iterations=0, min_iterations=1, repeats=2)
    benchmark_loops: list[asyncio.AbstractEventLoop] = []
    teardown_loops: list[asyncio.AbstractEventLoop] = []

    @system()
    async def fixture_loop():
        loop = asyncio.get_running_loop()
        yield loop
        teardown_loops.append(asyncio.get_running_loop())

    @case.input()
    async def input_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    @case.benchmark()
    async def bench(
        fixture_loop: asyncio.AbstractEventLoop,
        input_loop: asyncio.AbstractEventLoop,
    ) -> None:
        current = asyncio.get_running_loop()
        assert fixture_loop is current
        assert input_loop is current
        benchmark_loops.append(current)

    run_cases(list_cases())
    assert len(benchmark_loops) == 2
    assert len({id(loop) for loop in benchmark_loops}) == 1
    assert teardown_loops == [benchmark_loops[0]]
    assert benchmark_loops[0].is_closed()


def test_sync_benchmark_can_run_inside_an_active_event_loop() -> None:
    clear_registry()
    case = Case(name="sync-in-loop", warmup_iterations=0, min_iterations=1, repeats=1)

    @case.benchmark()
    def bench() -> None:
        pass

    async def invoke() -> BenchmarkRun:
        return run_cases(list_cases())

    result = asyncio.run(invoke())
    assert len(result.benchmarks) == 1


def test_async_benchmark_rejects_an_active_event_loop() -> None:
    clear_registry()
    case = Case(name="async-in-loop", warmup_iterations=0, min_iterations=1, repeats=1)

    @case.benchmark()
    async def bench() -> None:
        pass

    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            run_cases(list_cases())

    asyncio.run(invoke())


def test_memory_failure_restores_tracemalloc_state() -> None:
    clear_registry()
    tracemalloc.stop()
    case = Case(
        name="memory-failure",
        metric_type="memory",
        warmup_iterations=0,
        min_iterations=1,
        repeats=1,
    )

    @case.benchmark()
    def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_cases(list_cases())
    assert not tracemalloc.is_tracing()


def test_memory_run_preserves_preexisting_tracemalloc() -> None:
    clear_registry()
    tracemalloc.start()
    try:
        case = Case(
            name="memory-existing-trace",
            metric_type="memory",
            warmup_iterations=0,
            min_iterations=1,
            repeats=1,
        )

        @case.benchmark()
        def allocate() -> list[int]:
            return [1, 2, 3]

        run_cases(list_cases())
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_comparison_rejects_incompatible_environment_by_default() -> None:
    baseline = _run(python_version="3.12.9")
    current = _run(python_version="3.13.1")

    assert comparison_environment_mismatches(baseline, current) == {
        "python_version": ("3.12.9", "3.13.1")
    }
    with pytest.raises(ValueError, match="incompatible"):
        compare_runs(baseline, current)
    assert compare_runs(baseline, current, allow_environment_mismatch=True) == []


def test_comparison_reports_machine_mismatch() -> None:
    assert comparison_environment_mismatches(
        _run(machine="x86_64"), _run(machine="arm64")
    ) == {"machine": ("x86_64", "arm64")}


def test_write_json_rejects_non_standard_nan(tmp_path: Path) -> None:
    run = _run()
    run.benchmarks.append(
        BenchmarkResult(
            case_name="case",
            benchmark_name="bench",
            case_type="cpu",
            metric_type="time",
            gc_control="inherit",
            regression_threshold_pct=100.0,
            iterations=1,
            repeats=1,
            metrics={"mean_s": float("nan")},
        )
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(tmp_path / "invalid.json", run)


def test_write_markdown_escapes_table_cells(tmp_path: Path) -> None:
    run = _run()
    run.benchmarks.append(
        BenchmarkResult(
            case_name="case|name",
            benchmark_name="line\nbreak",
            case_type="cpu",
            metric_type="time",
            gc_control="inherit",
            regression_threshold_pct=100.0,
            iterations=1,
            repeats=1,
            metrics={"mean_s": 1.0},
        )
    )
    output = tmp_path / "results.md"
    write_markdown(output, run)
    rendered = output.read_text(encoding="utf-8")
    assert "case\\|name" in rendered
    assert "line<br>break" in rendered


def test_cli_rejects_empty_selection_without_writing_baseline(
    tmp_path: Path,
) -> None:
    module = tmp_path / "sample_benchmarks.py"
    module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='present', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark()\n"
        "def bench(): return 1\n",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert main([str(module), "--case", "missing"]) == 1
        assert not (tmp_path / ".benchbro").exists()
    finally:
        os.chdir(original_cwd)


def test_cli_rejects_directory_without_benchmarks(tmp_path: Path) -> None:
    assert main([str(tmp_path)]) == 1


def test_cli_rejects_invalid_iteration_flags() -> None:
    with pytest.raises(SystemExit):
        main(["--repeats", "0"])
    with pytest.raises(SystemExit):
        main(["--warmup", "-1"])
    with pytest.raises(SystemExit):
        main(["--min-iterations", "0"])


def test_ci_rejects_new_benchmark_until_baseline_is_replaced(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first_benchmarks.py"
    first.write_text(
        "from benchbro import Case\n"
        "case = Case(name='ci', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark(name='one')\n"
        "def one(): return 1\n",
        encoding="utf-8",
    )
    second = tmp_path / "second_benchmarks.py"
    second.write_text(
        first.read_text(encoding="utf-8")
        + "@case.benchmark(name='two')\n"
        + "def two(): return 2\n",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert main([str(first), "--ci", "--new-baseline"]) == 0
        baseline_path = tmp_path / ".benchbro" / "baseline.ci.json"
        before = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert main([str(second), "--ci"]) == 1
        after = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert before == after
    finally:
        os.chdir(original_cwd)


def test_cli_requires_override_for_environment_mismatch(tmp_path: Path) -> None:
    module = tmp_path / "environment_benchmarks.py"
    module.write_text(
        "from benchbro import Case\n"
        "case = Case(name='environment', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark()\n"
        "def bench(): return 1\n",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert main([str(module)]) == 0
        baseline_path = tmp_path / ".benchbro" / "baseline.local.json"
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        payload["python_version"] = "2.7.18"
        payload["environment"]["python_version"] = "2.7.18"
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")

        assert main([str(module)]) == 1
        assert main([str(module), "--allow-environment-mismatch"]) == 0
    finally:
        os.chdir(original_cwd)


def test_cli_can_run_same_system_module_twice(tmp_path: Path) -> None:
    module = tmp_path / "system_benchmarks.py"
    module.write_text(
        "from benchbro import Case, system\n"
        "@system()\n"
        "def value(): return 1\n"
        "case = Case(name='repeat', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark()\n"
        "def bench(value): return value\n",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert main([str(module)]) == 0
        assert main([str(module)]) == 0
    finally:
        os.chdir(original_cwd)


def test_python_m_benchbro_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "benchbro", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage: benchbro" in completed.stdout


def test_package_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchbro.cli.main", lambda argv: 7)
    assert benchbro.main(["--help"]) == 7
