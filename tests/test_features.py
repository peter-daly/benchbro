from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import benchbro.environment as environment_module
import benchbro.runner as runner_module
from benchbro import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSettings,
    Case,
    CommandSpec,
    clear_registry,
    compare_runs,
    list_cases,
    register_profiler,
    run_cases,
    system,
)
from benchbro.cli import EXIT_EXECUTION, EXIT_USAGE, main
from benchbro.config import load_config
from benchbro.history import list_history, load_history, save_history
from benchbro.measurement import (
    collect_timing_samples,
    confidence_interval_95,
    outlier_indexes,
    should_stop_adaptive,
    time_metrics,
)
from benchbro.models import (
    BenchmarkCase,
    BenchmarkRegistry,
    SystemDefinition,
    SystemScope,
    normalize_system_scope,
)
from benchbro.profiling import create_profiler, default_profile_path
from benchbro.pytest_plugin import PytestBenchmarkRunner, pytest_configure
from benchbro.serialization import migrate_payload, read_json, write_json


def _run_result(
    value: float,
    *,
    standard_error: float = 0.0,
    noisy: bool = False,
) -> BenchmarkRun:
    environment = {
        "python_version": "3.13.1",
        "python_implementation": "CPython",
        "system": "Darwin",
        "machine": "arm64",
    }
    return BenchmarkRun(
        started_at="start",
        finished_at="finish",
        python_version="3.13.1",
        platform="Darwin-arm64",
        environment=environment,
        benchmarks=[
            BenchmarkResult(
                case_name="case",
                benchmark_name="bench",
                case_type="cpu",
                metric_type="time",
                gc_control="inherit",
                regression_threshold_pct=10.0,
                warning_threshold_pct=5.0,
                iterations=1,
                repeats=20,
                metrics={
                    "median_s": value,
                    "mean_s": value,
                    "standard_error_s": standard_error,
                },
                is_noisy=noisy,
            )
        ],
    )


def _benchmark_module(path: Path, *, sleep_s: float = 0.0) -> Path:
    path.write_text(
        "from benchbro import Case\n"
        "import time\n"
        "case = Case(name='feature', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark()\n"
        "def bench():\n"
        f"    time.sleep({sleep_s})\n",
        encoding="utf-8",
    )
    return path


def test_parameterized_benchmarks_form_a_cartesian_product() -> None:
    clear_registry()
    seen: list[tuple[int, str]] = []
    case = Case(name="parameters")

    @case.benchmark(repeats=1, min_iterations=1, warmup_iterations=0)
    @case.parametrize("label", ["a", "b"])
    @case.parametrize("size", [1, 2], ids=["small", "large"])
    def bench(size: int, label: str) -> None:
        seen.append((size, label))

    registered = list_cases()
    assert len(registered) == 4
    assert {item.parameter_id for item in registered} == {
        "small-a",
        "small-b",
        "large-a",
        "large-b",
    }
    result = run_cases(registered)
    assert set(seen) == {(1, "a"), (1, "b"), (2, "a"), (2, "b")}
    assert all(item.parameters for item in result.benchmarks)


@pytest.mark.parametrize(
    ("name", "values", "ids", "message"),
    [
        ("", [1], None, "name"),
        ("size", [], None, "values"),
        ("size", [1, 2], ["one"], "ids"),
        ("size", [1], [""], "ids"),
    ],
)
def test_parameterization_validation(
    name: str,
    values: list[int],
    ids: list[str] | None,
    message: str,
) -> None:
    case = Case(name="invalid-parameter")
    with pytest.raises(ValueError, match=message):
        case.parametrize(name, values, ids=ids)


def test_duplicate_parameter_name_is_rejected() -> None:
    case = Case(name="duplicate-parameter")

    @case.parametrize("size", [1])
    def bench(size: int) -> None:
        pass

    with pytest.raises(ValueError, match="already defined"):
        case.parametrize("size", [2])(bench)


def test_iteration_benchmark_and_session_system_scopes() -> None:
    clear_registry()
    events: list[str] = []

    @system(scope="iteration")
    def each_iteration():
        events.append("iteration-setup")
        yield 1
        events.append("iteration-teardown")

    @system(scope="benchmark")
    def each_benchmark():
        events.append("benchmark-setup")
        yield 2
        events.append("benchmark-teardown")

    @system(scope="session")
    def once():
        events.append("session-setup")
        yield 3
        events.append("session-teardown")

    case = Case(name="scopes", repeats=2, min_iterations=2, warmup_iterations=0)

    @case.benchmark()
    def first(each_iteration: int, each_benchmark: int, once: int) -> None:
        assert each_iteration + each_benchmark + once == 6

    @case.benchmark()
    def second(each_iteration: int, each_benchmark: int, once: int) -> None:
        assert each_iteration + each_benchmark + once == 6

    run_cases(list_cases())
    assert events.count("iteration-setup") == 8
    assert events.count("iteration-teardown") == 8
    assert events.count("benchmark-setup") == 2
    assert events.count("benchmark-teardown") == 2
    assert events.count("session-setup") == 1
    assert events.count("session-teardown") == 1


def test_setup_and_teardown_can_be_excluded_from_timing() -> None:
    clear_registry()

    @system(scope="iteration")
    def slow_fixture():
        time.sleep(0.003)
        yield None
        time.sleep(0.003)

    included = Case(
        name="timing",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        setup_timing="include",
        teardown_timing="include",
    )
    excluded = Case(
        name="timing-excluded",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        setup_timing="exclude",
        teardown_timing="exclude",
    )

    @included.benchmark()
    def include(slow_fixture: None) -> None:
        pass

    @excluded.benchmark()
    def exclude(slow_fixture: None) -> None:
        pass

    results = run_cases(list_cases()).benchmarks
    assert results[0].metrics["median_s"] > results[1].metrics["median_s"] * 5


def test_adaptive_sampling_stops_when_precision_target_is_met() -> None:
    clear_registry()
    case = Case(
        name="adaptive",
        repeats=20,
        min_repeats=3,
        min_iterations=1,
        warmup_iterations=0,
        adaptive=True,
        target_relative_margin_pct=10_000,
    )

    @case.benchmark()
    def stable() -> None:
        pass

    result = run_cases(list_cases()).benchmarks[0]
    assert result.repeats == 3
    assert result.metrics["sample_count"] == 3
    assert "ci95_low_s" in result.metrics
    assert "variance_s2" in result.metrics
    assert "cv_pct" in result.metrics


def test_run_level_reliability_overrides_are_applied() -> None:
    clear_registry()
    case = Case(name="run-overrides", repeats=10, min_iterations=1, warmup_iterations=0)

    @case.benchmark()
    def stable() -> None:
        pass

    result = run_cases(
        list_cases(),
        adaptive=True,
        min_repeats=2,
        min_time_s=0,
        max_time_s=1,
        target_relative_margin_pct=10_000,
        noise_threshold_pct=10_000,
        stabilization_delay_s=0,
    ).benchmarks[0]
    assert result.repeats == 2
    assert not result.is_noisy


def test_measurement_helpers_cover_confidence_outliers_and_time_budget() -> None:
    assert confidence_interval_95([]) == (0.0, 0.0, 0.0)
    assert confidence_interval_95([2.0]) == (2.0, 2.0, 0.0)
    samples = [1.0, 1.0, 1.0, 100.0]
    assert outlier_indexes(samples) == {3}
    metrics, count, noisy = time_metrics(samples, noise_threshold_pct=1.0)
    assert count == 1
    assert noisy
    assert metrics["outlier_count"] == 1
    assert not should_stop_adaptive(
        [1.0],
        min_repeats=2,
        min_time_s=0,
        max_time_s=None,
        target_relative_margin_pct=1,
        elapsed_s=10,
    )
    assert should_stop_adaptive(
        [1.0, 2.0],
        min_repeats=2,
        min_time_s=0,
        max_time_s=1,
        target_relative_margin_pct=0,
        elapsed_s=2,
    )
    values = iter([1.0, 1.0, 1.0, 1.0])
    assert (
        len(
            collect_timing_samples(
                lambda: next(values),
                repeats=4,
                adaptive=True,
                min_repeats=2,
                min_time_s=0,
                max_time_s=None,
                target_relative_margin_pct=0,
            )
        )
        == 2
    )


def test_confidence_aware_comparison_distinguishes_inconclusive_regression() -> None:
    inconclusive = compare_runs(
        _run_result(1.0, standard_error=0.2),
        _run_result(1.2, standard_error=0.2),
    )[0]
    assert inconclusive.status == "inconclusive"
    assert not inconclusive.is_regression
    confirmed = compare_runs(
        _run_result(1.0, standard_error=0.001),
        _run_result(1.2, standard_error=0.001),
    )[0]
    assert confirmed.status == "regression"
    assert confirmed.is_regression

    one_sample_baseline = _run_result(1.0)
    one_sample_current = _run_result(1.2)
    one_sample_baseline.benchmarks[0].metrics["sample_count"] = 1
    one_sample_current.benchmarks[0].metrics["sample_count"] = 1
    one_sample = compare_runs(one_sample_baseline, one_sample_current)[0]
    assert one_sample.status == "inconclusive"
    assert one_sample.confidence_pct == 0.0


def test_comparison_reports_noisy_and_improved_results() -> None:
    noisy = compare_runs(_run_result(1.0), _run_result(1.0, noisy=True))[0]
    assert noisy.status == "noisy"
    improved = compare_runs(_run_result(1.0), _run_result(0.8))[0]
    assert improved.status == "improvement"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="The isolation test uses fork-local benchmark functions",
)
def test_process_isolation_runs_benchmark_in_a_child_process(tmp_path: Path) -> None:
    clear_registry()
    marker = tmp_path / "pid.txt"
    case = Case(
        name="isolated",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        isolation="process",
    )

    @case.benchmark()
    def isolated() -> None:
        marker.write_text(str(os.getpid()), encoding="utf-8")

    result = run_cases(list_cases())
    assert int(marker.read_text(encoding="utf-8")) != os.getpid()
    assert len(result.benchmarks) == 1


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="The isolation test uses fork-local benchmark functions",
)
def test_process_isolation_propagates_worker_errors() -> None:
    clear_registry()
    case = Case(
        name="isolated-error",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        isolation="process",
    )

    @case.benchmark()
    def fail() -> None:
        raise LookupError("child boom")

    with pytest.raises(RuntimeError, match="LookupError.*child boom"):
        run_cases(list_cases())


def test_isolated_worker_protocol_reports_success_and_error() -> None:
    class ResultQueue:
        def __init__(self) -> None:
            self.messages: list[tuple[Any, ...]] = []

        def put(self, message: tuple[Any, ...]) -> None:
            self.messages.append(message)

    success_queue = ResultQueue()
    success = BenchmarkCase(
        case_name="worker",
        benchmark_name="success",
        func=lambda: None,
        settings=BenchmarkSettings(
            repeats=1, min_repeats=1, min_iterations=1, warmup_iterations=0
        ),
    )
    runner_module._isolated_worker(success_queue, success, {})
    assert success_queue.messages[0][0] == "ok"

    error_queue = ResultQueue()

    def fail() -> None:
        raise RuntimeError("worker error")

    failure = replace(success, benchmark_name="failure", func=fail)
    runner_module._isolated_worker(error_queue, failure, {})
    assert error_queue.messages[0][0:3] == (
        "error",
        "RuntimeError",
        "worker error",
    )


def test_in_process_timeout_is_reported() -> None:
    clear_registry()
    case = Case(
        name="timeout",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        timeout_s=0.001,
    )

    @case.benchmark()
    def slow() -> None:
        time.sleep(0.003)

    with pytest.raises(RuntimeError, match="in-process timeout"):
        run_cases(list_cases())


def test_cprofile_and_allocation_profile_hooks_write_artifacts(tmp_path: Path) -> None:
    clear_registry()
    profile = tmp_path / "result.prof"
    allocations = tmp_path / "allocations.txt"
    cpu = Case(
        name="profiles",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        profiler="cprofile",
        profile_output=str(profile),
    )
    memory = Case(
        name="allocation-profile",
        repeats=1,
        min_iterations=1,
        warmup_iterations=0,
        profiler="tracemalloc",
        profile_output=str(allocations),
    )

    @cpu.benchmark()
    def profiled() -> int:
        return sum(range(10))

    @memory.benchmark()
    def allocated() -> list[int]:
        return list(range(10))

    results = run_cases(list_cases()).benchmarks
    assert profile.is_file() and profile.stat().st_size > 0
    assert allocations.is_file()
    assert results[0].profile_path == str(profile)


def test_custom_profiler_registration_and_validation(tmp_path: Path) -> None:
    output = tmp_path / "custom.txt"

    class Hook:
        def start(self) -> None:
            output.write_text("started", encoding="utf-8")

        def stop(self, output: Path) -> None:
            output.write_text("stopped", encoding="utf-8")

    register_profiler("test-custom-profiler", Hook)
    hook = create_profiler("test-custom-profiler")
    hook.start()
    hook.stop(output)
    assert output.read_text(encoding="utf-8") == "stopped"
    with pytest.raises(ValueError, match="already registered"):
        register_profiler("test-custom-profiler", Hook)
    with pytest.raises(ValueError, match="Unknown profiler"):
        create_profiler("missing-profiler")
    with pytest.raises(ValueError, match="non-empty"):
        register_profiler("", Hook)
    assert default_profile_path("cprofile", "a/b", "c\\d").suffix == ".prof"


def test_schema_migration_and_future_schema_rejection(tmp_path: Path) -> None:
    old = {
        "started_at": "s",
        "finished_at": "f",
        "python_version": "3.10",
        "platform": "test",
        "environment": {},
        "benchmarks": [],
    }
    migrated = migrate_payload(old)
    assert migrated["schema_version"] == 2
    assert migrated["run_id"] == ""
    with pytest.raises(ValueError, match="Invalid"):
        migrate_payload({"schema_version": 0})
    with pytest.raises(ValueError, match="Unsupported"):
        migrate_payload({"schema_version": 999})
    path = tmp_path / "run.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    assert read_json(path).schema_version == 2


def test_history_save_list_load_and_ambiguity(tmp_path: Path) -> None:
    assert list_history(tmp_path) == []
    run = _run_result(1.0)
    run.run_id = "1234567890"
    run.environment["git_commit"] = "abcdef1234567890"
    first = save_history(tmp_path, run)
    assert list_history(tmp_path) == [first]
    selected_path, selected = load_history(tmp_path, "abcdef")
    assert selected_path == first
    assert selected.run_id == run.run_id
    assert load_history(tmp_path, str(first))[0] == first
    second = first.with_name("another-abcdef.json")
    write_json(second, run)
    with pytest.raises(ValueError, match="ambiguous"):
        load_history(tmp_path, "abcdef")
    with pytest.raises(ValueError, match="No history"):
        load_history(tmp_path, "missing")


def test_config_supports_pyproject_and_standalone_precedence(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.benchbro.ini_options]\nbenchmark_paths = ['old']\n"
        "[tool.benchbro.run]\nrepeats = 5\n",
        encoding="utf-8",
    )
    (tmp_path / "benchbro.toml").write_text(
        "benchmark_paths = ['benchmarks']\nbaseline = 'team'\nsave_history = true\n"
        "[run]\nrepeats = 3\nwarmup = 0\nmin_iterations = 2\nadaptive = true\n"
        "min_repeats = 2\nmin_time_s = 0.1\nmax_time_s = 1.5\n"
        "target_relative_margin_pct = 3.0\nnoise_threshold_pct = 9.0\n"
        "stabilization_delay_s = 0.2\nisolation = 'process'\ncpu_affinity = [0]\n"
        "[output]\njson = 'out.json'\ncsv = 'out.csv'\nmarkdown = 'out.md'\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.benchmark_paths == ("benchmarks",)
    assert config.repeats == 3
    assert config.warmup == 0
    assert config.adaptive is True
    assert config.min_repeats == 2
    assert config.min_time_s == 0.1
    assert config.target_relative_margin_pct == 3.0
    assert config.noise_threshold_pct == 9.0
    assert config.stabilization_delay_s == 0.2
    assert config.isolation == "process"
    assert config.cpu_affinity == (0,)
    assert config.baseline == "team"
    assert config.output_md == "out.md"


@pytest.mark.parametrize(
    "contents",
    [
        "benchmark_paths = []",
        "[run]\nrepeats = 0",
        "[run]\nwarmup = -1",
        "[run]\nmax_time_s = 0",
        "[run]\nmin_time_s = -1",
        "[run]\ntarget_relative_margin_pct = -1",
        "[run]\nadaptive = 'yes'",
        "[run]\nisolation = 'container'",
        "[run]\ncpu_affinity = [-1]",
        "baseline = ''",
        "save_history = 'yes'",
        "run = 1",
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path, str(path))


def test_config_rejects_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_config(tmp_path, str(tmp_path / "missing.toml"))


def test_environment_metadata_records_repository_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")

    def command(argv: list[str], cwd: Path | None = None) -> str:
        del cwd
        if argv[-1] == "--show-toplevel":
            return str(tmp_path)
        if argv[-1] == "HEAD":
            return "abc123"
        if argv[-1] == "--show-current":
            return "main"
        if argv[-1] == "--porcelain":
            return " M file"
        return ""

    monkeypatch.setattr(environment_module, "_command_output", command)
    metadata = environment_module.collect_environment_metadata(tmp_path)
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_dirty"] == "True"
    assert metadata["dependency_lock"] == "uv.lock"
    assert len(metadata["dependency_lock_sha256"]) == 64


def test_command_output_handles_missing_executable() -> None:
    assert environment_module._command_output(["definitely-not-a-command"]) == ""


def test_cpu_and_power_metadata_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment_module.platform, "processor", lambda: "")
    monkeypatch.setattr(environment_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        environment_module,
        "_command_output",
        lambda argv, cwd=None: (
            "Example CPU"
            if any("brand_string" in item for item in argv)
            else "Now drawing from 'Battery Power'"
        ),
    )
    assert environment_module._cpu_model() == "Example CPU"
    assert environment_module._power_source() == "battery"
    monkeypatch.setattr(
        environment_module,
        "_command_output",
        lambda argv, cwd=None: "Now drawing from 'AC Power'",
    )
    assert environment_module._power_source() == "ac"


def test_model_and_registry_validation_branches() -> None:
    scope_cases: list[tuple[SystemScope, str]] = [
        ("function", "benchmark"),
        ("module", "session"),
        ("iteration", "iteration"),
    ]
    for scope, canonical in scope_cases:
        assert normalize_system_scope(scope) == canonical
    with pytest.raises(ValueError, match="Invalid system scope"):
        normalize_system_scope("bad")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="min_repeats"):
        BenchmarkSettings(repeats=1, min_repeats=2)
    with pytest.raises(ValueError, match="max_time"):
        BenchmarkSettings(min_time_s=2, max_time_s=1)
    with pytest.raises(ValueError, match="cpu_affinity"):
        BenchmarkSettings(cpu_affinity=())
    with pytest.raises(ValueError, match="setup_timing"):
        BenchmarkSettings(setup_timing="bad")  # ty: ignore[invalid-argument-type]
    registry = BenchmarkRegistry()
    case = BenchmarkCase(case_name="x", benchmark_name="y", func=lambda: None)
    assert registry.register(case) is case
    assert registry.all_cases() == [case]
    registry.clear()
    assert registry.all_cases() == []
    SystemDefinition(name="fixture", func=lambda: None, scope="session")
    with pytest.raises(ValueError):
        CommandSpec(argv=[1])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        CommandSpec(argv=["x"], env={"x": 1})  # ty: ignore[invalid-argument-type]


def test_pytest_runner_reuses_fixture_values() -> None:
    runner = PytestBenchmarkRunner()
    result = runner(
        lambda value: value + 1,
        2,
        repeats=1,
        warmup=0,
        min_iterations=1,
        name="pytest-case",
    )
    assert result.case_name == "pytest"
    assert result.benchmark_name == "pytest-case"


def test_pytest_plugin_registers_marker() -> None:
    values: list[tuple[str, str]] = []

    class Config:
        def addinivalue_line(self, name: str, value: str) -> None:
            values.append((name, value))

    pytest_configure(Config())  # ty: ignore[invalid-argument-type]
    assert values[0][0] == "markers"


def test_cli_subcommands_config_history_and_named_baselines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    assert main(["init", str(project)]) == EXIT_USAGE
    assert main(["init", str(project), "--force"]) == 0
    benchmark = project / "benchmarks" / "bench_example.py"
    original = Path.cwd()
    try:
        os.chdir(project)
        assert main(["list", str(benchmark), "--verbose"]) == 0
        assert (
            main(
                [
                    "run",
                    str(benchmark),
                    "--repeats",
                    "1",
                    "--warmup",
                    "0",
                    "--min-iterations",
                    "1",
                    "--baseline",
                    "team",
                    "--save-history",
                ]
            )
            == 0
        )
        baseline = project / ".benchbro" / "baselines" / "team.json"
        assert baseline.is_file()
        assert main(["baseline", "list"]) == 0
        assert main(["compare", str(baseline), str(baseline)]) == 0
        assert main(["history", "list", "--limit", "1"]) == 0
        history = list_history(project)
        assert history
        selector = history[0].name[-20:-5]
        assert main(["history", "show", selector]) == 0
        assert main(["history", "compare", selector, selector]) == 0
        assert (
            main(
                [
                    "baseline",
                    "update",
                    str(benchmark),
                    "--baseline",
                    "release",
                    "--repeats",
                    "1",
                    "--warmup",
                    "0",
                    "--min-iterations",
                    "1",
                ]
            )
            == 0
        )
        assert (project / ".benchbro" / "baselines" / "release.json").is_file()
    finally:
        os.chdir(original)
    assert "Discovered Benchmarks" in capsys.readouterr().out


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_cli_prints_shell_completion(
    shell: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["completion", shell]) == 0
    assert "benchbro" in capsys.readouterr().out


def test_cli_reports_execution_and_configuration_failures(tmp_path: Path) -> None:
    failing = tmp_path / "bench_failing.py"
    failing.write_text(
        "from benchbro import Case\n"
        "case = Case(name='fail', repeats=1, min_iterations=1, warmup_iterations=0)\n"
        "@case.benchmark()\n"
        "def bench(): raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    assert main(["run", str(failing)]) == EXIT_EXECUTION
    good = _benchmark_module(tmp_path / "bench_good.py")
    assert main(["run", str(good), "--baseline", "../unsafe"]) == EXIT_USAGE
    assert main(["list", str(good), "--case", "missing"]) == EXIT_USAGE
    with pytest.raises(SystemExit):
        main(["run", str(good), "--max-time", "0"])
    with pytest.raises(SystemExit):
        main(["run", str(good), "--cpu-affinity", "-1"])
    with pytest.raises(SystemExit):
        main(["run", str(good), "--min-time", "-1"])


def test_cli_uses_standalone_config_for_default_target(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    _benchmark_module(tmp_path / "benchmarks" / "bench_configured.py")
    (tmp_path / "benchbro.toml").write_text(
        "benchmark_paths = ['benchmarks']\nbaseline = 'configured'\n"
        "[run]\nrepeats = 1\nwarmup = 0\nmin_iterations = 1\n",
        encoding="utf-8",
    )
    original = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert main(["run"]) == 0
        assert (tmp_path / ".benchbro" / "baselines" / "configured.json").is_file()
    finally:
        os.chdir(original)


def test_cpu_affinity_reports_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(runner_module.os, "sched_getaffinity", raising=False)
    with (
        pytest.raises(RuntimeError, match="not supported"),
        runner_module._cpu_affinity((0,)),
    ):
        pass


def test_cpu_affinity_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[set[int]] = []
    monkeypatch.setattr(
        runner_module.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    monkeypatch.setattr(
        runner_module.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append(set(cpus)),
        raising=False,
    )
    with runner_module._cpu_affinity((1,)):
        assert calls == [{1}]
    assert calls == [{1}, {0, 1}]
