from __future__ import annotations

import asyncio
import gc
import math
import multiprocessing
import os
import queue
import statistics
import time
import traceback
import tracemalloc
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from benchbro.environment import collect_environment_metadata
from benchbro.lifecycle import (
    InvocationContext,
    ScopeState,
    execute_iteration,
    finalize_scope,
    has_running_event_loop,
)
from benchbro.measurement import collect_timing_samples, time_metrics
from benchbro.models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSettings,
    IsolationMode,
    SystemDefinition,
)
from benchbro.profiling import create_profiler, default_profile_path


def _with_overrides(
    original: BenchmarkCase,
    repeats: int | None,
    warmup: int | None,
    min_iterations: int | None,
    adaptive: bool | None,
    min_repeats: int | None,
    min_time_s: float | None,
    max_time_s: float | None,
    target_relative_margin_pct: float | None,
    noise_threshold_pct: float | None,
    isolation: IsolationMode | None,
    cpu_affinity: tuple[int, ...] | None,
    stabilization_delay_s: float | None,
) -> BenchmarkCase:
    source = original.settings
    settings = BenchmarkSettings(
        warmup_iterations=(warmup if warmup is not None else source.warmup_iterations),
        min_iterations=(
            min_iterations if min_iterations is not None else source.min_iterations
        ),
        repeats=repeats if repeats is not None else source.repeats,
        gc_control=source.gc_control,
        adaptive=adaptive if adaptive is not None else source.adaptive,
        min_repeats=min(
            min_repeats if min_repeats is not None else source.min_repeats,
            repeats if repeats is not None else source.repeats,
        ),
        min_time_s=min_time_s if min_time_s is not None else source.min_time_s,
        max_time_s=max_time_s if max_time_s is not None else source.max_time_s,
        target_relative_margin_pct=(
            target_relative_margin_pct
            if target_relative_margin_pct is not None
            else source.target_relative_margin_pct
        ),
        noise_threshold_pct=(
            noise_threshold_pct
            if noise_threshold_pct is not None
            else source.noise_threshold_pct
        ),
        setup_timing=source.setup_timing,
        teardown_timing=source.teardown_timing,
        timeout_s=source.timeout_s,
        isolation=isolation if isolation is not None else source.isolation,
        cpu_affinity=(
            cpu_affinity if cpu_affinity is not None else source.cpu_affinity
        ),
        stabilization_delay_s=(
            stabilization_delay_s
            if stabilization_delay_s is not None
            else source.stabilization_delay_s
        ),
        profiler=source.profiler,
        profile_output=source.profile_output,
    )
    return replace(original, settings=settings)


@contextmanager
def _gc_control(mode: str) -> Iterator[None]:
    if mode == "inherit":
        yield
        return
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@contextmanager
def _cpu_affinity(cpus: tuple[int, ...] | None) -> Iterator[None]:
    if cpus is None:
        yield
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this operating system")
    original = os.sched_getaffinity(0)  # type: ignore[attr-defined]
    try:
        os.sched_setaffinity(0, set(cpus))  # type: ignore[attr-defined]
        yield
    finally:
        os.sched_setaffinity(0, original)  # type: ignore[attr-defined]


def _measure_time(
    case: BenchmarkCase, context: InvocationContext
) -> tuple[dict[str, float], list[float], int, bool]:
    def measure_repeat() -> float:
        total = sum(
            execute_iteration(case, context)
            for _ in range(case.settings.min_iterations)
        )
        return total / case.settings.min_iterations

    samples = collect_timing_samples(
        measure_repeat,
        repeats=case.settings.repeats,
        adaptive=case.settings.adaptive,
        min_repeats=case.settings.min_repeats,
        min_time_s=case.settings.min_time_s,
        max_time_s=case.settings.max_time_s,
        target_relative_margin_pct=case.settings.target_relative_margin_pct,
    )
    metrics, outliers, noisy = time_metrics(samples, case.settings.noise_threshold_pct)
    return metrics, samples, outliers, noisy


def _measure_memory(
    case: BenchmarkCase, context: InvocationContext
) -> tuple[dict[str, float], int, bool]:
    peaks: list[float] = []
    nets: list[float] = []
    for _ in range(case.settings.repeats):
        started_here = not tracemalloc.is_tracing()
        if started_here:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()
        try:
            before_current, _ = tracemalloc.get_traced_memory()
            for _ in range(case.settings.min_iterations):
                execute_iteration(case, context)
            after_current, peak = tracemalloc.get_traced_memory()
        finally:
            if started_here and tracemalloc.is_tracing():
                tracemalloc.stop()
        peaks.append(float(peak))
        nets.append(float(after_current - before_current))
    peak_mean = statistics.fmean(peaks)
    peak_stddev = statistics.pstdev(peaks) if len(peaks) > 1 else 0.0
    peak_standard_error = (
        statistics.stdev(peaks) / math.sqrt(len(peaks)) if len(peaks) > 1 else 0.0
    )
    net_stddev = statistics.pstdev(nets) if len(nets) > 1 else 0.0
    net_standard_error = (
        statistics.stdev(nets) / math.sqrt(len(nets)) if len(nets) > 1 else 0.0
    )
    cv_pct = (peak_stddev / peak_mean * 100.0) if peak_mean > 0 else 0.0
    noisy = cv_pct > case.settings.noise_threshold_pct
    return (
        {
            "peak_alloc_bytes": peak_mean,
            "net_alloc_bytes": statistics.fmean(nets),
            "peak_alloc_bytes_max": max(peaks),
            "peak_alloc_stddev_bytes": peak_stddev,
            "peak_alloc_standard_error_bytes": peak_standard_error,
            "peak_alloc_ci95_low_bytes": peak_mean - (1.96 * peak_standard_error),
            "peak_alloc_ci95_high_bytes": peak_mean + (1.96 * peak_standard_error),
            "net_alloc_stddev_bytes": net_stddev,
            "net_alloc_standard_error_bytes": net_standard_error,
            "cv_pct": cv_pct,
            "sample_count": float(len(peaks)),
        },
        0,
        noisy,
    )


def _profile_path(case: BenchmarkCase) -> Path | None:
    profiler = case.settings.profiler
    if profiler is None:
        return None
    if case.settings.profile_output is None:
        return default_profile_path(
            profiler, case.case_name, case.benchmark_name
        ).resolve()
    rendered = case.settings.profile_output.format(
        case=case.case_name, benchmark=case.benchmark_name
    )
    return Path(rendered).resolve()


def _run_one_case(
    case: BenchmarkCase,
    *,
    systems: dict[str, SystemDefinition],
    session_state: ScopeState,
    event_loop: asyncio.AbstractEventLoop,
    environment: dict[str, str],
) -> BenchmarkResult:
    benchmark_state = ScopeState()
    context = InvocationContext(
        case=case,
        systems=systems,
        session_state=session_state,
        benchmark_state=benchmark_state,
        event_loop=event_loop,
    )
    started = perf_counter()
    profile_path = _profile_path(case)
    profiler_name = case.settings.profiler
    profiler = (
        create_profiler(profiler_name)
        if profile_path is not None and profiler_name is not None
        else None
    )
    try:
        with _cpu_affinity(case.settings.cpu_affinity):
            for _ in range(case.settings.warmup_iterations):
                execute_iteration(case, context)
            if case.settings.stabilization_delay_s:
                time.sleep(case.settings.stabilization_delay_s)
            if profiler is not None:
                profiler.start()
            try:
                with _gc_control(case.settings.gc_control):
                    if case.metric_type == "time":
                        metrics, samples, outliers, noisy = _measure_time(case, context)
                    else:
                        metrics, outliers, noisy = _measure_memory(case, context)
                        samples = []
            finally:
                if profiler is not None and profile_path is not None:
                    profiler.stop(profile_path)
    finally:
        finalize_scope(benchmark_state, event_loop)
    actual_repeats = int(metrics.get("sample_count", case.settings.repeats))
    return BenchmarkResult(
        case_name=case.case_name,
        benchmark_name=case.benchmark_name,
        case_type=case.case_type,
        metric_type=case.metric_type,
        gc_control=case.settings.gc_control,
        comparison_metric=case.comparison_metric,
        regression_threshold_pct=case.regression_threshold_pct,
        environment=environment.copy(),
        iterations=case.settings.min_iterations,
        repeats=actual_repeats,
        metrics=metrics,
        warning_threshold_pct=case.warning_threshold_pct,
        time_samples_s=samples,
        parameters={key: repr(value) for key, value in case.parameters.items()},
        parameter_id=case.parameter_id,
        is_noisy=noisy,
        outlier_count=outliers,
        profile_path=str(profile_path) if profile_path else None,
        duration_s=perf_counter() - started,
    )


def _isolated_worker(
    result_queue: Any,
    case: BenchmarkCase,
    systems: dict[str, SystemDefinition],
) -> None:
    event_loop = asyncio.new_event_loop()
    state = ScopeState()
    environment = collect_environment_metadata()
    local_case = replace(case, settings=replace(case.settings, isolation="in_process"))
    try:
        result = _run_one_case(
            local_case,
            systems=systems,
            session_state=state,
            event_loop=event_loop,
            environment=environment,
        )
        result_queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            ("error", type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        try:
            finalize_scope(state, event_loop)
        finally:
            if not has_running_event_loop():
                event_loop.run_until_complete(event_loop.shutdown_asyncgens())
            event_loop.close()


def _run_isolated_case(
    case: BenchmarkCase, systems: dict[str, SystemDefinition]
) -> BenchmarkResult:
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    result_queue = context.Queue()
    process = context.Process(  # ty: ignore[unresolved-attribute]
        target=_isolated_worker, args=(result_queue, case, systems)
    )
    process.start()
    timeout = None
    if case.settings.timeout_s is not None:
        invocation_count = case.settings.warmup_iterations + (
            case.settings.repeats * case.settings.min_iterations
        )
        timeout = (case.settings.timeout_s * max(invocation_count, 1)) + 5.0
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise RuntimeError(
            f"Isolated benchmark '{case.benchmark_name}' exceeded its process timeout"
        )
    try:
        message = result_queue.get(timeout=1)
    except queue.Empty as exc:
        raise RuntimeError(
            f"Isolated benchmark '{case.benchmark_name}' exited without a result "
            f"(exit code {process.exitcode})"
        ) from exc
    if message[0] == "error":
        _, error_type, error_message, child_traceback = message
        raise RuntimeError(
            f"Isolated benchmark failed with {error_type}: {error_message}\n"
            f"{child_traceback}"
        )
    return message[1]


def run_cases(
    cases: list[BenchmarkCase],
    repeats: int | None = None,
    warmup: int | None = None,
    min_iterations: int | None = None,
    *,
    adaptive: bool | None = None,
    min_repeats: int | None = None,
    min_time_s: float | None = None,
    max_time_s: float | None = None,
    target_relative_margin_pct: float | None = None,
    noise_threshold_pct: float | None = None,
    isolation: IsolationMode | None = None,
    cpu_affinity: tuple[int, ...] | None = None,
    stabilization_delay_s: float | None = None,
) -> BenchmarkRun:
    """Execute registered cases and return a complete benchmark run.

    Explicit function arguments override the settings stored on each case. Scoped
    systems and the shared async event loop are finalized even when execution
    fails.

    Args:
        cases: Concrete benchmark cases to execute in order.
        repeats: Fixed repeat count or adaptive maximum.
        warmup: Unmeasured invocations before sampling.
        min_iterations: Invocations combined into each repeat sample.
        adaptive: Enable precision-aware early stopping.
        min_repeats: Minimum samples collected in adaptive mode.
        min_time_s: Minimum measurement duration.
        max_time_s: Maximum measurement duration.
        target_relative_margin_pct: Target relative margin for the 95% interval.
        noise_threshold_pct: Coefficient-of-variation threshold for noisy results.
        isolation: Override in-process or worker-process execution.
        cpu_affinity: Linux CPU identifiers to use during measurement.
        stabilization_delay_s: Delay before benchmark measurement begins.

    Returns:
        The environment metadata and result for every executed benchmark.
    """

    from benchbro.api import get_system_registry

    started = datetime.now(timezone.utc).isoformat()
    environment = collect_environment_metadata()
    systems = get_system_registry().all()
    session_state = ScopeState()
    event_loop = asyncio.new_event_loop()
    results: list[BenchmarkResult] = []
    try:
        for original in cases:
            case = _with_overrides(
                original,
                repeats,
                warmup,
                min_iterations,
                adaptive,
                min_repeats,
                min_time_s,
                max_time_s,
                target_relative_margin_pct,
                noise_threshold_pct,
                isolation,
                cpu_affinity,
                stabilization_delay_s,
            )
            if case.settings.isolation == "process":
                results.append(_run_isolated_case(case, systems))
            else:
                results.append(
                    _run_one_case(
                        case,
                        systems=systems,
                        session_state=session_state,
                        event_loop=event_loop,
                        environment=environment,
                    )
                )
    finally:
        try:
            finalize_scope(session_state, event_loop)
        finally:
            try:
                if not has_running_event_loop():
                    event_loop.run_until_complete(event_loop.shutdown_asyncgens())
            finally:
                event_loop.close()
    finished = datetime.now(timezone.utc).isoformat()
    return BenchmarkRun(
        started_at=started,
        finished_at=finished,
        python_version=environment["python_version"],
        platform=environment["platform"],
        environment=environment,
        benchmarks=results,
        run_id=uuid.uuid4().hex,
    )
