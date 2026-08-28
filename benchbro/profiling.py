from __future__ import annotations

import cProfile
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ProfilerHook(Protocol):
    """Lifecycle implemented by built-in and user-registered profilers."""

    def start(self) -> None: ...

    def stop(self, output: Path) -> None: ...


class CProfileHook:
    def __init__(self) -> None:
        self._profiler = cProfile.Profile()

    def start(self) -> None:
        self._profiler.enable()

    def stop(self, output: Path) -> None:
        self._profiler.disable()
        output.parent.mkdir(parents=True, exist_ok=True)
        self._profiler.dump_stats(str(output))


class AllocationSnapshotHook:
    def __init__(self) -> None:
        self._started_here = False

    def start(self) -> None:
        self._started_here = not tracemalloc.is_tracing()
        if self._started_here:
            tracemalloc.start()

    def stop(self, output: Path) -> None:
        snapshot = tracemalloc.take_snapshot()
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{stat.traceback}: {stat.size} bytes in {stat.count} allocation(s)"
            for stat in snapshot.statistics("traceback")[:100]
        ]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self._started_here and tracemalloc.is_tracing():
            tracemalloc.stop()


_PROFILER_FACTORIES: dict[str, Callable[[], ProfilerHook]] = {
    "cprofile": CProfileHook,
    "tracemalloc": AllocationSnapshotHook,
}


def register_profiler(name: str, factory: Callable[[], ProfilerHook]) -> None:
    """Register a factory that creates a fresh profiler hook for each benchmark."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Profiler name must be a non-empty string")
    if name in _PROFILER_FACTORIES:
        raise ValueError(f"Profiler '{name}' is already registered")
    _PROFILER_FACTORIES[name] = factory


def create_profiler(name: str) -> ProfilerHook:
    try:
        return _PROFILER_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILER_FACTORIES))
        raise ValueError(
            f"Unknown profiler '{name}'. Registered profilers: {available}"
        ) from exc


def default_profile_path(
    profiler_name: str, case_name: str, benchmark_name: str
) -> Path:
    suffix = ".prof" if profiler_name == "cprofile" else ".txt"
    safe_name = "-".join(
        part.replace("/", "-").replace("\\", "-")
        for part in (case_name, benchmark_name)
    )
    return Path(".benchbro") / "profiles" / f"{safe_name}{suffix}"
