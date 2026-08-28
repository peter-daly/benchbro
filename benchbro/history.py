from __future__ import annotations

from pathlib import Path

from benchbro.models import BenchmarkRun
from benchbro.serialization import read_json, write_json


def history_directory(repo_root: Path) -> Path:
    return repo_root / ".benchbro" / "history"


def save_history(repo_root: Path, run: BenchmarkRun) -> Path:
    directory = history_directory(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run.started_at.replace(":", "-").replace("+", "_")
    commit = run.environment.get("git_commit", "working-tree")[:12] or "working-tree"
    path = directory / f"{stamp}-{commit}-{run.run_id[:8]}.json"
    write_json(path, run)
    return path


def list_history(repo_root: Path) -> list[Path]:
    directory = history_directory(repo_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), reverse=True)


def load_history(repo_root: Path, selector: str) -> tuple[Path, BenchmarkRun]:
    candidates = list_history(repo_root)
    exact = Path(selector)
    if exact.is_file():
        return exact, read_json(exact)
    matches = [path for path in candidates if selector in path.name]
    if not matches:
        raise ValueError(f"No history entry matches '{selector}'")
    if len(matches) > 1:
        raise ValueError(
            f"History selector '{selector}' is ambiguous: "
            + ", ".join(path.name for path in matches[:5])
        )
    return matches[0], read_json(matches[0])
