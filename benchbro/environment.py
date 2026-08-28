from __future__ import annotations

import gc
import hashlib
import os
import platform
import socket
import subprocess
from pathlib import Path


def _command_output(argv: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _repository_metadata(start: Path) -> dict[str, str]:
    root_text = _command_output(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if not root_text:
        return {}
    root = Path(root_text)
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _command_output(["git", "branch", "--show-current"], cwd=root)
    dirty = bool(_command_output(["git", "status", "--porcelain"], cwd=root))
    metadata = {
        "git_root": str(root),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": str(dirty),
    }
    for lock_name in (
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "requirements.lock",
        "requirements.txt",
    ):
        lock_path = root / lock_name
        if lock_path.is_file():
            metadata["dependency_lock"] = lock_name
            metadata["dependency_lock_sha256"] = hashlib.sha256(
                lock_path.read_bytes()
            ).hexdigest()
            break
    return metadata


def _cpu_model() -> str:
    model = platform.processor().strip()
    if model:
        return model
    if platform.system() == "Darwin":
        return _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.partition(":")[2].strip()
    return ""


def _power_source() -> str:
    if platform.system() != "Darwin":
        return os.environ.get("BENCHBRO_POWER_SOURCE", "unknown")
    output = _command_output(["pmset", "-g", "batt"])
    if "AC Power" in output:
        return "ac"
    if "Battery Power" in output:
        return "battery"
    return "unknown"


def collect_environment_metadata(cwd: Path | None = None) -> dict[str, str]:
    uname = platform.uname()
    start = (cwd or Path.cwd()).resolve()
    runner_name = (
        os.environ.get("RUNNER_NAME")
        or os.environ.get("CI_RUNNER_DESCRIPTION")
        or socket.gethostname()
    )
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": _cpu_model(),
        "cpu_count": str(os.cpu_count() or 0),
        "power_source": _power_source(),
        "runner_identity": runner_name,
        "ci": str(bool(os.environ.get("CI"))),
        "gc_enabled_at_start": str(gc.isenabled()),
        "gc_threshold": ",".join(str(value) for value in gc.get_threshold()),
    }
    metadata.update(_repository_metadata(start))
    return metadata
