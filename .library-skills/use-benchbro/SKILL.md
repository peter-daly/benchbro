---
name: use-benchbro
description: Use BenchBro to design, write, configure, run, debug, review, and compare reliable Python benchmarks with Case, systems, parameters, subprocesses, adaptive measurement, baselines, history, profiling, or the pytest integration. Use whenever code imports benchbro or a task specifically uses the BenchBro library or CLI; do not use for unrelated benchmarking tools.
---

# Use BenchBro

Define performance experiments in Python, measure repeat-level samples, and compare
compatible runs without treating ordinary noise as a regression.

## Confirm the installed interface

Treat the installed BenchBro version as the source of truth. Inspect the CLI or a
public signature before using an unfamiliar option:

```bash
benchbro --help
benchbro run --help
python -c 'import inspect; from benchbro import Case; print(inspect.signature(Case))'
```

Import the supported public API from `benchbro`. Do not depend on implementation
modules unless extending BenchBro itself.

## Define the smallest useful experiment

Use one `Case` for benchmarks that share an input, metric type, tags, measurement
settings, and comparison policy:

```python
import hashlib

from benchbro import Case

case = Case(name="hashing", tags=["fast", "core"])


@case.input()
def payload() -> bytes:
    return b"benchbro"


@case.benchmark()
def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
```

Inputs are case-local values. Systems are reusable dependencies. Parameters,
inputs, and systems are injected by function parameter name, so names must match
exactly.

Keep the default `metric_type="time"` for latency or throughput questions. Use
`metric_type="memory"` for Python allocation questions. A memory benchmark measures
`tracemalloc` allocations, not whole-process resident memory.

## Parameterize comparisons

Put `parametrize` below `benchmark`, so the parameter metadata exists when the
outer benchmark decorator registers the callable:

```python
case = Case(name="encoding")


@case.benchmark()
@case.parametrize("size", [100, 10_000], ids=["small", "large"])
@case.parametrize("sort_keys", [False, True], ids=["unsorted", "sorted"])
def encode(size: int, sort_keys: bool) -> str:
    import json

    return json.dumps(list(range(size)), sort_keys=sort_keys)
```

Multiple declarations form a Cartesian product. Give values stable IDs because the
generated names are used in reports, history, and baseline matching.

## Run and discover benchmarks

With no target, BenchBro searches configured benchmark paths, which default to
`benchmarks/`:

```bash
benchbro run
benchbro list --verbose
benchbro run benchmarks/bench_encoding.py --case encoding --tag fast
```

A target can be a module, Python file, or directory. Prefer the command-oriented
interface; the legacy `benchbro TARGET` syntax remains compatible.

Store project-wide discovery and run defaults in `benchbro.toml` or
`[tool.benchbro]` in `pyproject.toml`. Command-line values take precedence over
configuration. Use `benchbro init` to create a starter configuration and benchmark
without overwriting existing files.

## Manage outputs and baselines deliberately

Regular runs create or compare `.benchbro/baseline.local.json`. Reports are opt-in:

```bash
benchbro run \
  --output-json artifacts/current.json \
  --output-csv artifacts/current.csv \
  --output-md artifacts/current.md
```

Use `--new-baseline` only when the current environment and behavior should replace
the selected reference. Named baselines separate machines, environments, or
branches. `--ci` selects the committed `.benchbro/baseline.ci.json` and requires it
to contain every selected benchmark.

Read [references/reliable-measurements.md](references/reliable-measurements.md)
completely before tuning adaptive sampling, comparison metrics, thresholds, CI
gates, or interpreting noisy, likely, inconclusive, and regression results.

## Use advanced features with their ownership rules

Read [references/advanced-patterns.md](references/advanced-patterns.md) completely
before implementing any of these:

- systems with scopes, teardown, dependencies, or async behavior;
- setup/teardown measurement boundaries;
- subprocess benchmarks or process isolation;
- timeouts, CPU affinity, or stabilization delays;
- built-in or custom profiler hooks;
- pytest integration or direct programmatic execution.

## Verify the benchmark workflow

Run focused checks that exercise the intended boundary:

- use `benchbro list --verbose` to confirm names, parameters, and tags;
- run once with `--no-compare` while shaping the experiment;
- inspect sample count, confidence bounds, relative margin, CV, outliers, and the
  noisy flag before choosing thresholds;
- rerun unchanged code to establish ordinary within-environment variance;
- compare only like-for-like Python, operating-system, and machine environments;
- exercise subprocess failure, timeout, and resource teardown paths explicitly;
- keep `.benchbro/` ignored except for an intentional CI baseline.

## Avoid common mistakes

- Do not benchmark setup accidentally; state whether setup and teardown belong in
  the measured interval.
- Do not use one repeat as evidence for a regression.
- Do not tighten thresholds before observing normal variance.
- Do not compare different Python minor versions, implementations, operating
  systems, or machine architectures without treating that difference as part of
  the experiment.
- Do not put `@case.parametrize(...)` above `@case.benchmark()`.
- Do not inject a system into an input or an input into a system.
- Do not pass shell command strings to `CommandSpec`; provide explicit arguments.
- Do not use process isolation for local or lambda benchmark functions that cannot
  be imported on spawn-based platforms.
- Do not assume profiling leaves benchmark timings comparable with unprofiled runs.
- Do not commit machine-local history, profiles, or baselines.
