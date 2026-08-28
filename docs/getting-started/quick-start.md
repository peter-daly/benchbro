# Quick start

## 1. Define a case

Create `benchmarks/bench_hashing.py`:

```python
import hashlib

from benchbro import Case, system


@system(scope="session")
def salt() -> bytes:
    return b"-fixture"


case = Case(
    name="hashing",
    case_type="cpu",
    metric_type="time",
    tags=["fast", "core"],
)


@case.input()
def payload() -> bytes:
    return b"benchbro"


@case.benchmark()
def sha256(payload: bytes, salt: bytes) -> str:
    return hashlib.sha256(payload + salt).hexdigest()
```

Inputs and systems are injected by parameter name. The input supplies data for the
case; the session-scoped system supplies one reusable dependency.

## 2. Run it

```console
$ uv run benchbro run
```

With no target, BenchBro searches `benchmarks/**/*.py`. You can also pass a module,
file, or directory explicitly:

```console
$ uv run benchbro run benchmarks/bench_hashing.py
$ uv run benchbro list benchmarks --verbose
```

The first normal run creates `.benchbro/baseline.local.json`. Later runs compare
against it automatically.

## 3. Save reports

Reports are opt-in:

```console
$ uv run benchbro run \
    --output-json artifacts/current.json \
    --output-csv artifacts/current.csv \
    --output-md artifacts/current.md
```

## 4. Tighten the experiment

Start with defaults, then tune measurement only when the benchmark calls for it:

```python
case = Case(
    name="hashing",
    adaptive=True,
    min_repeats=5,
    repeats=100,
    target_relative_margin_pct=2.0,
    max_time_s=10.0,
)
```

Adaptive sampling stops once the requested precision is reached, or when its time
or repeat budget is exhausted. Read [Reliable measurements](../guides/reliable-measurements.md)
before using a threshold as a performance gate.

## Where next?

- [Define parameterized time and memory benchmarks](../guides/defining-benchmarks.md)
- [Manage dependencies with systems and inputs](../guides/systems-and-inputs.md)
- [Create and compare baselines](../guides/baselines-and-history.md)
- [Configure discovery and defaults](../guides/configuration.md)
