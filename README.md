# benchbro

`benchbro` is a Python benchmarking library and CLI with pytest-style discovery and rich terminal output.

![Bench Bro Mascot](docs/assets/images/mascot-full-dark.png)

## Quick start

Install dependencies:

```bash
uv sync --group dev
```

Create benchmark cases in any importable module:

```python
from benchbro import Case

case = Case(name="hashing", case_type="cpu", metric_type="time", tags=["fast", "core"])


@case.input()
def payload() -> bytes:
    return b"benchbro"


@case.benchmark()
def sha1(payload: bytes) -> str:
    import hashlib

    return hashlib.sha1(payload).hexdigest()


@case.benchmark()
def sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
```

Regression tolerance defaults to `100.0` percent at the case level and can be overridden per benchmark:

```python
case = Case(name="hashing", regression_threshold_pct=10.0)

@case.benchmark(regression_threshold_pct=3.0)
def critical_path(payload: bytes) -> str:
    ...
```

GC is disabled during measured iterations by default. To keep the interpreter GC behavior unchanged, set:

```python
Case(name="hashing", gc_control="inherit")
```

Run benchmarks:

```bash
uv run benchbro --repeats 10 --warmup 2
```

When no target is provided, `benchbro` discovers benchmarks from:
- `benchmarks/**/*.py` (relative to repo root)

`benchbro` compares against the baseline by default (`.benchbro/baseline.json`).
If the baseline is missing, benchbro creates it automatically.
If new cases/benchmarks are introduced later, missing entries are merged into baseline.
Pass `--new-baseline` to replace the entire baseline with the current run.

By default, regular runs do not write artifacts.
Use explicit output flags (`--output-json`, `--output-csv`, `--output-md`) when needed.

The baseline is always written to:
- `.benchbro/baseline.json`

If requested, markdown output can also be written with `--output-md`.

JSON artifacts include environment metadata for reproducibility (Python/runtime/platform/CPU fields) both at run level and on each benchmark entry.

## CLI basics

Run selected cases/tags and write outputs:

```bash
uv run benchbro my_benchmarks.py \
  --case hashing \
  --tag fast \
  --output-json artifacts/current.json \
  --output-csv artifacts/current.csv \
  --output-md artifacts/current.md
```

Compare against baseline:

```bash
uv run benchbro my_benchmarks.py
```

Regression status uses each benchmark's effective threshold (`benchmark override -> case threshold -> 100% default`), and the comparison table shows the threshold for each row.

## End-to-end example

For a complete runnable workflow (baseline + candidate comparison), use:

- `examples/README.md`
- `make examples`
