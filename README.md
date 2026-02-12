# bench-bro

`bench-bro` is a small benchmark library and CLI tool, with a workflow similar to pytest discovery via decorators.
CLI output is rendered with Rich tables for readable benchmark summaries.

![Bench Bro Mascot](docs/assets/images/mascot-full-dark.png)

## Quick start

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

GC is disabled during measured iterations by default. To keep the interpreter GC behavior unchanged, set:

```python
Case(name="hashing", gc_control="inherit")
```

Run them from CLI:

```bash
benchbro run --repeats 10 --warmup 2
```

When no target is provided, benchbro discovers benchmarks from:
- `benchmarks/**/*.py` (relative to repo root)

By default, run artifacts are written to:
- `.benchbro/current.json`
- `.benchbro/current.csv`

JSON artifacts include run environment metadata (Python/runtime/platform/CPU fields) for comparison context.

## CLI basics

Run selected cases/tags and write outputs:

```bash
benchbro run my_benchmarks.py \
  --case hashing \
  --tag fast \
  --output-json artifacts/current.json \
  --output-csv artifacts/current.csv \
  --output-md artifacts/current.md
```

Compare against a baseline and fail CI on regressions above 5%:

```bash
benchbro run my_benchmarks.py \
  --baseline-json artifacts/baseline.json \
  --fail-on-regression 5
```

Compare two saved runs directly:

```bash
benchbro compare artifacts/baseline.json artifacts/current.json --fail-on-regression 5
```

## End-to-end example

For a complete runnable workflow (baseline, candidate, and compare), use:

- `examples/README.md`
- `make examples`
