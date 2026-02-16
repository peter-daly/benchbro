# benchbro

`benchbro` is a Python benchmarking library and CLI with pytest-style discovery and rich terminal output.

![Bench Bro Mascot](https://raw.githubusercontent.com/peter-daly/benchbro/main/docs/assets/images/mascot-full-dark.png)

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

Regression thresholds default to `50.0` percent warning and `100.0` percent error at the case level, and can be overridden per benchmark:

```python
case = Case(name="hashing", warning_threshold_pct=5.0, regression_threshold_pct=10.0)

@case.benchmark(warning_threshold_pct=2.0, regression_threshold_pct=3.0)
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

`benchbro` compares against the baseline by default (`.benchbro/baseline.local.json`).
If the baseline is missing, benchbro creates it automatically.
If new cases/benchmarks are introduced later, missing entries are merged into baseline.
Pass `--new-baseline` to replace the entire baseline with the current run.
Pass `--ci` to use `.benchbro/baseline.ci.json` for baseline read/write/compare.
Pass `--no-compare` to skip comparison while still backfilling missing benchmark entries in baseline.

By default, regular runs do not write artifacts.
Use explicit output flags (`--output-json`, `--output-csv`, `--output-md`) when needed.

The baseline is always written to:
- `.benchbro/baseline.local.json` (default local mode)
- `.benchbro/baseline.ci.json` when using `--ci`

Recommended:
- ignore `.benchbro/` for machine-local benchmarking artifacts.
- commit `.benchbro/baseline.ci.json` for CI comparisons.

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

Render time benchmark histograms in terminal output:

```bash
uv run benchbro my_benchmarks.py --histogram
```

Skip comparison for a run while still maintaining baseline structure:

```bash
uv run benchbro my_benchmarks.py --no-compare
```

Regression status uses each benchmark's effective thresholds (`benchmark override -> case threshold -> defaults`):
- warning default: `50%`
- error threshold default: `100%`

The comparison table shows warning and threshold values for each row.

Histograms are terminal-only in v1 and are shown for time benchmarks.

## End-to-end example

For a complete runnable workflow (baseline + candidate comparison), use:

- `examples/README.md`
- `make examples`
