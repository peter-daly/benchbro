# benchbro

`benchbro` is a Python benchmarking library and CLI with parameterized discovery,
noise-aware statistics, reproducible environments, and rich terminal output.

[Read the documentation](https://peter-daly.github.io/benchbro/) or jump straight
to the [quick start](https://peter-daly.github.io/benchbro/getting-started/quick-start/).

![Bench Bro Mascot](https://raw.githubusercontent.com/peter-daly/benchbro/main/docs/assets/images/mascot-full-dark.png)

## Quick start

Add the CLI to a project:

```bash
uv add --dev benchbro
```

For a cloned development checkout, use `uv sync --all-groups` instead. A global
installation with `uv tool install benchbro` is also supported; in that case run
`benchbro` directly instead of `uv run benchbro` in the examples below.

BenchBro bundles a version-matched agent skill. After installing the package, add
it to the current project with:

```bash
uvx library-skills --skill use-benchbro --yes
```

Create benchmark cases in any importable module:

```python
from benchbro import Case, system


@system(scope="module")
def salt() -> bytes:
    return b"-fixture"


case = Case(name="hashing", case_type="cpu", metric_type="time", tags=["fast", "core"])


@case.input()
def payload() -> bytes:
    return b"benchbro"


@case.benchmark()
def sha1(payload: bytes, salt: bytes) -> str:
    import hashlib

    return hashlib.sha1(payload + salt).hexdigest()


@case.benchmark()
def sha256(payload: bytes, salt: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload + salt).hexdigest()
```

Standalone systems behave like lightweight fixtures:

- declare them with `@system()` outside any `Case`
- inject them into benchmarks and other systems by parameter name
- choose `scope="function"` (default) or `scope="module"`
- systems may be sync, async, yield-based, or async-yield-based
- async inputs, systems, benchmarks, and teardown share one managed event loop
- keep them separate from inputs: inputs cannot depend on systems, and systems cannot depend on inputs

Subprocess benchmarks use a dedicated subprocess API:

```python
import sys

from benchbro import Case, CommandSpec

case = Case(name="cli", metric_type="time")


@case.subprocess()
def python_cli() -> CommandSpec:
    return CommandSpec(argv=[sys.executable, "-c", "pass"])
```

- `@case.subprocess()` benchmarks are time-only in v1.
- Commands are explicit argv, not shell strings.
- Unexpected exit codes and timeouts fail the run.
- Long-lived subprocesses should be started in `@system(...)` and consumed by benchmarks or subprocess benchmarks.

Regression thresholds default to `50.0` percent warning and `100.0` percent error at the case level, and can be overridden per benchmark:

```python
case = Case(name="hashing", warning_threshold_pct=5.0, regression_threshold_pct=10.0)


@case.benchmark(warning_threshold_pct=2.0, regression_threshold_pct=3.0)
def critical_path(payload: bytes) -> str: ...
```

Comparison metric can be configured at case-level and benchmark-level:

```python
case = Case(name="hashing", comparison_metric="p95_s")


@case.benchmark(comparison_metric="median_s")
def critical_path(payload: bytes) -> str: ...
```

Valid comparison metrics:

- time: `median_s` (default), `mean_s`, `iqr_s`, `p95_s`, `stddev_s`, `ops_per_sec`
- memory: `peak_alloc_bytes` (default), `net_alloc_bytes`, `peak_alloc_bytes_max`

GC is disabled during measured iterations by default. To keep the interpreter GC behavior unchanged, set:

```python
Case(name="hashing", gc_control="inherit")
```

Run benchmarks:

```bash
uv run benchbro run --repeats 10 --warmup 2
```

The defaults are 20 repeats, 50 measured iterations per repeat, and 5 warmup
iterations. Percentiles are calculated across repeat-level per-iteration means.

For adaptive sampling, give BenchBro a precision target and/or time budget:

```python
case = Case(
    name="hashing",
    adaptive=True,
    min_repeats=5,
    repeats=100,  # maximum
    target_relative_margin_pct=2.0,
    max_time_s=10.0,
    noise_threshold_pct=10.0,
)
```

Adaptive runs stop after the minimum sample count once the 95% confidence
interval reaches the requested relative margin, or at the time/repeat limit.
Results include confidence bounds, variance, standard error, coefficient of
variation, outlier count, and a noisy/stable quality flag.

## Parameters, scopes, and measurement boundaries

Parameter sets create independently named benchmarks. Put `parametrize` below
`benchmark`, as shown here:

```python
case = Case(name="encoding")


@case.benchmark()
@case.parametrize("size", [100, 10_000], ids=["small", "large"])
@case.parametrize("sort_keys", [False, True], ids=["unsorted", "sorted"])
def encode(size: int, sort_keys: bool) -> str:
    import json

    return json.dumps(list(range(size)), sort_keys=sort_keys)
```

Multiple parameter declarations form a Cartesian product. Parameters are
injected by name and recorded in JSON, CSV, Markdown, listing, and history data.

Systems support `iteration`, `benchmark`, and `session` scopes. The legacy names
`function` and `module` remain aliases for `benchmark` and `session`:

```python
@system(scope="iteration")
def temporary_resource():
    resource = create_resource()
    yield resource
    resource.close()
```

Use `setup_timing="exclude"` or `teardown_timing="exclude"` on a `Case` or
benchmark override to keep fixture work outside the measured interval. Warmup,
iterations, repeats, timeout, isolation, timing boundaries, and profiling can be
overridden on individual `@case.benchmark(...)` declarations.

## Isolation and profiling

Run each selected benchmark in its own worker process:

```bash
uv run benchbro run benchmarks --isolation process
```

Or configure `isolation="process"` on a case. Python benchmark functions must be
importable when the operating system does not support `fork`. `timeout_s` is a
hard limit for subprocess benchmarks and isolated workers; in-process synchronous
benchmarks are checked immediately after each invocation. Linux runners can be
pinned with `--cpu-affinity 2,3` or `cpu_affinity=(2, 3)`.

Built-in profiler hooks support `cprofile` and `tracemalloc` allocation snapshots:

```bash
uv run benchbro run benchmarks --profile cprofile \
  --profile-output '.benchbro/profiles/{case}-{benchmark}.prof'
```

Custom integrations—including wrappers around external profilers—can implement
`ProfilerHook` and register a factory with `register_profiler(...)`.

When no target is provided, `benchbro` discovers benchmarks from:

- `benchmarks/**/*.py` (relative to repo root)

You can configure discovery in `pyproject.toml` or a standalone `benchbro.toml`:

```toml
[tool.benchbro.ini_options]
benchmark_paths = ["benchmarks"]
file_pattern = ["bench_*.py", "*_bench.py", "*benchmark.py", "*benchmarks.py"]
```

- `benchmark_paths`: directories to scan when no CLI target is provided
- `file_pattern`: glob pattern(s) for benchmark file names in directory discovery

A standalone configuration can also hold common run and output options:

```toml
benchmark_paths = ["benchmarks"]
baseline = "feature-branch"
save_history = true

[run]
repeats = 100
warmup = 5
min_iterations = 50
adaptive = true
min_repeats = 5
min_time_s = 0.25
max_time_s = 10.0
target_relative_margin_pct = 2.0
noise_threshold_pct = 10.0
stabilization_delay_s = 0.5
isolation = "in_process"
# cpu_affinity = [2, 3] # Linux only

[output]
json = "artifacts/current.json"
markdown = "artifacts/current.md"
```

Command-line values take precedence over configuration. Bootstrap a project with
`benchbro init`; it creates `benchbro.toml` and a parameterized starter benchmark.

`benchbro` compares against the baseline by default (`.benchbro/baseline.local.json`).
If the baseline is missing, benchbro creates it automatically.
If new cases/benchmarks are introduced later, missing entries are merged into baseline.
Pass `--new-baseline` to replace the entire baseline with the current run.
Pass `--ci` to use `.benchbro/baseline.ci.json` for baseline read/write/compare.
Pass `--no-compare` to skip comparison while still backfilling missing benchmark entries in baseline.

CI mode is deliberately strict: it fails when `baseline.ci.json` is missing or
does not contain every selected benchmark. Create or replace that file explicitly
with `--ci --new-baseline`, then commit it. Runs also reject comparisons across a
different Python major/minor version, implementation, operating system, or machine
architecture. Use `--allow-environment-mismatch` only when that difference is
intentional.

Named baselines keep environments or branches separate:

```bash
uv run benchbro baseline update benchmarks --baseline macos-arm64
uv run benchbro run benchmarks --baseline macos-arm64
uv run benchbro baseline list
```

Names other than `local` and `ci` live under `.benchbro/baselines/`. Result files
carry `schema_version = 2`; the reader migrates version-1 artifacts and rejects
unknown future schemas rather than silently misreading them.

By default, regular runs do not write artifacts.
Use explicit output flags (`--output-json`, `--output-csv`, `--output-md`) when needed.

The baseline is always written to:
- `.benchbro/baseline.local.json` (default local mode)
- `.benchbro/baseline.ci.json` when using `--ci`

Recommended:

- ignore `.benchbro/` for machine-local benchmarking artifacts.
- commit `.benchbro/baseline.ci.json` for CI comparisons.

### Recommended `.gitignore`

```gitignore
# Benchbro local artifacts
.benchbro/*
!.benchbro/baseline.ci.json
```

If requested, markdown output can also be written with `--output-md`.

JSON artifacts include environment metadata for reproducibility (Python/runtime/platform/CPU fields) both at run level and on each benchmark entry.

## CLI basics

The command-oriented interface is:

```text
benchbro run [target]
benchbro list [target] [--verbose]
benchbro compare BASELINE.json CURRENT.json
benchbro baseline update [target] --baseline NAME
benchbro baseline list
benchbro history list|show|compare
benchbro init [path]
benchbro completion bash|zsh|fish
```

The original `benchbro TARGET [options]` syntax remains supported.

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

Exit codes distinguish outcomes:

- `0`: successful or non-regressing run
- `1`: invalid input, discovery, configuration, or comparison setup
- `2`: statistically supported regression
- `3`: benchmark execution failure

Threshold crossings with insufficient evidence are reported as `LIKELY` or
`INCONCLUSIVE` without returning the regression exit code. A single-sample run
cannot claim statistical confidence.

## Local history

Enable `save_history = true` or pass `--save-history` to retain schema-versioned
runs beneath `.benchbro/history/`:

```bash
uv run benchbro history list
uv run benchbro history show COMMIT_OR_FILENAME_FRAGMENT
uv run benchbro history compare OLDER NEWER
```

History selectors accept an exact path or an unambiguous filename fragment.

## Pytest integration

Install the optional integration with `uv add --dev 'benchbro[pytest]'`, then
load the plugin from a pytest configuration:

```toml
[tool.pytest.ini_options]
addopts = "-p benchbro.pytest_plugin"
```

Then reuse pytest fixtures in a measured callable:

```python
def test_parser_speed(benchbro_runner, parsed_fixture):
    result = benchbro_runner(
        parse,
        parsed_fixture,
        repeats=20,
        warmup=5,
        min_iterations=50,
    )
    assert result.metrics["median_s"] < 0.01
```

The integration is opt-in, so BenchBro does not add pytest as a runtime
dependency or interfere with normal test collection.

## End-to-end example

For a complete runnable workflow (baseline + candidate comparison), use:

- `examples/README.md`
- `make examples`

## Development

Run the same checks used by CI:

```bash
uv sync --all-groups
make ci
make docs
uv run tox
```

See `CONTRIBUTING.md` for the contributor workflow and `CHANGELOG.md` for release
history. Maintainers should also complete the account-level controls in
`docs/maintenance.md`.
