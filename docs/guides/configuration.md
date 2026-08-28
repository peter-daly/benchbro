# Configuration

BenchBro reads project configuration from `pyproject.toml` and then overlays a
standalone `benchbro.toml` when present. Command-line options have final precedence.

## `benchbro.toml`

```toml
benchmark_paths = ["benchmarks"]
file_pattern = ["bench_*.py", "*_bench.py"]
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
csv = "artifacts/current.csv"
markdown = "artifacts/current.md"
```

## `pyproject.toml`

Put the same keys beneath `[tool.benchbro]`. The legacy
`[tool.benchbro.ini_options]` discovery table is also supported:

```toml
[tool.benchbro]
benchmark_paths = ["benchmarks"]
baseline = "local"

[tool.benchbro.run]
adaptive = true
max_time_s = 5.0
```

## Discovery

When no target is passed, BenchBro scans the configured benchmark paths. Its default
file patterns are:

```text
bench_*.py
*_bench.py
*benchmark.py
*benchmarks.py
```

A direct module, file, or directory target bypasses the default path selection but
still uses the configured file patterns for directory discovery.

## Generate a starter

```console
$ uv run benchbro init
```

The command creates `benchbro.toml` and a parameterized starter benchmark without
overwriting existing files. Use `benchbro init path/to/project` for another project
directory.

## Precedence

From lowest to highest priority:

1. BenchBro defaults;
2. settings declared on `Case`;
3. per-benchmark decorator overrides;
4. `pyproject.toml`;
5. `benchbro.toml`;
6. explicit command-line options.

Configuration only overrides values it defines, so a small project file can leave
benchmark-specific choices with the code that owns them.
