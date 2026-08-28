# Benchbro End-to-End Example

Run these commands from the repository root.

## 1) Baseline run

```bash
uv run benchbro examples/e2e_benchmarks.py
```

## 2) Candidate run (simulate slowdown)

```bash
BENCHBRO_EXAMPLE_SLOW_FACTOR=3 \
uv run benchbro examples/e2e_benchmarks.py \
  --output-json examples/artifacts/current.json \
  --output-csv examples/artifacts/current.csv \
  --output-md examples/artifacts/current.md
```

## 3) Compare during run

```bash
BENCHBRO_EXAMPLE_SLOW_FACTOR=3 \
uv run benchbro examples/e2e_benchmarks.py
```

The output table shows:

- `case`
- `benchmark`
- `metric_type`
- metric keys and values

When a baseline is present, the comparison table also shows warning/error
thresholds, percentage change, and status.

## Advanced gallery

`advanced_benchmarks.py` separately demonstrates:

- adaptive, confidence-bounded sampling
- Cartesian parameter sets with readable IDs
- session-scoped compiled resources
- explicit setup and teardown timing boundaries
- realistic JSON serialization and text parsing workloads

Explore it without changing a baseline:

```bash
uv run benchbro list examples/advanced_benchmarks.py --verbose
uv run benchbro run examples/advanced_benchmarks.py --no-compare
```
