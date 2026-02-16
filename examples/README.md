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
- `threshold %`
- metric keys and values
