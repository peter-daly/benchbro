# Benchbro End-to-End Example

Run these commands from the repository root.

## 1) Baseline run

```bash
python -m benchbro.cli run examples/e2e_benchmarks.py \
  --output-json examples/artifacts/baseline.json \
  --output-csv examples/artifacts/baseline.csv
```

## 2) Candidate run (simulate slowdown)

```bash
BENCHBRO_EXAMPLE_SLOW_FACTOR=3 \
python -m benchbro.cli run examples/e2e_benchmarks.py \
  --output-json examples/artifacts/current.json \
  --output-csv examples/artifacts/current.csv
```

## 3) Compare saved runs

```bash
python -m benchbro.cli compare \
  examples/artifacts/baseline.json \
  examples/artifacts/current.json \
  --fail-on-regression 5
```

## 4) Compare during run

```bash
BENCHBRO_EXAMPLE_SLOW_FACTOR=3 \
python -m benchbro.cli run examples/e2e_benchmarks.py \
  --baseline-json examples/artifacts/baseline.json \
  --fail-on-regression 5
```

The output table shows:
- `case`
- `benchmark`
- `metric_type`
- metric keys and values
