# Reliable measurements

Benchmark numbers are observations, not constants. Scheduler activity, CPU power
state, garbage collection, background work, and workload setup can all affect a
run. BenchBro records evidence about that variation instead of hiding it.

## The sampling model

For a time benchmark, BenchBro:

1. performs warmup invocations;
2. runs `min_iterations` invocations per repeat;
3. records one per-iteration mean for each repeat;
4. summarizes those repeat samples.

Defaults are 5 warmups, 50 measured iterations per repeat, and 20 repeats.
Percentiles are calculated across repeat-level samples.

## Adaptive sampling

Adaptive mode can stop early once the estimate is precise enough:

```python
case = Case(
    name="hashing",
    adaptive=True,
    min_repeats=5,
    repeats=100,  # maximum
    min_time_s=0.25,
    max_time_s=10.0,
    target_relative_margin_pct=2.0,
    noise_threshold_pct=10.0,
)
```

After `min_repeats` and `min_time_s`, the run stops when the 95% confidence
interval's relative margin reaches the target. It always stops at the maximum
repeat count or `max_time_s`.

## Quality signals

Time results include:

- `variance_s2`, `stddev_s`, and `standard_error_s`;
- `ci95_low_s`, `ci95_high_s`, and `relative_margin_pct`;
- `cv_pct` (coefficient of variation);
- `outlier_count`, `sample_count`, and the `is_noisy` flag.

An `is_noisy` result is not automatically a regression. It is a prompt to inspect
the environment, increase the work per sample, or collect more samples.

## Comparison metrics

| Metric type | Default | Alternatives |
| --- | --- | --- |
| Time | `median_s` | `mean_s`, `iqr_s`, `p95_s`, `stddev_s`, `ops_per_sec` |
| Memory | `peak_alloc_bytes` | `net_alloc_bytes`, `peak_alloc_bytes_max` |

Choose a metric that matches the question. Median is robust for typical latency;
p95 emphasizes slower samples; operations per second reverses the direction so a
drop is a regression.

## Regression confidence

Warnings and errors have separate thresholds. The defaults are deliberately loose:
50% for a warning and 100% for an error. Set tighter values for stable, important
benchmarks.

When a result crosses the error threshold, BenchBro uses the available standard
errors to estimate confidence. A statistically supported crossing is a regression;
weaker evidence is reported as `LIKELY` or `INCONCLUSIVE`. A one-sample run cannot
claim statistical confidence.

## Reduce avoidable noise

- Close unrelated CPU- and disk-heavy applications.
- Keep the Python version, operating system, and machine architecture consistent.
- Prefer a fixed power mode and stable thermal state.
- Use `stabilization_delay_s` before measurement when a service needs to settle.
- On Linux, consider `cpu_affinity=(2, 3)` or `--cpu-affinity 2,3`.
- Keep the default `gc_control="disable_during_measure"` unless GC is part of the
  behavior being tested.
- Compare committed code and a stable lockfile; BenchBro records both when present.

!!! warning
    Use `--allow-environment-mismatch` only when comparing different environments
    is intentional. A cross-machine percentage is rarely an application regression.
