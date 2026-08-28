# Reliable BenchBro Measurements

Read this reference when choosing sampling settings, comparison metrics, baselines,
thresholds, or automation behavior.

## Model the performance question first

Name the behavior and metric before tuning repetitions:

- use time metrics for latency, variability, tail behavior, or throughput;
- use memory metrics for Python allocation behavior;
- separate different input sizes or modes with parameters;
- use distinct cases when workloads need different inputs, metrics, or policies;
- include realistic setup when it is part of user-visible performance.

Microbenchmarks answer narrow questions. Do not generalize them to system-level
performance without an end-to-end measurement.

## Understand the sampling model

For each time benchmark, BenchBro warms up the callable, performs
`min_iterations` invocations per repeat, and records one per-iteration mean for
that repeat. Summary statistics and percentiles describe those repeat samples.

Defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `warmup_iterations` | 5 | Unmeasured invocations before sampling |
| `min_iterations` | 50 | Invocations combined into one repeat sample |
| `repeats` | 20 | Fixed samples or adaptive maximum |
| `gc_control` | `disable_during_measure` | Disable GC around measured work |
| `noise_threshold_pct` | 10 | CV threshold for the noisy flag |

Increase `min_iterations` when a callable is too fast for stable timer resolution.
Increase repeats when between-sample variance or confidence is the limiting factor.
Keep `gc_control="inherit"` when garbage collection is part of the behavior being
measured.

## Use adaptive sampling for a precision target

```python
from benchbro import Case

case = Case(
    name="parser",
    adaptive=True,
    min_repeats=5,
    repeats=100,
    min_time_s=0.25,
    max_time_s=10.0,
    target_relative_margin_pct=2.0,
    noise_threshold_pct=10.0,
)
```

After the minimum repeat count and time, sampling stops when the 95% confidence
interval's relative margin reaches the target. The repeat count and `max_time_s`
are hard ceilings. Adaptive stopping does not make a noisy workload stable; inspect
the quality fields and improve the experiment when the target is not reached.

## Interpret result quality

Time results include:

- mean, median, p80, p90, p95, p99, quartiles, and IQR;
- variance, standard deviation, and standard error;
- 95% confidence bounds and relative margin;
- coefficient of variation (`cv_pct`);
- sample and outlier counts;
- operations per second and the `is_noisy` flag.

BenchBro flags a result as noisy when CV exceeds the configured threshold or the
IQR rule finds an outlier. Noise is not itself a regression. Investigate background
load, work per sample, warmup, fixture timing, power state, and sample count.

## Choose the comparison metric

| Metric type | Default | Alternatives |
| --- | --- | --- |
| Time | `median_s` | `mean_s`, `iqr_s`, `p95_s`, `stddev_s`, `ops_per_sec` |
| Memory | `peak_alloc_bytes` | `net_alloc_bytes`, `peak_alloc_bytes_max` |

Median is a robust default for typical latency. Use p95 when slower samples matter,
mean for aggregate cost, IQR or standard deviation for variability, and operations
per second for throughput. A drop in operations per second is a regression; an
increase in latency or allocations is a regression.

## Set thresholds from observed variance

Warning and error thresholds default to 50% and 100%, and can be configured on a
case or overridden per benchmark:

```python
case = Case(
    name="parser",
    comparison_metric="p95_s",
    warning_threshold_pct=5.0,
    regression_threshold_pct=10.0,
)


@case.benchmark(
    warning_threshold_pct=3.0,
    regression_threshold_pct=5.0,
)
def critical_path() -> None:
    parse_document()
```

Run unchanged code repeatedly in the intended environment before tightening these
values. A threshold should be larger than routine variance and small enough to
represent a decision-worthy change.

When an error threshold is crossed, BenchBro combines available standard errors to
estimate confidence. Classifications are:

| Status | Meaning |
| --- | --- |
| `stable` | No meaningful threshold crossing |
| `improvement` | Metric moved in the favorable direction |
| `warning` | Warning threshold crossed |
| `likely_regression` | Error threshold crossed with moderate evidence |
| `inconclusive` | Error threshold crossed without enough evidence |
| `regression` | Error threshold crossed with the required confidence |
| `noisy` | Variation exceeded the quality threshold |

One sample cannot support a confident regression. `likely_regression` and
`inconclusive` do not use the regression exit code.

## Keep baselines environment-specific

Normal runs use `.benchbro/baseline.local.json`. A missing local baseline is created
automatically, and newly discovered benchmarks are backfilled without replacing
established entries. `--new-baseline` replaces the selected baseline.

Use named baselines for distinct stable environments:

```bash
benchbro baseline update benchmarks --baseline macos-arm64
benchbro run benchmarks --baseline macos-arm64
benchbro baseline list
```

Use `--ci --new-baseline` to create `.benchbro/baseline.ci.json` deliberately in the
intended runner environment, then commit it. Later `--ci` runs fail if the file or a
selected benchmark is missing. Do not generate a CI baseline opportunistically in
an ordinary comparison run.

BenchBro rejects comparisons across a different Python major/minor version,
implementation, operating system, or machine architecture. Use
`--allow-environment-mismatch` only when that difference is intentional and report
it as part of the experiment.

## Use exit codes in automation

| Code | Meaning |
| ---: | --- |
| 0 | Successful or non-regressing run |
| 1 | Discovery, configuration, input, or comparison setup error |
| 2 | Statistically supported regression |
| 3 | Benchmark execution failure |

Preserve these distinctions in wrappers. Do not convert every non-zero status into
a claimed performance regression.

## Stabilize the environment

- close unrelated CPU-, disk-, and network-heavy work;
- keep Python, dependencies, OS, architecture, and runner type consistent;
- use a stable power and thermal state;
- add `stabilization_delay_s` when an external service needs to settle;
- consider CPU affinity on a suitable Linux runner;
- retain repository and lockfile metadata with the result;
- store schema-versioned JSON when another run will consume the result.
