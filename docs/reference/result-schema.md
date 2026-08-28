# Result schema

JSON artifacts are the durable interchange format for runs and baselines. The
current `schema_version` is `2`.

## Run object

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Payload compatibility version |
| `run_id` | string | Unique run identifier |
| `started_at` / `finished_at` | ISO 8601 string | UTC run boundaries |
| `python_version` | string | Full Python version |
| `platform` | string | Platform description |
| `environment` | object | Runtime, hardware, repository, and runner metadata |
| `benchmarks` | array | One result per registered benchmark/parameter combination |

## Benchmark result

Each entry identifies the case and benchmark, records its effective settings and
environment, and stores numeric observations beneath `metrics`.

| Field | Meaning |
| --- | --- |
| `case_name`, `benchmark_name` | Stable comparison key together with `metric_type` |
| `case_type`, `metric_type`, `gc_control` | Benchmark classification and measurement mode |
| `parameters`, `parameter_id` | Parameter values and generated display identifier |
| `iterations`, `repeats`, `duration_s` | Work and elapsed measurement counts |
| `comparison_metric` | Explicit comparison metric, or `null` for the type default |
| `warning_threshold_pct`, `regression_threshold_pct` | Effective thresholds |
| `metrics` | Measurement-specific numeric values |
| `is_noisy`, `outlier_count` | Sample quality signals |
| `profile_path` | Generated profile artifact, when profiling was active |

Raw time samples are used during a live run but are omitted from JSON output. The
summary contains the statistics needed for later confidence-aware comparison.

## Compatibility policy

The reader migrates version-1 artifacts to the current model. It rejects invalid
versions and unknown future schema versions instead of guessing how to interpret
them.

Environment metadata is present at run level and copied onto benchmark entries.
Comparisons check Python major/minor, implementation, operating system, and machine
architecture before evaluating metrics.

## Other formats

- CSV writes one row per metric, suitable for analysis tools.
- Markdown writes a human-readable summary table.
- Only JSON contains the complete durable run model and should be used for baselines.
