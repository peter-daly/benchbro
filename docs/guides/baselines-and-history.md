# Baselines and history

Baselines answer “did this get slower?” History answers “what happened across many
runs?” Both use the same schema-versioned JSON results.

## Local baseline

A regular run compares against `.benchbro/baseline.local.json`. If it does not
exist, BenchBro creates it. New benchmark entries are merged into an existing local
baseline without replacing established values.

```console
$ uv run benchbro run
$ uv run benchbro run --new-baseline
$ uv run benchbro run --no-compare
```

`--new-baseline` replaces the selected baseline. `--no-compare` skips the comparison
but still fills missing local entries.

## Named baselines

Keep environments or branches separate:

```console
$ uv run benchbro baseline update benchmarks --baseline macos-arm64
$ uv run benchbro run benchmarks --baseline macos-arm64
$ uv run benchbro baseline list
```

Names other than `local` and `ci` are stored under `.benchbro/baselines/`.

## CI baseline

`--ci` selects `.benchbro/baseline.ci.json`. CI mode is strict: the file must exist
and contain every selected benchmark.

```console
$ uv run benchbro run benchmarks --ci --new-baseline
```

Commit that baseline, then use `--ci` for later comparisons. Keep machine-local
artifacts ignored while allowing the CI baseline:

```gitignore
.benchbro/*
!.benchbro/baseline.ci.json
```

## Environment compatibility

BenchBro rejects a baseline comparison when the Python major/minor version,
implementation, operating system, or machine architecture differs. Override with
`--allow-environment-mismatch` only when the difference is part of the experiment.

## Comparison statuses

| Status | Meaning |
| --- | --- |
| `stable` | No meaningful threshold crossing |
| `improvement` | Metric moved in the favorable direction |
| `warning` | Warning threshold crossed |
| `likely_regression` | Error threshold crossed with moderate evidence |
| `inconclusive` | Error threshold crossed without enough evidence |
| `regression` | Error threshold crossed with the required confidence |
| `noisy` | Variation exceeded the configured noise threshold |

## Local history

Enable `save_history = true` or pass `--save-history`:

```console
$ uv run benchbro history list
$ uv run benchbro history show COMMIT_OR_FILENAME_FRAGMENT
$ uv run benchbro history compare OLDER NEWER
```

Entries live under `.benchbro/history/`. A selector can be an exact path or an
unambiguous filename fragment.
