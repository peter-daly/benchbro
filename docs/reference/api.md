# Python API

The package root re-exports the supported API. Prefer `from benchbro import ...`
unless you are extending an implementation module.

## Definition API

::: benchbro.api.Case
    options:
      members:
        - input
        - parametrize
        - benchmark
        - subprocess

::: benchbro.api.system

::: benchbro.models.CommandSpec

## Execution API

::: benchbro.runner.run_cases

::: benchbro.api.list_cases

::: benchbro.api.clear_registry

## Comparison API

::: benchbro.comparison.compare_runs

::: benchbro.comparison.comparison_environment_mismatches

## Result models

::: benchbro.models.BenchmarkSettings

::: benchbro.models.BenchmarkResult

::: benchbro.models.BenchmarkRun

::: benchbro.models.Regression

## Serialization

::: benchbro.serialization.read_json

::: benchbro.serialization.write_json

::: benchbro.serialization.write_csv

::: benchbro.serialization.write_markdown

## Profiling

::: benchbro.profiling.ProfilerHook

::: benchbro.profiling.register_profiler
