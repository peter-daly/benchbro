# Profiling and pytest

BenchBro can attach a profiler to a benchmark and can also measure a callable from
inside a pytest test.

## Built-in profilers

Use `cprofile` for call statistics or `tracemalloc` for an allocation snapshot:

```console
$ uv run benchbro run benchmarks \
    --profile cprofile \
    --profile-output '.benchbro/profiles/{case}-{benchmark}.prof'
```

Set `profiler` and `profile_output` on a `Case` or `@case.benchmark(...)` when the
profile belongs to the benchmark definition:

```python
case = Case(
    name="parser",
    profiler="cprofile",
    profile_output=".benchbro/profiles/{case}-{benchmark}.prof",
)
```

Profiling changes runtime characteristics. Use it to explain a result, not as the
number you compare against an unprofiled baseline.

## Custom profiler hooks

Implement `ProfilerHook`, then register a factory:

```python
from pathlib import Path

from benchbro import register_profiler


class MyProfiler:
    def start(self) -> None: ...

    def stop(self, output: Path) -> None: ...


register_profiler("my-profiler", MyProfiler)
```

Factories create one hook instance per profiled benchmark.

## Pytest integration

Install the optional dependency and opt into the plugin:

```console
$ uv add --dev 'benchbro[pytest]'
```

```toml title="pyproject.toml"
[tool.pytest.ini_options]
addopts = "-p benchbro.pytest_plugin"
```

The `benchbro_runner` fixture measures a callable while ordinary pytest fixtures
prepare its arguments:

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

The integration is opt-in, so BenchBro does not add pytest to runtime dependencies
or change normal test collection.
