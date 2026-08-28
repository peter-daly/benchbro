# Advanced BenchBro Patterns

Read this reference when a benchmark uses managed dependencies, async lifecycle,
subprocesses, isolation, profiling, pytest, or the programmatic API.

## Separate inputs from systems

An input belongs to one case and provides its primary subject. A system is a
standalone reusable dependency. BenchBro injects both by parameter name:

```python
from collections.abc import Iterator

from benchbro import Case, system


@system(scope="benchmark")
def database() -> Iterator[Database]:
    instance = Database.in_memory()
    yield instance
    instance.close()


case = Case(name="lookup")


@case.input()
def records() -> list[Record]:
    return make_records()


@case.benchmark()
def find_record(records: list[Record], database: Database) -> Record:
    return database.find(records[0].id)
```

Systems can depend on other systems. Inputs cannot depend on systems, and systems
cannot depend on inputs. Keep names unique and allow BenchBro to reject missing or
cyclic dependencies.

## Choose system ownership

| Scope | Lifetime | Typical use |
| --- | --- | --- |
| `iteration` | One measured invocation | Transaction, temporary file, request state |
| `benchmark` | All repeats for one benchmark | Database, client, seeded data |
| `session` | All selected benchmarks in a run | Server process, shared service |

`function` and `module` remain aliases for `benchmark` and `session`. Prefer the
canonical names in new code.

Sync and async functions, generators, and context-manager-shaped callables are
supported. Async inputs, systems, benchmark functions, and teardown share one
managed event loop. Put cleanup after `yield` and make sure the owner exits even
when the benchmark fails.

## State measurement boundaries explicitly

Setup is included and teardown is excluded by default. Override the boundary on a
case or benchmark:

```python
case = Case(
    name="query",
    setup_timing="exclude",
    teardown_timing="exclude",
)
```

Excluding fixture work narrows the question to the callable. Include it when setup
is part of user-visible performance. Document the choice because two otherwise
identical benchmarks can answer different questions.

## Benchmark external commands safely

Return a `CommandSpec` from a synchronous `@case.subprocess()` callable:

```python
import sys

from benchbro import Case, CommandSpec

case = Case(name="cli")


@case.subprocess()
def startup() -> CommandSpec:
    return CommandSpec(
        argv=[sys.executable, "-c", "pass"],
        timeout_s=2.0,
        expected_exit_codes=(0,),
    )
```

Subprocess benchmarks support time metrics only. Use explicit argument vectors,
never shell strings. `CommandSpec` can also set `cwd`, merge string environment
values, provide `stdin_bytes`, and inherit or discard stdout and stderr. Unexpected
exit codes and timeouts fail the run.

Start long-lived external services in a session-scoped system and inject them into
the command factory or Python benchmark.

## Isolate Python benchmarks when needed

Use process isolation to prevent state or interpreter effects from leaking between
benchmarks and to enforce a hard worker timeout:

```bash
benchbro run benchmarks --isolation process --timeout 10
```

Or set `isolation="process"` and `timeout_s` on a case. Spawn-based platforms
require benchmark functions to be importable from their module; avoid lambdas,
closures, and local functions. Isolation adds process startup and serialization
constraints, so use it to solve a concrete interference problem.

In-process synchronous timeout checks happen after an invocation returns and cannot
interrupt Python midway through a call. Subprocess and isolated-worker timeouts are
hard boundaries.

CPU affinity is supported on platforms exposing the required scheduler APIs,
normally Linux. Preserve and restore the process's original affinity. Do not assume
`--cpu-affinity` is portable.

## Attach profilers to explain results

Built-in names are `cprofile` and `tracemalloc`:

```bash
benchbro run benchmarks \
  --profile cprofile \
  --profile-output '.benchbro/profiles/{case}-{benchmark}.prof'
```

Custom profilers implement `ProfilerHook` and are registered with a factory:

```python
from pathlib import Path

from benchbro import register_profiler


class MyProfiler:
    def start(self) -> None: ...

    def stop(self, output: Path) -> None: ...


register_profiler("my-profiler", MyProfiler)
```

The factory creates one hook per profiled benchmark. Profiling changes execution
behavior; compare profiled runs with like-for-like profiled baselines, or use the
profile only to investigate an unprofiled result.

## Integrate with pytest explicitly

Install the optional extra and load the plugin:

```bash
uv add --dev 'benchbro[pytest]'
```

```toml
[tool.pytest.ini_options]
addopts = "-p benchbro.pytest_plugin"
```

Use normal pytest fixtures to prepare arguments and `benchbro_runner` to measure the
callable:

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

The pytest integration is opt-in. Treat hard timing assertions with caution on
shared runners; comparison against a stable environment-aware baseline is usually
more informative.

## Use the programmatic API for custom orchestration

Decorators populate the registry. Retrieve and filter cases, then execute them:

```python
from benchbro import filter_cases, list_cases, run_cases, write_json

selected = filter_cases(list_cases(), names=["parser"], tags=["fast"])
run = run_cases(selected, repeats=30, warmup=5, min_iterations=100)
write_json("artifacts/current.json", run)
```

Call `clear_registry()` between independent dynamic loading sessions or tests to
avoid duplicate registrations. The CLI already manages discovery and registry
state; do not clear it from ordinary benchmark modules.

Use `read_json` and `compare_runs` for a custom comparison workflow. Preserve JSON
as the durable format: CSV and Markdown are reporting views, not complete baseline
models. The reader migrates supported schema-version-1 results and rejects unknown
future schema versions.

## Test advanced failure paths

- verify each system scope reuses and tears down resources at the intended point;
- test sync and async cleanup when a benchmark raises;
- confirm parameter IDs and generated benchmark names are stable;
- exercise unexpected command exits and timeouts;
- run isolated benchmarks on a spawn-based platform when portability matters;
- verify profile output paths do not collide across parameterized benchmarks;
- clear global registrations between tests that import benchmark modules repeatedly.
