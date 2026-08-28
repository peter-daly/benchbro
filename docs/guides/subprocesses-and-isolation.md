# Subprocesses and isolation

BenchBro supports two related tools: subprocess benchmarks for external commands,
and process isolation for Python benchmarks.

## Benchmark a command

Return a `CommandSpec` from a `@case.subprocess()` function:

```python
import sys

from benchbro import Case, CommandSpec

case = Case(name="cli", metric_type="time")


@case.subprocess()
def startup() -> CommandSpec:
    return CommandSpec(
        argv=[sys.executable, "-c", "pass"],
        timeout_s=2.0,
        expected_exit_codes=(0,),
    )
```

Commands use an explicit argument vector, never a shell command string. This avoids
shell parsing differences and accidental interpolation. Unexpected exit codes and
timeouts fail the benchmark run.

`CommandSpec` also supports:

- `cwd` for the working directory;
- an `env` mapping merged into the child environment;
- `stdin_bytes` for fixed input;
- `stdout` and `stderr` set to `"devnull"` or `"inherit"`.

Subprocess benchmarks are time-only. Start long-lived services in a session-scoped
system, then inject that system into the benchmark.

## Isolate a Python benchmark

Run every selected Python benchmark in its own worker process:

```console
$ uv run benchbro run benchmarks --isolation process
```

Or make isolation part of one case:

```python
case = Case(name="parser", isolation="process", timeout_s=10.0)
```

Isolation prevents state and interpreter effects from leaking between benchmarks.
It also gives timeouts a hard process boundary. The tradeoff is additional startup
cost and stricter importability requirements.

!!! note
    On platforms without `fork`, isolated benchmark functions must be importable
    from their defining module. Avoid local functions and lambdas.

## Timeout behavior

- Subprocess commands and isolated workers are terminated at their hard timeout.
- In-process synchronous callables are checked immediately after each invocation;
  they cannot be interrupted halfway through a Python call.
- Async callables run on BenchBro's managed event loop.
