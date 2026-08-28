# Defining benchmarks

A `Case` groups benchmarks that share an input, metric type, settings, tags, and
comparison policy.

## Time benchmarks

`metric_type="time"` is the default. The measured callable may be synchronous or
asynchronous:

```python
from benchbro import Case

case = Case(name="parser", tags=["core"])


@case.benchmark()
def parse_small() -> object:
    return parse("name = 'benchbro'")


@case.benchmark()
async def parse_remote_shape() -> object:
    return await async_parse("name = 'benchbro'")
```

The default result includes the mean, median, interquartile range, p95, standard
deviation, operations per second, standard error, confidence bounds, coefficient
of variation, and sample/outlier counts.

## Memory benchmarks

Use `metric_type="memory"` to measure Python allocations with `tracemalloc`:

```python
memory_case = Case(name="allocation", metric_type="memory")


@memory_case.benchmark()
def build_records() -> list[dict[str, int]]:
    return [{"value": value} for value in range(1_000)]
```

Memory results include peak and net allocated bytes. Subprocess benchmarks do not
support memory mode.

## Parameters

Stack `parametrize` below `benchmark`. Multiple declarations create a Cartesian
product, and every combination becomes an independently named result:

```python
case = Case(name="encoding")


@case.benchmark()
@case.parametrize("size", [100, 10_000], ids=["small", "large"])
@case.parametrize("sort_keys", [False, True], ids=["unsorted", "sorted"])
def encode(size: int, sort_keys: bool) -> str:
    import json

    return json.dumps(list(range(size)), sort_keys=sort_keys)
```

The example registers names such as `encode[small-unsorted]`. Parameter values are
also recorded in JSON, CSV, Markdown, listing, and history data.

## Per-benchmark overrides

Case settings provide a shared default. Override the settings that differ on an
individual callable:

```python
case = Case(
    name="critical",
    warning_threshold_pct=5.0,
    regression_threshold_pct=10.0,
)


@case.benchmark(
    repeats=50,
    warning_threshold_pct=2.0,
    regression_threshold_pct=3.0,
    comparison_metric="p95_s",
)
def hot_path() -> None:
    perform_work()
```

Warmup, iteration count, repeats, adaptive mode, timeout, isolation, timing
boundaries, and profiling can also be overridden per benchmark.

## Filtering

Select cases and tags from the command line:

```console
$ uv run benchbro run --case encoding --tag core
$ uv run benchbro list --tag fast --verbose
```
