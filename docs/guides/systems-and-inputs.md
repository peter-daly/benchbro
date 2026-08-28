# Systems and inputs

BenchBro injects inputs, systems, and parameter values into a benchmark by matching
function parameter names.

## Inputs belong to a case

An input prepares the benchmark's primary subject:

```python
from benchbro import Case

case = Case(name="lookup")


@case.input()
def records() -> dict[str, int]:
    return {str(value): value for value in range(1_000)}


@case.benchmark()
def existing_key(records: dict[str, int]) -> int:
    return records["500"]
```

Inputs may be sync, async, yield-based, or async-yield-based. They cannot depend on
systems, and systems cannot depend on inputs; this keeps the dependency graph
unambiguous.

## Systems are reusable dependencies

Declare a system independently of any case:

```python
from collections.abc import Iterator

from benchbro import Case, system


@system(scope="benchmark")
def database() -> Iterator[Database]:
    db = Database.in_memory()
    yield db
    db.close()


@system(scope="iteration")
def transaction(database: Database) -> Iterator[Transaction]:
    with database.transaction() as tx:
        yield tx


case = Case(name="insert")


@case.benchmark()
def one_row(transaction: Transaction) -> None:
    transaction.insert({"name": "BenchBro"})
```

Systems can depend on other systems. BenchBro resolves the graph by name and
rejects missing dependencies or cycles before silently producing a bad run.

## Choose a scope

| Scope | Lifetime | Typical use |
| --- | --- | --- |
| `iteration` | One measured invocation | Transaction, temporary file, request context |
| `benchmark` | All repeats for one benchmark | Database, client, seeded data set |
| `session` | All selected benchmarks in a run | Server process, shared service |

The legacy scope names `function` and `module` remain aliases for `benchmark` and
`session`.

## Async lifecycle

Async inputs, systems, benchmark functions, and teardown share one managed event
loop. An async generator can therefore safely create and close a resource on the
same loop:

```python
@system(scope="session")
async def client():
    async with AsyncClient() as instance:
        yield instance
```

## Measurement boundaries

Setup is included and teardown is excluded by default. Move lifecycle work outside
the measured interval when you want to isolate only the callable:

```python
case = Case(
    name="query",
    setup_timing="exclude",
    teardown_timing="exclude",
)
```

Be explicit about this choice in performance reports: excluding realistic setup
can make a microbenchmark less representative of production behavior.
