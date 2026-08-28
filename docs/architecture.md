# Architecture

BenchBro keeps the public compatibility facade in `benchbro.core` while its
implementation is divided by responsibility:

- `models`: public data structures, validation, registries, types, and schema version
- `api`: decorators, parameters, and public registration helpers
- `lifecycle`: dependency injection, scoped systems, async handling, setup, and teardown
- `measurement`: statistical summaries, confidence intervals, outliers, and adaptive stopping
- `runner`: orchestration, isolation, affinity, timeouts, GC control, and profiling
- `comparison`: environment compatibility and confidence-aware regression classification
- `serialization`: schema migration and JSON, CSV, and Markdown reports
- `environment`: runtime, hardware, repository, lockfile, power, and runner metadata
- `history`: local run storage and selection
- `config`: `pyproject.toml` and `benchbro.toml` loading
- `profiling`: built-in and user-registered profiler hooks
- `cli`: command parsing, discovery, presentation, and workflow coordination
- `pytest_plugin`: opt-in pytest fixture integration

Dependencies point inward toward `models`. The public `core` module re-exports
the established API so upgrades do not force import changes. Result payloads use
an explicit schema version, and all compatibility migrations live in
`serialization` rather than leaking into the runner or comparison logic.
