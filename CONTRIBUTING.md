# Contributing

## Set up the project

```bash
uv sync --all-groups
uv run pre-commit install
```

## Validate a change

```bash
make ci
uv run tox
```

The test matrix covers Python 3.10 through 3.14. Changes to CLI behavior should
include an installed-wheel smoke test or an equivalent regression test.

Benchmark baselines are machine-sensitive. Local runs use
`.benchbro/baseline.local.json`; update the committed CI baseline only with an
explicit `--ci --new-baseline` run in the intended CI environment.
