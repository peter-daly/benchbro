# Changelog

This project follows semantic versioning. Dates use the `YYYY-MM-DD` format.

## Unreleased

## 1.0.0 - 2026-08-28

- Bundle a version-matched `use-benchbro` Library Skill in source and wheel
  distributions.
- Add adaptive sampling with 95% confidence intervals, variance, coefficient of
  variation, outlier detection, time budgets, and noisy-result classification.
- Add per-benchmark process isolation, CPU affinity, timeouts, stabilization
  delays, explicit setup/teardown timing, and profiler hooks.
- Add parameterized benchmarks and iteration/benchmark/session system scopes.
- Add command-oriented run, list, compare, baseline, history, init, and
  completion workflows while retaining the legacy CLI syntax.
- Add named baselines, local history, schema-versioned result migrations, and
  confidence-aware comparisons.
- Add standalone configuration and optional pytest fixture integration.
- Split models, lifecycle, measurement, comparison, serialization, environment,
  profiling, history, configuration, and runner responsibilities into modules.
- Install the package and its CLI correctly in uv-managed development environments.
- Run async inputs, systems, benchmarks, and teardown on one managed event loop.
- Validate benchmark configuration and reject empty or unsafe CI runs.
- Preserve `tracemalloc` state and calculate inclusive percentiles.
- Reject cross-environment comparisons unless explicitly overridden.
- Build and test wheels across Python 3.10 through 3.14.
- Refresh dependencies and remove the unrelated `rust` package.
- Harden automated publishing with PyPI trusted publishing and GitHub Releases.

## 0.5.0 - 2026-03-15

- Add standalone sync and async systems for Python and subprocess benchmarks.

## 0.4.2 - 2026-02-18

- Correct package metadata and console-script configuration.

## 0.4.0 - 2026-02-18

- Add async benchmark support and multi-version test coverage.

## 0.2.0 - 2026-02-16

- Add configurable comparison metrics and regression thresholds.

## 0.1.0 - 2026-02-16

- Add pre-commit checks and automated releases.

## 0.0.1 - 2026-02-13

- Initial release.
