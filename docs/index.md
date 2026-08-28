---
hide:
  - navigation
---

<div class="bb-hero" markdown>
<div markdown>

# Benchmarks you can trust

BenchBro is a Python library and CLI for repeatable performance experiments.
Define benchmarks beside your code, measure them with noise-aware statistics,
and catch regressions against environment-aware baselines.

<div class="bb-actions" markdown>
[Get started](getting-started/quick-start.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/peter-daly/benchbro){ .md-button }
</div>

</div>

![BenchBro mascot lifting a barbell](assets/images/mascot-full-bright.png)
</div>

<div class="grid cards" markdown>

- :material-chart-bell-curve-cumulative: **Noise-aware by default**

    <span class="bb-metric">95% CI</span>

    Adaptive sampling, confidence intervals, outlier detection, and noisy-result
    classification help separate meaningful changes from ordinary variance.

- :material-source-branch-check: **Regression ready**

    <span class="bb-metric">0 · 1 · 2 · 3</span>

    Compare named baselines locally or in automation, with distinct exit codes for
    regressions, setup errors, and benchmark failures.

- :material-flask-outline: **Flexible experiments**

    <span class="bb-metric">sync + async</span>

    Parameterize cases, inject scoped systems, benchmark subprocesses, isolate
    workers, and attach built-in or custom profilers.

</div>

## A benchmark in under a minute

```python title="benchmarks/bench_hashing.py"
import hashlib

from benchbro import Case

case = Case(name="hashing", tags=["fast"])


@case.input()
def payload() -> bytes:
    return b"benchbro"


@case.benchmark()
def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
```

```console
$ uv run benchbro run
```

BenchBro discovers `benchmarks/**/*.py`, warms up the callable, collects repeat
samples, prints a rich result table, and creates a local baseline on the first
run. Start with the [installation guide](getting-started/installation.md), then
learn how to make [reliable measurements](guides/reliable-measurements.md).

## What it covers

| Capability | Built in |
| --- | --- |
| Python callables | Sync, async, parameterized, time, and memory benchmarks |
| Dependencies | Sync, async, generator, and async-generator systems with three scopes |
| External programs | Explicit-argv subprocess benchmarks with timeout and exit-code checks |
| Reproducibility | Runtime, platform, CPU, repository, lockfile, power, and runner metadata |
| Output | Terminal tables and histograms, JSON, CSV, and Markdown |
| Workflows | Named baselines, local history, configuration files, and pytest integration |
