# Agent skill

BenchBro ships an agent skill with the Python package. It gives compatible coding
agents version-matched guidance for defining benchmarks, choosing measurement
settings, managing baselines, interpreting results, and using advanced features.

## Install it in a project

Install BenchBro first, then run Library Skills from the project root:

```console
$ uv add --dev benchbro
$ uvx library-skills --skill use-benchbro --yes
```

This discovers the skill embedded in the installed BenchBro distribution and links
it into the project's `.agents/skills/` directory. Because the link points at the
installed package, upgrading BenchBro also updates the skill content.

Run the interactive command when you want to choose from every skill exposed by a
project's dependencies:

```console
$ uvx library-skills
```

Use `uvx library-skills --check` to verify managed links without changing them.

## Invoke it

Compatible agents can select the skill automatically when a task imports or uses
BenchBro. Invoke it explicitly when you want to ensure the guidance is loaded:

```text
$use-benchbro add a parameterized benchmark for this parser and create a local baseline
```

The skill treats the installed BenchBro API and CLI as the final source of truth,
so it remains useful in editable installations and when working across versions.

## What is included

- the common `Case`, input, system, parameterization, discovery, and run workflow;
- reliable-measurement and confidence-aware comparison guidance;
- local, named, and strict CI baseline behavior;
- subprocess, isolation, timeout, CPU-affinity, and profiling patterns;
- pytest and programmatic API integration guidance;
- common failure modes that can invalidate a benchmark or comparison.

The source lives in `.library-skills/use-benchbro/`. Package builds map it to
`benchbro/.agents/skills/use-benchbro/` inside the wheel, which is the location
Library Skills discovers after installation.
