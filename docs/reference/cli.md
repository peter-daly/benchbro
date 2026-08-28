# Command-line reference

```text
benchbro run [target] [options]
benchbro list [target] [--verbose]
benchbro compare BASELINE.json CURRENT.json
benchbro baseline update [target] --baseline NAME
benchbro baseline list
benchbro history list
benchbro history show SELECTOR
benchbro history compare OLDER NEWER
benchbro init [path]
benchbro completion bash|zsh|fish
```

The original `benchbro TARGET [options]` form remains supported.

## Common run options

| Option | Purpose |
| --- | --- |
| `--config PATH` | Read a specific standalone configuration file |
| `--case NAME` | Select a case; repeat to select more than one |
| `--tag TAG` | Select cases by tag; repeat to select more than one |
| `--repeats N` | Set the maximum repeat count |
| `--warmup N` | Set warmup invocations |
| `--min-iterations N` | Set invocations per repeat |
| `--adaptive` / `--fixed` | Enable or disable adaptive stopping |
| `--min-repeats N` | Require an adaptive sample floor |
| `--min-time SECONDS` | Require a minimum measurement duration |
| `--max-time SECONDS` | Cap measurement duration |
| `--target-margin PCT` | Set the target 95% CI relative margin |
| `--noise-threshold PCT` | Set the noisy-result CV threshold |
| `--isolation process` | Run each benchmark in a worker process |
| `--cpu-affinity 2,3` | Pin measurement to Linux CPU IDs |
| `--timeout SECONDS` | Set a benchmark timeout |
| `--profile NAME` | Attach a registered profiler |
| `--histogram` | Draw terminal histograms for time benchmarks |

## Outputs and comparison

| Option | Purpose |
| --- | --- |
| `--output-json PATH` | Write a schema-versioned JSON result |
| `--output-csv PATH` | Write long-form metric rows |
| `--output-md PATH` | Write a Markdown summary table |
| `--baseline NAME` | Select a named baseline |
| `--new-baseline` | Replace the selected baseline |
| `--save-history` | Retain the run in local history |
| `--no-compare` | Skip comparison |
| `--ci` | Use the strict committed CI baseline |
| `--allow-environment-mismatch` | Permit an intentional cross-environment comparison |

Run `benchbro COMMAND --help` for the complete options accepted by one command.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Successful or non-regressing run |
| `1` | Invalid input, discovery, configuration, or comparison setup |
| `2` | Statistically supported regression |
| `3` | Benchmark execution failure |

`LIKELY` and `INCONCLUSIVE` threshold crossings do not return exit code 2.

## Shell completion

Generate a script for Bash, Zsh, or Fish:

```console
$ uv run benchbro completion zsh > ~/.zfunc/_benchbro
```

Load it using the completion mechanism for your shell.
