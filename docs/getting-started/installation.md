# Installation

BenchBro supports Python 3.10 through 3.14.

=== "Project development dependency"

    Add BenchBro to a project that uses uv:

    ```console
    $ uv add --dev benchbro
    $ uv run benchbro --help
    ```

=== "Global CLI"

    Install the command once and run it directly:

    ```console
    $ uv tool install benchbro
    $ benchbro --help
    ```

=== "Pytest integration"

    Include the optional pytest extra:

    ```console
    $ uv add --dev 'benchbro[pytest]'
    ```

## Development checkout

When working on BenchBro itself, install every dependency group from the repository
root:

```console
$ uv sync --all-groups
$ uv run benchbro --help
```

The package exposes both `benchbro` and `python -m benchbro`. All examples in this
site use `uv run benchbro`; omit `uv run` when using the global tool installation.

!!! tip
    Run `benchbro init` in a project to create a starter `benchbro.toml` and a
    parameterized benchmark module.

Continue with the [quick start](quick-start.md).
