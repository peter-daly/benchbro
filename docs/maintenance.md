# Maintainer setup

The repository workflows are secure by default where settings can be expressed
as code. Complete the following account-level settings in GitHub and PyPI.

## GitHub

1. Enable dependency-graph, Dependabot alerts, and Dependabot security updates.
2. Protect `main` and require the `quality` and Python 3.10–3.14 `tox` checks.
3. Create a `pypi` deployment environment, restrict it to `main`, and require a
   reviewer for package publication.
4. Enable private vulnerability reporting for the process documented in
   `SECURITY.md`.

## PyPI trusted publishing

Configure a trusted publisher for:

- owner: `peter-daly`
- repository: `benchbro`
- workflow: `release.yml`
- environment: `pypi`

Then set the GitHub Actions repository variable `PYPI_TRUSTED_PUBLISHING=true`
and remove the `PYPI_TOKEN` secret. Until that variable is enabled, the release
workflow retains the existing token path so maintenance changes do not interrupt
publishing. Releases are manual: bump the project version, merge a green build to
`main`, then dispatch the `Release` workflow.
