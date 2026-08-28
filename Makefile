PYTHON ?= uv run python
BENCHBRO ?= uv run benchbro
EXAMPLES_MODULE ?= examples/e2e_benchmarks.py
EXAMPLES_ARTIFACTS ?= examples/artifacts

.PHONY: ruff format ty deptry pytest tox build docs docs-serve ci examples examples-baseline examples-test-run examples-histogram pre-commit pre-commit-install

ruff:
	uv run ruff check .

format:
	uv run ruff format --check .

ty:
	uv run ty check .

deptry:
	uv run deptry .

pytest:
	uv run python -m pytest

tox:
	uv run tox

build:
	uv build

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

ci: ruff format ty deptry pytest build

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install

examples: examples-baseline examples-test-run

examples-baseline:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	$(BENCHBRO) $(EXAMPLES_MODULE) \
		--new-baseline \
		--output-json $(EXAMPLES_ARTIFACTS)/baseline.json \
		--output-csv $(EXAMPLES_ARTIFACTS)/baseline.csv \
		--output-md $(EXAMPLES_ARTIFACTS)/baseline.md

examples-test-run:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	@BENCHBRO_EXAMPLE_SLOW_FACTOR=3 $(BENCHBRO) $(EXAMPLES_MODULE) \
		--output-json $(EXAMPLES_ARTIFACTS)/current.json \
		--output-csv $(EXAMPLES_ARTIFACTS)/current.csv \
		--output-md $(EXAMPLES_ARTIFACTS)/current.md; \
	status=$$?; \
	if [ $$status -ne 2 ]; then \
		echo "Expected the simulated slowdown to exit with regression status 2, got $$status" >&2; \
		exit 1; \
	fi

examples-histogram:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	$(BENCHBRO) $(EXAMPLES_MODULE) --histogram
