PYTHON ?= uv run python
BENCHBRO ?= $(PYTHON) -m benchbro
EXAMPLES_MODULE ?= examples/e2e_benchmarks.py
EXAMPLES_ARTIFACTS ?= examples/artifacts

.PHONY: ruff ty deptry pytest tox ci examples examples-baseline examples-test-run examples-histogram pre-commit pre-commit-install

ruff:
	uv run ruff check .

ty:
	uv run ty check .

deptry:
	uv run deptry .

pytest:
	uv run python -m pytest

tox:
	uv run tox

ci: ruff ty deptry pytest

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
	BENCHBRO_EXAMPLE_SLOW_FACTOR=3 $(BENCHBRO) $(EXAMPLES_MODULE) \
		--output-json $(EXAMPLES_ARTIFACTS)/current.json \
		--output-csv $(EXAMPLES_ARTIFACTS)/current.csv \
		--output-md $(EXAMPLES_ARTIFACTS)/current.md

examples-histogram:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	$(BENCHBRO) $(EXAMPLES_MODULE) --histogram
