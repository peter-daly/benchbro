PYTHON ?= uv run python
BENCHBRO ?= $(PYTHON) -m benchbro
EXAMPLES_MODULE ?= examples/e2e_benchmarks.py
EXAMPLES_ARTIFACTS ?= examples/artifacts

.PHONY: examples examples-baseline examples-candidate examples-compare

examples: examples-baseline examples-candidate examples-compare

examples-baseline:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	$(BENCHBRO) run $(EXAMPLES_MODULE) \
		--output-json $(EXAMPLES_ARTIFACTS)/baseline.json \
		--output-csv $(EXAMPLES_ARTIFACTS)/baseline.csv
		--output-md $(EXAMPLES_ARTIFACTS)/baseline.md

examples-candidate:
	@mkdir -p $(EXAMPLES_ARTIFACTS)
	BENCHBRO_EXAMPLE_SLOW_FACTOR=3 $(BENCHBRO) run $(EXAMPLES_MODULE) \
		--output-json $(EXAMPLES_ARTIFACTS)/current.json \
		--output-csv $(EXAMPLES_ARTIFACTS)/current.csv \
		--output-md $(EXAMPLES_ARTIFACTS)/current.md
examples-compare:
	-$(BENCHBRO) compare \
		$(EXAMPLES_ARTIFACTS)/baseline.json \
		$(EXAMPLES_ARTIFACTS)/current.json \
		--fail-on-regression 5
