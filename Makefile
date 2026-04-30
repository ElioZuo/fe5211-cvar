.PHONY: install install-vine all quick clean test sanity help

PYTHON := python

help:
	@echo "Available targets:"
	@echo "  make install       Install required dependencies"
	@echo "  make install-vine  Install optional R-Vine dependency (pyvinecopulib)"
	@echo "  make all           Run the full pipeline (1M paths, ~25 min)"
	@echo "  make quick         Run pipeline with 100k paths and 100 Tier-B boots (~3 min)"
	@echo "  make test          Run unit tests (no main-pipeline dependency)"
	@echo "  make sanity        Run strict sanity check against canonical numbers"
	@echo "  make clean         Remove cached intermediate results and outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-vine: install
	$(PYTHON) -m pip install -r requirements-vine.txt

all:
	$(PYTHON) main.py

quick:
	$(PYTHON) main.py --quick

test:
	$(PYTHON) -m pytest tests/test_unsmoothing.py tests/test_marginals.py \
	         tests/test_copula.py tests/test_simulation.py -v

sanity:
	$(PYTHON) -m pytest tests/test_numbers.py -v

clean:
	rm -rf cache/* output/figures/*.png output/figures/appendix/*.png \
	       output/tables/*.csv output/numbers.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
