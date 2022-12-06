.PHONY: benchmark
benchmark:
	pytest --benchmark-sort=mean torchmps/tests/benchmarks

.PHONY: check-format
check-format:
	black --check --diff torchmps/

.PHONY: check-style
check-style:
	flake8

.PHONY: clean
clean:
	rm -rf torchmps.egg-info
	rm -rf .pytest_cache/

.PHONY: dev-requirements
dev-requirements: dev_requirements.txt
	pip install -r dev_requirements.txt

.PHONY: dist
dist:
	python setup.py

.PHONY: docs
docs:
	make -C docs html

.PHONY: format
format:
	black torchmps/

.PHONY: install
install:
	pip install -e .

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: test

test:
	pytest -x --ignore=torchmps/tests/benchmarks torchmps/tests
.PHONY: test-report
test-report:
	pytest --cov-report term-missing --cov=torchmps torchmps/tests
