.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init:
	@python3 -m venv .enterprise --prompt enterprise
	@./.enterprise/bin/python3 -m pip install -U pip setuptools wheel
	@./.enterprise/bin/python3 -m pip install -r requirements.txt -U
	@./.enterprise/bin/python3 -m pip install -r requirements_dev.txt -U
	@./.enterprise/bin/python3 -m pre_commit install --install-hooks --overwrite
	@./.enterprise/bin/python3 -m pip install -e .
	@echo "run source .enterprise/bin/activate to activate environment"


format:
	black .

lint:
	black --check .
	flake8 .

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -rf coverage.xml

COV_COVERAGE_PERCENT ?= 85
test: lint ## run tests quickly with the default Python
	pytest -v --durations=10 --full-trace --cov-report html --cov-report xml \
		--cov-config .coveragerc --cov-fail-under=$(COV_COVERAGE_PERCENT) \
		--cov=enterprise tests

coverage: test ## check code coverage quickly with the default Python
	$(BROWSER) htmlcov/index.html

jupyter-docs: ## biuld jupyter notebook docs
	jupyter nbconvert --template docs/nb-rst.tpl --to rst docs/_static/notebooks/*.ipynb --output-dir docs/
	cp -r docs/_static/notebooks/img docs/

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/enterprise*.rst
	rm -f docs/modules.rst
	rm -rf docs/_build
	sphinx-apidoc -o docs/ -M enterprise
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m build --sdist --wheel
	ls -l dist
