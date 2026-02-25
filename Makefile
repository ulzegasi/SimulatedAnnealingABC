# Note: Run 'make env-dev' (conda) or 'make install-dev' (venv/system Python) before using any commands below.

# =============================================================================
# Environment (conda/micromamba)
# =============================================================================
.PHONY: env env-dev
env:
	micromamba create --name sabc --file environment.sabc.yml --yes
	@echo "Activate with: micromamba activate sabc"

env-dev:
	micromamba create --name sabc-dev --file environment.sabc.yml --file environment.sabc-dev.yml --yes
	micromamba run -n sabc-dev npm install -g markdownlint-cli2 markdownlint-cli2-formatter-pretty markdownlint-cli2-formatter-summarize markdown-table-formatter opencode-ai
	@echo "Activate with: micromamba activate sabc-dev"

# =============================================================================
# Installation (for venv/system Python)
# =============================================================================
.PHONY: install install-dev
install:
	pip install -e .

install-dev:
	pip install -e .[numba,viz,test]

# =============================================================================
# Linting & Formatting
# =============================================================================
.PHONY: lint lint-fix format lint-md lint-md-fix
lint:
	ruff check src/ tests/ examples/

lint-fix:
	ruff check --fix src/ tests/ examples/

format:
	ruff format src/ tests/ examples/

lint-md:
	markdownlint-cli2 README.md AGENTS.md

lint-md-fix:
	markdown-table-formatter --columnpadding 1 README.md AGENTS.md
	markdownlint-cli2 --fix README.md AGENTS.md

# =============================================================================
# Type Checking
# =============================================================================
.PHONY: typecheck
typecheck:
	ty check src/

# =============================================================================
# Testing
# =============================================================================
.PHONY: test test-all test-slow test-cov test-html
test:
	pytest tests/

test-all:
	pytest

test-slow:
	pytest -m slow

test-cov:
	pytest --cov=src --cov-report=term-missing

test-html:
	pytest --cov=src --cov-report=html

# =============================================================================
# Clean
# =============================================================================
.PHONY: clean clean-cache
clean:
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-cache:
	pip cache purge
	micromamba clean --all --force-pkgs-dirs --yes
	npm cache clean --force

# =============================================================================
# Notebooks
# =============================================================================
.PHONY: notebooks
notebooks:
	jupytext --to ipynb examples/notebooks/*.py
