VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
BIN  := $(VENV)/bin

.PHONY: help setup test test-notebooks lint format check lab clean

help:  ## Lista os alvos disponiveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o venv e instala tudo (runtime + dev + pacote editavel + hooks)
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(BIN)/pre-commit install --hook-type pre-commit --hook-type pre-push

test:  ## Roda os testes de utils/
	$(PY) -m pytest

test-notebooks:  ## Executa os notebooks (nbmake) e falha se algum quebrar
	$(PY) -m pytest --nbmake projetos/

lint:  ## Verifica estilo e erros com ruff (sem alterar arquivos)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Formata e corrige o que der com ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

check: lint test  ## O que o CI cobra: lint + testes

lab:  ## Abre o JupyterLab
	$(BIN)/jupyter lab

clean:  ## Remove caches e artefatos locais
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf ./*.egg-info
