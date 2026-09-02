# Atalhos para Linux/macOS. Cada alvo delega para scripts/dev.py (mesma logica
# usada no Windows). PYTHON = interpretador base p/ criar o venv no `setup`
# (Ubuntu nao tem `python`, so `python3`); sobrescreva com: make setup PYTHON=python3.12
PYTHON ?= python3
DEV    := $(PYTHON) scripts/dev.py

.PHONY: help setup test test-notebooks lint format check lab clean

help:  ## Lista os alvos disponiveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o venv, instala tudo (runtime + dev + pacote editavel) e ativa os hooks
	$(DEV) setup

test:  ## Roda os testes de utils/
	$(DEV) test

test-notebooks:  ## Executa os notebooks (nbmake) e falha se algum quebrar
	$(DEV) test-notebooks

lint:  ## Verifica estilo e erros com ruff (sem alterar arquivos)
	$(DEV) lint

format:  ## Formata e corrige o que der com ruff
	$(DEV) format

check:  ## O que o CI cobra: lint + testes
	$(DEV) check

lab:  ## Abre o JupyterLab
	$(DEV) lab

clean:  ## Remove caches e artefatos locais
	$(DEV) clean
