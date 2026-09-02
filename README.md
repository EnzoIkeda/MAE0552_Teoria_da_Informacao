# MAE0552 — Teoria da Informação

[![CI](https://github.com/EnzoIkeda/MAE0552_Teoria_da_Informacao/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoIkeda/MAE0552_Teoria_da_Informacao/actions/workflows/ci.yml)

Repositório do grupo para as atividades práticas da disciplina. Cada **proposta**
entregue ao longo do semestre vira um subprojeto independente dentro de
`projetos/`. As propostas não têm necessariamente a mesma finalidade — o que elas
compartilham é o ferramental de teoria da informação em `utils/`.

Desenvolvimento sempre em **Python + Jupyter Notebook**.

## Atalhos

Os mesmos comandos em qualquer sistema operacional. Em **Linux/macOS** use
`make <alvo>`; em **Windows** (ou onde não houver `make`) use
`python scripts\dev.py <alvo>`. As duas formas rodam exatamente a mesma coisa.

| `make …` | `python scripts/dev.py …` | O que faz |
|----------|---------------------------|-----------|
| `make setup` | `... setup` | cria o `.venv`, instala tudo e ativa os hooks |
| `make test` | `... test` | roda os testes de `utils/` (`pytest`) |
| `make test-notebooks` | `... test-notebooks` | executa todos os notebooks e falha se algum quebrar (`nbmake`) |
| `make lint` | `... lint` | `ruff check` + `ruff format --check` (não altera arquivos) |
| `make format` | `... format` | formata e corrige o que der com `ruff` |
| `make check` | `... check` | `lint` + `test` (o mesmo que o CI cobra) |
| `make lab` | `... lab` | abre o JupyterLab |
| `make clean` | `... clean` | remove caches (`__pycache__`, `.pytest_cache`, checkpoints, `*.egg-info`) |

Rode o `setup` com o Python do sistema (`python3` no Linux/macOS, `python` no
Windows); os demais alvos usam o Python do `.venv` automaticamente. Para escolher
o interpretador do venv: `make setup PYTHON=python3.12` ou
`py -3.12 scripts\dev.py setup`.

## Dependências

Python **>= 3.10**. Pacotes (fixados em `requirements.txt`; os de runtime também
em `pyproject.toml`):

| Pacote | Versão | Para quê |
|--------|--------|----------|
| `numpy` | >= 1.24 | arrays, gerador aleatório, álgebra vetorial — base de tudo |
| `scipy` | >= 1.10 | conferências independentes (`scipy.stats`: `chi2`, `binom`, ...) |
| `matplotlib` | >= 3.6 | gráficos dos relatórios |
| `pandas` | >= 2.0 | organizar resultados de muitas repetições (DataFrame) |
| `jupyterlab` | >= 4.0 | ambiente dos notebooks |
| `ipykernel` | >= 6.29 | kernel dos notebooks (usado também pelo `nbmake` no CI) |
| `pytest` | >= 7.0 | testes de `utils/` |
| `nbmake` | >= 1.5 | executar os notebooks de ponta a ponta no CI |
| `ruff` | >= 0.6 | lint + formatação do código Python |
| `pre-commit` | >= 3.5 | hooks de commit (limpa notebook, roda `ruff`) |

Instalação em **Para começar a trabalhar**.

## Estrutura

```
.
├── conteudos/                 # PDFs de aula (material de referência)
├── propostas/                 # PDFs dos enunciados de cada proposta
├── utils/                     # código compartilhado por TODAS as propostas
│   ├── info_theory.py         #   entropia, conjunta/condicional, cruzada, KL, I(X;Y) (na mão)
│   ├── simulation.py          #   amostragem de pmf, canal BSC, cadeias de Markov
│   ├── estimation.py          #   bootstrap: estimativa + intervalo de confiança
│   └── plotting.py            #   estilo padrão dos gráficos e salvamento de figuras
├── tests/                     # testes de sanidade do utils/
├── scripts/dev.py             # comandos de desenvolvimento multiplataforma
├── Makefile                   # atalhos (Linux/macOS) que chamam scripts/dev.py
├── .pre-commit-config.yaml    # hooks: nbstripout, ruff, sem commit direto na main
├── .github/workflows/ci.yml   # CI: lint + testes + execução dos notebooks
└── projetos/
    ├── _template/             # esqueleto para começar uma proposta nova
    └── 01_simulacoes/         # propostas/1_Simulacoes.pdf — 4 tópicos
        ├── topico1_entropia_mi/
        ├── topico2_dpi_markov/
        ├── topico3_aep_kelly/
        └── topico4_canal_codigos/
```

Cada tópico (ou entrega) tem sempre a mesma forma:

```
topicoN_xxx/
├── <topicoN_xxx>.ipynb    # notebook principal, um passo do roteiro por seção
├── src/
│   ├── __init__.py
│   └── model.py           # código específico do tópico (modelo, codec, ...)
├── dados/                 # amostras/resultados gerados (não versionado)
├── figuras/               # gráficos exportados (não versionado)
└── relatorio.md           # relatório curto pedido no enunciado
```

## Para começar a trabalhar

Uma vez por máquina (clonar e preparar o ambiente):

```bash
git clone <url-do-repositorio>
cd MAE0552_Teoria_da_Informacao
```

Depois, **um único comando** cria o `.venv`, instala tudo e ativa os hooks:

| Sistema | Comando |
|---------|---------|
| Linux/macOS | `make setup` |
| Windows | `python scripts\dev.py setup` |
| Qualquer (sem `make`) | `python3 scripts/dev.py setup` |

<details>
<summary>Passo a passo manual (se preferir não usar o script)</summary>

```bash
# 1) criar e ativar o ambiente virtual
python3 -m venv .venv                 # Windows: py -3 -m venv .venv
source .venv/bin/activate             # bash/zsh (Linux/macOS)
# .venv\Scripts\Activate.ps1          # PowerShell (Windows)
# .venv\Scripts\activate.bat          # cmd.exe (Windows)

# 2) instalar dependências + o pacote utils + os hooks de commit
pip install -r requirements.txt
pip install -e .
pre-commit install
```
</details>

> Ubuntu/Debian: se `python3 -m venv` reclamar de `ensurepip`, instale o pacote
> do venv — `sudo apt install python3-venv` — e rode de novo.
> Windows: instale o Python de python.org (marque "Add python.exe to PATH") ou
> pela Microsoft Store; use `python` / `py -3`, não `python3`.

No começo de cada sessão de trabalho (ative o `.venv` conforme o seu shell):

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
git switch main && git pull          # traz o que os colegas subiram
git switch -c minha-tarefa           # trabalha sempre em um branch, nunca direto na main
jupyter lab                          # ou: make lab
```

Ao terminar um trecho de trabalho: `git add` / `git commit` / `git push`
(ver **Git — fluxo básico**) e abrir um Pull/Merge Request para integrar na `main`.

## Início dos notebooks

Todo notebook fica em `projetos/<proposta>/<topico>/` e começa com **uma célula
de preparação** que coloca a raiz do repositório no `sys.path` e importa o
ferramental compartilhado. O esqueleto em `projetos/_template/` já vem com ela:

```python
import sys, pathlib

REPO = pathlib.Path.cwd().parents[2]          # .../<topico>/ -> .../<proposta>/ -> raiz
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib.pyplot as plt

from utils.info_theory import (
    entropy, binary_entropy, joint_entropy, conditional_entropy,
    cross_entropy, kl_divergence, mutual_information,
    conditional_mutual_information, mi_gtest_statistic,
)
from utils.simulation import (
    rng, sample_pmf, sample_joint, empirical_pmf, bsc, markov_chain,
)
from utils.estimation import bootstrap_ci
from utils.plotting import setup_estilo, salvar_fig
import src.model as model                     # helpers específicos deste tópico

setup_estilo()
GEN = rng(20260902)                           # seed fixo -> resultados reproduzíveis
DADOS = pathlib.Path("dados")
FIGURAS = pathlib.Path("figuras")
```

Observações:

- O diretório de trabalho do notebook é a própria pasta do tópico, então
  `import src.model`, `DADOS` e `FIGURAS` são relativos a ela.
- Com `pip install -e .` feito, a linha do `sys.path` é redundante para
  `import utils...`, mas mantê-la faz o notebook rodar mesmo sem a instalação
  editável (basta ter as dependências).
- `parents[2]` vale para a profundidade `projetos/<proposta>/<topico>/`. Se um
  notebook ficar em outra profundidade, ajuste o índice.
- `setup_estilo()` padroniza a aparência dos gráficos; `salvar_fig("nome")` grava
  em `figuras/nome.png`; `bootstrap_ci(dados, estatística)` devolve
  `(estimativa, limite_inf, limite_sup)` para as barras de erro das curvas.
- **Não commite notebook com saída pesada nas células** — o hook `nbstripout`
  (via `pre-commit`) limpa isso automaticamente no commit.

## Git — fluxo básico

**Atualizar o repositório local** (fazer antes de começar e antes de abrir um PR):

```bash
git switch main
git pull
```

**Criar e usar um branch** (uma tarefa por branch; não commitar direto na `main`):

```bash
git switch -c topico1-estimativas     # cria o branch e já muda para ele
git branch                            # lista os branches locais (o atual vem marcado)
git switch main                       # volta para a main
git switch topico1-estimativas        # volta para o branch de trabalho
```

**Registrar mudanças**:

```bash
git status                            # o que mudou
git add projetos/01_simulacoes/topico1_entropia_mi
git commit -m "Tópico 1: estimativas de H e I vs N"
```

**Enviar para o remoto**:

```bash
git push -u origin topico1-estimativas   # só na 1ª vez do branch
git push                                 # nas vezes seguintes
```

Depois, abrir um Pull/Merge Request no serviço de hospedagem para revisar e
integrar o branch na `main`.

**Trazer para o seu branch o que já entrou na `main`**:

```bash
git switch main && git pull
git switch topico1-estimativas
git merge main                        # resolve conflitos, se houver, e commita
```

Artefatos gerados (`dados/`, `figuras/`, `.venv/`, checkpoints) já são ignorados
pelo `.gitignore` — não precisam entrar nos commits.

## Qualidade de código

- **`ruff`** cuida de lint e formatação de `utils/`, `tests/` e `src/`
  (configuração em `pyproject.toml`; notebooks e `.md` ficam de fora). Rode
  `make format` antes de commitar, ou deixe o `pre-commit` fazer.
- **`pre-commit`** roda a cada `git commit` (depois de `pre-commit install`, feito
  pelo `setup`): `nbstripout` limpa a saída das células dos notebooks — evita
  diffs enormes e conflitos de merge —, e `ruff` corrige/formata o código. Se um
  hook alterar arquivos, o commit é abortado: revise, dê `git add` de novo e
  recomite. Todos os hooks são multiplataforma (Windows/Linux/macOS).
- `.editorconfig` mantém indentação e fim de linha consistentes entre editores.

### Proteção da `main`

Todo trabalho entra na `main` por **Pull Request**, nunca por commit direto.

O hook `no-commit-to-branch` (em `.pre-commit-config.yaml`) recusa `git commit`
feito com a `main` em checkout — força criar um branch antes. É **local**: só
vale para quem rodou o `setup`/`pre-commit install`, e pode ser contornado com
`git commit --no-verify` numa emergência.

> Proteção real no servidor (impedir o merge sem PR para todos, inclusive pela
> interface web) depende de `Settings → Branches`/`Rules` no GitHub, que exige
> repositório **público** ou plano **GitHub Pro** — indisponível enquanto o repo
> for privado no plano gratuito. Se isso mudar, aplicar a regra "Require a pull
> request before merging" na branch `main`.

## Convenções

- **Implementar a quantidade de interesse "na mão" antes de usar biblioteca.**
  Entropia, informação mútua, divergência etc. já estão em `utils/info_theory.py`,
  escritas a partir da definição. Funções prontas (`scipy.stats.chi2_contingency`,
  `scipy.stats.entropy`, ...) entram só como **conferência independente** de um
  resultado já calculado, ou em tarefas auxiliares que não são o núcleo do
  exercício.
- **Reaproveitar `utils/`.** As mesmas quantidades reaparecem nos quatro tópicos;
  não recopiar implementação.
- **Um `relatorio.md` por tópico**, descrevendo o modelo definido, o que foi
  feito, os resultados numéricos/gráficos e se a teoria se confirmou.
- Seed fixo por notebook (`from utils.simulation import rng; GEN = rng(...)`)
  para resultados reproduzíveis.
- Cuidado com viés de amostra finita ao interpretar informação mútua estimada
  (amostras pequenas superestimam I; o alfabeto também influencia).

## Como adicionar uma proposta nova

1. Copie o esqueleto:
   ```bash
   cp -r projetos/_template projetos/0N_nome_curto
   ```
2. Renomeie/duplique as pastas de tópico conforme o enunciado e ajuste os
   `relatorio.md`.
3. Registre a proposta em `projetos/README.md` e guarde o PDF em `propostas/`.

## Testes

```bash
pytest                              # testes de utils/  (make test)
pytest --nbmake projetos/           # executa os notebooks de ponta a ponta  (make test-notebooks)

# sem depender de pytest:
python tests/test_info_theory.py
python tests/test_estimation.py
```

O **GitHub Actions** (`.github/workflows/ci.yml`) roda a cada `push` e a cada
Pull Request, em três jobs:

| Job | O que verifica |
|-----|----------------|
| `lint` | `ruff check` + `ruff format --check` |
| `testes` | `pytest` em Python 3.10, 3.11 e 3.12 |
| `notebooks` | `pytest --nbmake` — todos os notebooks executam sem erro |

O status aparece no badge no topo deste README e na aba **Actions**; um PR com o
CI vermelho não deve ser integrado à `main`.
