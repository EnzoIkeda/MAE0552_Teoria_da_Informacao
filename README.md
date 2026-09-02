# MAE0552 — Teoria da Informação

Repositório do grupo para as atividades práticas da disciplina. Cada **proposta**
entregue ao longo do semestre vira um subprojeto independente dentro de
`projetos/`. As propostas não têm necessariamente a mesma finalidade — o que elas
compartilham é o ferramental de teoria da informação em `utils/`.

Desenvolvimento sempre em **Python + Jupyter Notebook**.

## Estrutura

```
.
├── Conteudos/                 # PDFs de aula (material de referência)
├── Propostas/                 # PDFs dos enunciados de cada proposta
├── utils/                     # código compartilhado por TODAS as propostas
│   ├── info_theory.py         #   entropia, entropia conjunta/condicional,
│   │                          #   entropia cruzada, KL, informação mútua (na mão)
│   └── simulation.py          #   amostragem de pmf, canal BSC, cadeias de Markov
├── tests/                     # testes de sanidade do utils/
└── projetos/
    ├── _template/             # esqueleto para começar uma proposta nova
    └── 01_simulacoes/         # Propostas/1_Simulacoes.pdf — 4 tópicos
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

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                    # deixa `import utils...` disponível em qualquer notebook
```

No começo de cada sessão de trabalho:

```bash
source .venv/bin/activate
git switch main && git pull         # traz o que os colegas subiram
git switch -c minha-tarefa          # trabalha sempre em um branch, nunca direto na main
jupyter lab
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
import src.model as model                     # helpers específicos deste tópico

GEN = rng(20260902)                           # seed fixo -> resultados reproduzíveis
DADOS = pathlib.Path("dados"); FIGURAS = pathlib.Path("figuras")
```

Observações:

- O diretório de trabalho do notebook é a própria pasta do tópico, então
  `import src.model`, `DADOS` e `FIGURAS` são relativos a ela.
- Com `pip install -e .` feito, a linha do `sys.path` é redundante para
  `import utils...`, mas mantê-la faz o notebook rodar mesmo sem a instalação
  editável (basta ter as dependências).
- `parents[2]` vale para a profundidade `projetos/<proposta>/<topico>/`. Se um
  notebook ficar em outra profundidade, ajuste o índice.

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
3. Registre a proposta em `projetos/README.md` e guarde o PDF em `Propostas/`.

## Testes

```bash
pytest
# ou, sem instalar pytest:
python tests/test_info_theory.py
```
