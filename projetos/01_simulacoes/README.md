# 01: Primeiras simulações em grupo

Enunciado: `propostas/1_Simulacoes.pdf`.

Quatro atividades de simulação. Cada uma gira em torno de um problema concreto:
definir um modelo probabilístico, gerar dados segundo esse modelo, estimar
quantidades de teoria da informação "na mão" a partir dos dados e comparar com
os valores teóricos.

| Tópico | Pasta | Assunto |
|--------|-------|---------|
| 1 | `topico1_entropia_mi/` | Entropia, informação mútua e teste de independência (teste rápido de dengue) |
| 2 | `topico2_dpi_markov/` | Desigualdade de processamento de dados e cadeias de Markov (telefone sem fio, colapso de modelo) |
| 3 | `topico3_aep_kelly/` | Compressão, AEP e critério de Kelly (apostas / crescimento de patrimônio) |
| 4 | `topico4_canal_codigos/` | Capacidade de canal e códigos corretores de erro (link Terra-Marte, repetição e Hamming(7,4)) |

Cada pasta tem um notebook com **um passo do roteiro por seção** e um
`src/model.py` com o código específico do tópico. Todas reaproveitam
`utils/info_theory.py` e `utils/simulation.py`.

## Relatórios

O relatório de entrega é o **consolidado**, que reúne os quatro tópicos com
estrutura única, tabelas e figuras numeradas por tópico e uma conclusão geral:

- `relatorio.md`: versão em Markdown (renderiza no GitHub, com as figuras);
- `relatorio.pdf`: versão em PDF, gerada a partir do `.md` com
  `scripts/relatorio_pdf.py` (ver **Relatórios em PDF** no README da raiz).

Os relatórios individuais continuam nas pastas dos tópicos:
`topico1_entropia_mi/relatorio.md`, `topico2_dpi_markov/relatorio.md`,
`topico3_aep_kelly/Relatorio_Topico_3.md` e
`topico4_canal_codigos/Relatorio_Topico_4.md`.

Todos os números dos relatórios saem da execução dos notebooks (semente fixa).
Para regenerar as figuras e o PDF:

```bash
make test-notebooks      # executa os quatro notebooks e grava as figuras
.venv/bin/python scripts/relatorio_pdf.py projetos/01_simulacoes/relatorio.md \
    -o "$PWD/projetos/01_simulacoes/relatorio.pdf"
```
