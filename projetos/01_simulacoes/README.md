# 01 — Primeiras simulações em grupo

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
| 4 | `topico4_canal_codigos/` | Capacidade de canal e códigos corretores de erro (link Terra–Marte, repetição e Hamming(7,4)) |

Cada pasta tem um notebook com **um passo do roteiro por seção**, um `src/model.py`
com o código específico do tópico e um `relatorio.md` a preencher.

Todas reaproveitam `utils/info_theory.py` e `utils/simulation.py`.
