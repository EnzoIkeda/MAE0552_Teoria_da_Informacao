"""Codigo compartilhado entre todas as propostas/topicos.

- `info_theory`: entropia, entropia conjunta/condicional, entropia cruzada,
  divergencia de Kullback-Leibler e informacao mutua, implementadas do zero.
- `simulation`: geradores e canais usados nas simulacoes (amostragem de pmf,
  canal binario simetrico, cadeias de Markov).
- `estimation`: reamostragem bootstrap para estimativa com intervalo de confianca.
- `plotting`: estilo padrao, graficos seaborn recorrentes e tabelas dos relatorios.
"""

from . import estimation, info_theory, plotting, simulation

__all__ = ["estimation", "info_theory", "plotting", "simulation"]
