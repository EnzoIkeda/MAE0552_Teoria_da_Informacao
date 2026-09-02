"""Codigo compartilhado entre todas as propostas/topicos.

- `info_theory`: entropia, entropia conjunta/condicional, entropia cruzada,
  divergencia de Kullback-Leibler e informacao mutua, implementadas do zero.
- `simulation`: geradores e canais usados nas simulacoes (amostragem de pmf,
  canal binario simetrico, cadeias de Markov).
"""

from . import info_theory, simulation

__all__ = ["info_theory", "simulation"]
