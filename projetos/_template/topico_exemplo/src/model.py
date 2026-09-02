"""Código específico deste tópico: definição do modelo, geradores auxiliares,
codecs etc.

As quantidades de teoria da informação (entropia, informação mútua, KL, ...)
NÃO são reimplementadas aqui — usar `utils.info_theory`. Este arquivo cuida da
parte "montar o mundo do problema".
"""

from __future__ import annotations

import numpy as np


def exemplo_modelo(parametro: float) -> np.ndarray:
    """Substituir pela construção do modelo probabilístico do tópico."""
    raise NotImplementedError("descrever e implementar o modelo desta atividade")
