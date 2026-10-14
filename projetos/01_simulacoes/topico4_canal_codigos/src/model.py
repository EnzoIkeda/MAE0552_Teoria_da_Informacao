"""Tópico 4: canal binário simétrico e códigos corretores de erro.

Canal: `utils.simulation.bsc`. Capacidade C = 1 - H_b(p) via
`utils.info_theory.binary_entropy`.

Aqui ficam os dois codecs pedidos no roteiro: código de repetição (n_rep cópias
+ voto de maioria) e Hamming(7,4) com decodificação por síndrome. Toda a
aritmética do Hamming é módulo 2.
"""

from __future__ import annotations

import numpy as np


def _como_inteiro(bits: np.ndarray) -> np.ndarray:
    """`asarray` que preserva um dtype inteiro ja existente (ex.: uint8).

    So cai para `uint8` quando a entrada nao e inteira (lista de bool, por
    exemplo). Evita promover mensagens longas para o inteiro de 8 bytes
    padrao do numpy quando o chamador ja passou algo mais compacto.
    """
    bits = np.asarray(bits)
    if not np.issubdtype(bits.dtype, np.integer):
        bits = bits.astype(np.uint8)
    return bits


# ---------------------------------------------------------------- repetição ---


def repetition_encode(bits: np.ndarray, n_rep: int) -> np.ndarray:
    """Cada bit vira `n_rep` cópias (use `n_rep` ímpar para não ter empate)."""
    return np.repeat(_como_inteiro(bits), n_rep)


def repetition_decode(received: np.ndarray, n_rep: int) -> np.ndarray:
    """Voto de maioria em blocos de `n_rep` bits."""
    blocks = _como_inteiro(received).reshape(-1, n_rep)
    # soma em int64 mesmo com blocks em uint8: n_rep cabe sem estourar e evita
    # que o acumulador da soma sature em blocos muito longos.
    maioria = blocks.sum(axis=1, dtype=np.int64) > n_rep / 2
    return maioria.astype(np.uint8)


# ------------------------------------------------------------- Hamming(7,4) ---
# Forma padrão G = [I4 | P], H = [P^T | I3]; bit i do código = coluna i.
G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """Codifica múltiplos de 4 bits; retorna blocos de 7 bits concatenados."""
    m = _como_inteiro(bits).reshape(-1, 4)
    return ((m @ G) % 2).ravel()


# cada coluna de H e um padrao de 3 bits distinto e nao nulo -> a sindrome,
# lida como numero de 0 a 7, aponta direto para a posicao a corrigir. Tabela
# fixa de 8 entradas (-1 = sindrome 0, sem erro) monta essa busca uma vez so.
_PESOS_SINDROME = np.array([4, 2, 1])
_COLUNA_POR_SINDROME = np.full(8, -1, dtype=int)
for _i in range(7):
    _COLUNA_POR_SINDROME[H[:, _i] @ _PESOS_SINDROME] = _i


def hamming74_decode(received: np.ndarray) -> np.ndarray:
    """Decodifica blocos de 7 bits corrigindo até 1 erro por bloco (síndrome).

    Vetorizado: a sindrome de cada bloco vira um indice em 0..7 que consulta
    `_COLUNA_POR_SINDROME` em vez de percorrer os blocos em um laco Python.
    """
    r = _como_inteiro(received).reshape(-1, 7).copy()
    syndromes = (r @ H.T) % 2  # (n_blocos, 3)
    indices = syndromes @ _PESOS_SINDROME  # (n_blocos,), cada um em 0..7
    colunas = _COLUNA_POR_SINDROME[indices]  # -1 onde a sindrome e 0
    linhas = np.nonzero(colunas >= 0)[0]
    r[linhas, colunas[linhas]] ^= 1
    return r[:, :4].ravel()
