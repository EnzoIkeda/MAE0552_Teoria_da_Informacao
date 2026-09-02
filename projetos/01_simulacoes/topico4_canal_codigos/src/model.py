"""Tópico 4 — canal binário simétrico e códigos corretores de erro.

Canal: `utils.simulation.bsc`. Capacidade C = 1 - H_b(p) via
`utils.info_theory.binary_entropy`.

Aqui ficam os dois codecs pedidos no roteiro: código de repetição (n_rep cópias
+ voto de maioria) e Hamming(7,4) com decodificação por síndrome. Toda a
aritmética do Hamming é módulo 2.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- repetição ---


def repetition_encode(bits: np.ndarray, n_rep: int) -> np.ndarray:
    """Cada bit vira `n_rep` cópias (use `n_rep` ímpar para não ter empate)."""
    return np.repeat(np.asarray(bits).astype(int), n_rep)


def repetition_decode(received: np.ndarray, n_rep: int) -> np.ndarray:
    """Voto de maioria em blocos de `n_rep` bits."""
    blocks = np.asarray(received).astype(int).reshape(-1, n_rep)
    return (blocks.sum(axis=1) > n_rep / 2).astype(int)


# ------------------------------------------------------------- Hamming(7,4) ---
# Forma padrão G = [I4 | P], H = [P^T | I3]; bit i do código = coluna i.
G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=int,
)

H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=int,
)


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """Codifica múltiplos de 4 bits; retorna blocos de 7 bits concatenados."""
    m = np.asarray(bits).astype(int).reshape(-1, 4)
    return ((m @ G) % 2).ravel()


def hamming74_decode(received: np.ndarray) -> np.ndarray:
    """Decodifica blocos de 7 bits corrigindo até 1 erro por bloco (síndrome)."""
    r = np.asarray(received).astype(int).reshape(-1, 7).copy()
    syndromes = (r @ H.T) % 2                 # (n_blocos, 3)
    cols = H.T                                # (7, 3): coluna i de H
    for k, s in enumerate(syndromes):
        if s.any():
            match = np.where((cols == s).all(axis=1))[0]
            if match.size:
                r[k, match[0]] ^= 1
    return r[:, :4].ravel()
