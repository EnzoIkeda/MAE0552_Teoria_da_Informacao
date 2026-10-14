"""Tópico 3: apostas, AEP e critério de Kelly.

m resultados por partida, probabilidades verdadeiras p_i, cotações o_i ("o_i
para 1"). Estratégia = vetor b (fração do patrimônio em cada resultado,
sum b_i = 1). O patrimônio evolui como S_n = S_0 prod_j b_{X_j} o_{X_j}.

Trabalhar SEMPRE em log: somar log2(b_{X_j} o_{X_j}) evita overflow/underflow.
A taxa de crescimento W(b, p) = sum_i p_i log2(b_i o_i) e a divergência D(p||q)
para o Passo 11 saem de `utils.info_theory` / da média empírica dos incrementos.
"""

from __future__ import annotations

import heapq

import numpy as np


def kelly_fractions(p: np.ndarray) -> np.ndarray:
    """b* = p (aposta proporcional às probabilidades verdadeiras)."""
    p = np.asarray(p, dtype=float)
    return p / p.sum()


def uniform_fractions(m: int) -> np.ndarray:
    return np.full(m, 1.0 / m)


def all_in_on_favorite(p: np.ndarray) -> np.ndarray:
    b = np.zeros_like(np.asarray(p, dtype=float))
    b[int(np.argmax(p))] = 1.0
    return b


def log_growth_increments(outcomes: np.ndarray, b: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """log2(b_{x} o_{x}) para cada rodada `x` em `outcomes`.

    Somar cumulativamente dá log2(S_n / S_0); a média dá a taxa de crescimento
    empírica, a ser comparada com W(b, p).
    """
    b = np.asarray(b, dtype=float)
    odds = np.asarray(odds, dtype=float)
    x = np.asarray(outcomes).astype(int)
    with np.errstate(divide="ignore"):
        return np.log2(b[x] * odds[x])  # -inf se b_x = 0 e o resultado x sai


def theoretical_growth_rate(b: np.ndarray, p: np.ndarray, odds: np.ndarray) -> float:
    """W(b, p) = sum_i p_i log2(b_i o_i); -inf se b_i = 0 em resultado com p_i > 0."""
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore"):
        termos = np.where(p > 0, p * np.log2(b * odds), 0.0)
    return float(termos.sum())


def huffman_lengths(p: np.ndarray) -> np.ndarray:
    """Comprimentos das palavras de um codigo de Huffman binario para a pmf `p`.

    Funde repetidamente os dois nos menos provaveis; cada simbolo ganha um bit
    a cada fusao de que participa. Serve para conferir H(p) <= L < H(p) + 1 sem
    recorrer a uma biblioteca de codificacao.
    """
    p = np.asarray(p, dtype=float)
    comprimentos = np.zeros(p.size, dtype=int)
    if p.size == 1:
        comprimentos[0] = 1
        return comprimentos
    heap = [(float(pi), i, [i]) for i, pi in enumerate(p)]
    heapq.heapify(heap)
    proximo_id = p.size
    while len(heap) > 1:
        pa, _, simbolos_a = heapq.heappop(heap)
        pb, _, simbolos_b = heapq.heappop(heap)
        for s in simbolos_a + simbolos_b:
            comprimentos[s] += 1
        heapq.heappush(heap, (pa + pb, proximo_id, simbolos_a + simbolos_b))
        proximo_id += 1
    return comprimentos
