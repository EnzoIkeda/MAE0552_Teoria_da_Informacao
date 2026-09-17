"""Geradores e canais usados nas simulacoes.

Sao utilidades de amostragem/simulacao. Nada aqui calcula uma quantidade de
teoria da informacao -- essas ficam em `utils.info_theory` e sao implementadas
do zero.
"""

from __future__ import annotations

import numpy as np


def rng(seed=None):
    """Gerador de numeros aleatorios do numpy (use um seed fixo por notebook)."""
    return np.random.default_rng(seed)


def sample_pmf(p, size, generator=None):
    """Amostra `size` inteiros em {0, ..., k-1} segundo a pmf `p`."""
    p = np.asarray(p, dtype=float).ravel()
    p = p / p.sum()
    g = generator if generator is not None else np.random.default_rng()
    return g.choice(p.size, size=size, p=p)


def sample_joint(joint, size, generator=None):
    """Amostra `size` pares (x, y) segundo a tabela conjunta `joint` (2D).

    Retorna dois arrays de inteiros (x, y). Implementado achatando a tabela num
    unico vetor de probabilidades e reconvertendo o indice sorteado em (linha,
    coluna).
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint precisa ser uma tabela 2D")
    n_cols = joint.shape[1]
    flat = (joint / joint.sum()).ravel()
    g = generator if generator is not None else np.random.default_rng()
    idx = g.choice(flat.size, size=size, p=flat)
    return idx // n_cols, idx % n_cols


def empirical_pmf(samples, size=None):
    """Frequencia relativa de `samples` (inteiros >= 0) como vetor de probabilidades."""
    samples = np.asarray(samples).astype(int).ravel()
    size = int(samples.max()) + 1 if size is None else int(size)
    counts = np.bincount(samples, minlength=size).astype(float)
    return counts / counts.sum()


def bsc(bits, p, generator=None):
    """Canal binario simetrico: inverte cada bit de forma independente com prob. p.

    `bits` e um array de 0/1 de qualquer formato; o retorno tem o mesmo formato.
    """
    bits = np.asarray(bits).astype(int)
    g = generator if generator is not None else np.random.default_rng()
    flips = (g.random(bits.shape) < p).astype(int)
    return np.bitwise_xor(bits, flips)


def markov_chain(P, n, generator=None, x0=None):
    """Sequencia de `n` estados de uma cadeia de Markov com matriz de transicao `P`.

    `P[i, j] = P(proximo = j | atual = i)`. Se `x0` for None, o estado inicial e
    sorteado uniformemente. Laco explicito -- transparente e facil de depurar
    para poucos estados.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P precisa ser uma matriz quadrada")
    k = P.shape[0]
    g = generator if generator is not None else np.random.default_rng()
    states = np.empty(n, dtype=int)
    states[0] = g.integers(k) if x0 is None else int(x0)
    for t in range(1, n):
        states[t] = g.choice(k, p=P[states[t - 1]])
    return states


def stationary_distribution(P):
    """Distribuicao estacionaria pi de uma cadeia de Markov: pi P = pi, sum(pi) = 1.

    Resolve o sistema linear diretamente (transposta menos identidade, mais a
    linha da normalizacao) em vez de iterar a cadeia ate convergir: para poucos
    estados e exato e nao depende de escolher um criterio de parada.

    Iniciar `markov_chain` em um estado sorteado por essa distribuicao deixa o
    processo estacionario desde o primeiro passo. Sem isso, H(X_1) reflete a
    condicao inicial arbitraria e nao o regime da cadeia, o que distorce
    qualquer comparacao entre o custo de transmitir a sequencia e n * H(X_1).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P precisa ser uma matriz quadrada")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("cada linha de P precisa somar 1")
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.concatenate([np.zeros(k), [1.0]])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.any(pi < -1e-9):
        raise ValueError("cadeia sem distribuicao estacionaria nao negativa unica")
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()
