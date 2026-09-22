"""Tópico 1: modelo do teste rápido de dengue.

D = estado real (0 saudável, 1 infectado); T = resultado do teste (0/1).
O grupo escolhe: prevalência P(D=1), taxa de falso positivo P(T=1|D=0) e taxa
de falso negativo P(T=0|D=1). Daí sai a conjunta p(d, t).

As quantidades de informação (H, I, H(p,q), D(p||q), teste-G) vêm de
`utils.info_theory`.
"""

from __future__ import annotations

import numpy as np


def true_joint(prevalence: float, fpr: float, fnr: float) -> np.ndarray:
    """Conjunta p(d, t) 2x2. Linhas = D (0, 1), colunas = T (0, 1)."""
    p_d = np.array([1.0 - prevalence, prevalence])
    # p(t | d): linha 0 = D=0 (falso positivo fpr); linha 1 = D=1 (falso negativo fnr)
    p_t_given_d = np.array(
        [
            [1.0 - fpr, fpr],
            [fnr, 1.0 - fnr],
        ]
    )
    return p_d[:, None] * p_t_given_d


def sample_pairs(joint: np.ndarray, n: int, generator) -> tuple[np.ndarray, np.ndarray]:
    """N pares (D, T): sorteia D pela marginal e T condicional a D."""
    p_d = joint.sum(axis=1)
    d = (generator.random(n) < p_d[1]).astype(int)
    t = sample_given_d(joint, d, generator)
    return d, t


def sample_given_d(joint: np.ndarray, d: np.ndarray, generator) -> np.ndarray:
    """T condicional a uma serie D ja dada (ex.: D_1..D_n de `markov_chain`).

    Usa a mesma conjunta `joint` so para extrair p(t | d); diferente de
    `sample_pairs`, D nao precisa ser i.i.d. -- ele vem pronto de fora.
    """
    p_d = joint.sum(axis=1)
    p_t_given_d = joint / p_d[:, None]
    d = np.asarray(d).astype(int)
    u = generator.random(d.shape)
    return (u < p_t_given_d[d, 1]).astype(int)


# --------------------------------------------------------- valores exatos ---
# Quando D segue uma cadeia de Markov e T depende so de D no mesmo dia, o par
# (D, T) e um modelo oculto de Markov: T sozinho NAO e uma cadeia de Markov.
# As funcoes abaixo calculam as quantidades da sequencia T_1..T_n de forma
# exata, para servir de valor teorico contra o qual conferir as estimativas por
# contagem (e para medir o quanto a aproximacao de ordem 1 erra).


def _p_t_given_d(joint: np.ndarray) -> np.ndarray:
    return joint / joint.sum(axis=1)[:, None]


def t_sequence_pmf(joint: np.ndarray, p_d: np.ndarray, n: int, pi0: np.ndarray) -> np.ndarray:
    """p(T_1, ..., T_n) exata, como vetor de 2^n posicoes.

    O indice de cada sequencia e o inteiro que ela representa em binario com
    T_1 como bit mais significativo -- a mesma convencao usada no notebook para
    contar as sequencias observadas, para que os dois vetores sejam comparaveis
    posicao a posicao.

    Calculada pelo algoritmo forward: `alpha[d]` acumula
    p(t_1..t_i, D_i = d), somando sobre todos os caminhos de D.
    """
    p_t_d = _p_t_given_d(joint)
    pmf = np.empty(2**n, dtype=float)
    for codigo in range(2**n):
        bits = [(codigo >> (n - 1 - i)) & 1 for i in range(n)]
        alpha = np.asarray(pi0, dtype=float) * p_t_d[:, bits[0]]
        for bit in bits[1:]:
            alpha = (alpha @ p_d) * p_t_d[:, bit]
        pmf[codigo] = alpha.sum()
    return pmf


def t_pair_joints(joint: np.ndarray, p_d: np.ndarray, n: int, pi0: np.ndarray) -> list[np.ndarray]:
    """[p(T_1, T_2), ..., p(T_(n-1), T_n)] exatas, uma tabela 2x2 por par."""
    p_t_d = _p_t_given_d(joint)
    p_d_dia = [np.asarray(pi0, dtype=float)]
    for _ in range(1, n):
        p_d_dia.append(p_d_dia[-1] @ p_d)
    pares = []
    for i in range(1, n):
        p_dd = p_d_dia[i - 1][:, None] * p_d  # p(D_(i-1), D_i)
        pares.append(np.einsum("ab,ax,by->xy", p_dd, p_t_d, p_t_d))
    return pares


def t_marginals(joint: np.ndarray, p_d: np.ndarray, n: int, pi0: np.ndarray) -> list[np.ndarray]:
    """[p(T_1), ..., p(T_n)] exatas."""
    p_t_d = _p_t_given_d(joint)
    p_d_dia = np.asarray(pi0, dtype=float)
    marginais = []
    for _ in range(n):
        marginais.append(p_d_dia @ p_t_d)
        p_d_dia = p_d_dia @ p_d
    return marginais
