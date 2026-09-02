"""Tópico 1 — modelo do teste rápido de dengue.

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
    p_t_given_d = joint / p_d[:, None]
    d = generator.random(n) < p_d[1]
    d = d.astype(int)
    u = generator.random(n)
    t = (u < p_t_given_d[d, 1]).astype(int)
    return d, t
