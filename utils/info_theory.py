"""Funcoes de teoria da informacao implementadas do zero.

Todas as quantidades sao medidas em bits (logaritmo na base 2). As funcoes
recebem vetores/matrizes de probabilidades ou de contagens; `numpy` e usado
apenas para algebra de vetores e para o logaritmo, nunca para a quantidade de
interesse em si (entropia, informacao mutua, divergencia).

A convencao 0 * log2(0) = 0 e aplicada em todos os somatorios, coerente com
lim_{p->0+} p log p = 0.

Convencao de eixos para tabelas conjuntas 2D: eixo 0 = X (linhas),
eixo 1 = Y (colunas).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_distribution",
    "counts_to_joint",
    "entropy",
    "binary_entropy",
    "joint_entropy",
    "conditional_entropy",
    "cross_entropy",
    "kl_divergence",
    "mutual_information",
    "conditional_mutual_information",
    "mi_gtest_statistic",
]

_EPS = 1e-12


def _xlog2x(x):
    """x * log2(x) elemento a elemento, com a convencao 0 * log2(0) = 0."""
    x = np.asarray(x, dtype=float)
    safe = np.where(x > 0.0, x, 1.0)
    return np.where(x > 0.0, x * np.log2(safe), 0.0)


def to_distribution(p, axis=None):
    """Normaliza contagens ou pesos nao negativos para uma distribuicao de probabilidade.

    Com `axis=None` (padrao) normaliza pelo total do array inteiro; com um eixo,
    normaliza ao longo desse eixo (util para obter p(y | x) de uma tabela conjunta).
    """
    p = np.asarray(p, dtype=float)
    if np.any(p < -_EPS):
        raise ValueError("probabilidades/contagens negativas nao sao permitidas")
    p = np.clip(p, 0.0, None)
    total = p.sum(axis=axis, keepdims=axis is not None)
    if np.any(total <= 0):
        raise ValueError("soma nula: nao e possivel normalizar")
    return p / total


def counts_to_joint(x, y, size_x=None, size_y=None):
    """Tabela de contagens conjuntas N[i, j] = #{amostras com x=i, y=j}.

    `x` e `y` sao vetores de inteiros nao negativos (rotulos do alfabeto). Passe
    `size_x`/`size_y` para fixar o tamanho do alfabeto quando algum simbolo puder
    nao aparecer na amostra.
    """
    x = np.asarray(x).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    if x.shape != y.shape:
        raise ValueError("x e y precisam ter o mesmo numero de amostras")
    if x.size == 0:
        raise ValueError("amostra vazia")
    size_x = int(x.max()) + 1 if size_x is None else int(size_x)
    size_y = int(y.max()) + 1 if size_y is None else int(size_y)
    joint = np.zeros((size_x, size_y), dtype=float)
    np.add.at(joint, (x, y), 1.0)
    return joint


def entropy(p, normalize=True):
    """H(X) = -sum_x p(x) log2 p(x), em bits."""
    p = np.asarray(p, dtype=float).ravel()
    if normalize:
        p = to_distribution(p)
    return float(-_xlog2x(p).sum())


def binary_entropy(p):
    """H_b(p) = -p log2 p - (1-p) log2 (1-p). Aceita escalar ou array."""
    p = np.asarray(p, dtype=float)
    h = -(_xlog2x(p) + _xlog2x(1.0 - p))
    return float(h) if h.ndim == 0 else h


def joint_entropy(joint, normalize=True):
    """H(X, Y) a partir de uma tabela conjunta (contagens ou probabilidades)."""
    joint = np.asarray(joint, dtype=float)
    if normalize:
        joint = to_distribution(joint)
    return float(-_xlog2x(joint).sum())


def conditional_entropy(joint, given="x", normalize=True):
    """Entropia condicional a partir de uma tabela conjunta 2D.

    `given="x"` (eixo 0)  -> H(Y | X) = H(X, Y) - H(X)
    `given="y"` (eixo 1)  -> H(X | Y) = H(X, Y) - H(Y)
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint precisa ser uma tabela 2D")
    if normalize:
        joint = to_distribution(joint)
    if given in ("x", "rows", 0):
        marg = joint.sum(axis=1)
    elif given in ("y", "cols", 1):
        marg = joint.sum(axis=0)
    else:
        raise ValueError("given deve ser 'x'/'rows'/0 ou 'y'/'cols'/1")
    return float(-_xlog2x(joint).sum() + _xlog2x(marg).sum())


def cross_entropy(p, q, normalize=True):
    """H(p, q) = -sum_x p(x) log2 q(x). Retorna +inf se q(x)=0 onde p(x)>0."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError("p e q precisam ter o mesmo suporte")
    if normalize:
        p = to_distribution(p)
        q = to_distribution(q)
    if np.any((p > 0.0) & (q <= 0.0)):
        return float("inf")
    return float(-(p * np.log2(np.where(q > 0.0, q, 1.0))).sum())


def kl_divergence(p, q, normalize=True):
    """D(p || q) = sum_x p(x) log2 [p(x) / q(x)] >= 0, em bits.

    Retorna +inf se q(x)=0 em algum ponto com p(x)>0.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError("p e q precisam ter o mesmo suporte")
    if normalize:
        p = to_distribution(p)
        q = to_distribution(q)
    if np.any((p > 0.0) & (q <= 0.0)):
        return float("inf")
    log_ratio = np.log2(np.where(p > 0.0, p, 1.0)) - np.log2(np.where(q > 0.0, q, 1.0))
    return float(np.where(p > 0.0, p * log_ratio, 0.0).sum())


def mutual_information(joint, normalize=True):
    """I(X; Y) = H(X) + H(Y) - H(X, Y), a partir de uma tabela conjunta 2D."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint precisa ser uma tabela 2D")
    if normalize:
        joint = to_distribution(joint)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    hx = -_xlog2x(px).sum()
    hy = -_xlog2x(py).sum()
    hxy = -_xlog2x(joint).sum()
    return float(hx + hy - hxy)


def conditional_mutual_information(joint_xyz, normalize=True):
    """I(X; Y | Z) a partir de p(x, y, z), com eixos (X, Y, Z) = (0, 1, 2).

    I(X; Y | Z) = H(X | Z) + H(Y | Z) - H(X, Y | Z).
    """
    p = np.asarray(joint_xyz, dtype=float)
    if p.ndim != 3:
        raise ValueError("joint_xyz precisa ser uma tabela 3D (X, Y, Z)")
    if normalize:
        p = to_distribution(p)
    pz = p.sum(axis=(0, 1))
    pxz = p.sum(axis=1)   # (X, Z)
    pyz = p.sum(axis=0)   # (Y, Z)
    h_x_given_z = -_xlog2x(pxz).sum() + _xlog2x(pz).sum()
    h_y_given_z = -_xlog2x(pyz).sum() + _xlog2x(pz).sum()
    h_xy_given_z = -_xlog2x(p).sum() + _xlog2x(pz).sum()
    return float(h_x_given_z + h_y_given_z - h_xy_given_z)


def mi_gtest_statistic(counts):
    """Estatistica do teste-G de independencia escrita via informacao mutua.

    G = 2 N ln(2) * I_hat(X; Y)  ->  qui-quadrado com (r-1)(c-1) g.l. sob H0,
    em que N e o total da tabela e I_hat e a informacao mutua estimada (bits)
    a partir das frequencias observadas.

    Retorna (G, dof, I_hat_bits). O p-valor sai de scipy.stats.chi2.sf(G, dof)
    e serve de conferencia independente (nao substitui a conta acima).
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts precisa ser uma tabela de contingencia 2D")
    n = counts.sum()
    if n <= 0:
        raise ValueError("tabela de contingencia vazia")
    i_hat = mutual_information(counts, normalize=True)
    g = 2.0 * n * np.log(2.0) * i_hat
    r, c = counts.shape
    dof = (r - 1) * (c - 1)
    return float(g), int(dof), float(i_hat)
