"""Estimadores de teoria da informação (tudo em bits).

Contém: entropia, informação mútua (condicional), correção de Miller-Madow,
teste de permutação para viés, bootstrap, binning por quantis e a
desigualdade de Fano.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

LN2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# Codificação de variáveis discretas
# --------------------------------------------------------------------------- #
def encode(*cols):
    """Codifica a tupla (col1, col2, ...) em inteiros 0..m-1.

    NaN vira uma categoria própria. Só existem as categorias observadas.
    """
    code = None
    for c in cols:
        inv, uniq = pd.factorize(np.asarray(c), use_na_sentinel=False)
        inv = inv.astype(np.int64)
        code = inv if code is None else code * len(uniq) + inv
        code = pd.factorize(code)[0].astype(np.int64)  # recompacta (evita overflow)
    return code


def qbin(x, k):
    """Discretiza em até k faixas de frequência igual (quantis)."""
    return pd.qcut(pd.Series(np.asarray(x, dtype=float)), k, labels=False, duplicates="drop").to_numpy()


# --------------------------------------------------------------------------- #
# Entropias
# --------------------------------------------------------------------------- #
def entropy_counts(counts, mm=False):
    """H a partir de contagens. mm=True soma a correção de Miller-Madow:
    (m-1)/(2N) nats, onde m = nº de células com contagem > 0."""
    counts = np.asarray(counts)
    counts = counts[counts > 0].astype(float)
    n = counts.sum()
    p = counts / n
    h = -np.sum(p * np.log2(p))
    if mm:
        h += (len(counts) - 1) / (2.0 * n * LN2)
    return h


def entropy(*cols, mm=False):
    return entropy_counts(np.bincount(encode(*cols)), mm)


def cond_entropy(v, x, mm=False):
    """H(V | X) = H(X, V) - H(X)."""
    return entropy(x, v, mm=mm) - entropy(x, mm=mm)


def mutual_info(x, v, z=None, mm=False):
    """I(X; V) ou, se z for dado, I(X; V | Z)."""
    if z is None:
        return entropy(x, mm=mm) + entropy(v, mm=mm) - entropy(x, v, mm=mm)
    return (entropy(x, z, mm=mm) + entropy(v, z, mm=mm)
            - entropy(x, v, z, mm=mm) - entropy(z, mm=mm))


class CondMI:
    """I(X; V | Z) com X e Z fixos e V trocável. Rápido para permutações.

    x, z: arrays quaisquer (serão codificados). v deve vir como inteiros 0..nv-1.
    """

    def __init__(self, x, z=None):
        self.x = encode(x)
        self.nx = int(self.x.max()) + 1
        self.z = np.zeros(len(self.x), dtype=np.int64) if z is None else encode(z)
        self.xz = self.z * self.nx + self.x
        self.c_xz = np.bincount(self.xz)
        self.c_z = np.bincount(self.z)

    def __call__(self, v, mm=False):
        v = np.asarray(v, dtype=np.int64)
        nv = int(v.max()) + 1
        c_vz = np.bincount(self.z * nv + v)
        c_xvz = np.bincount(self.xz * nv + v)
        e = lambda c: entropy_counts(c, mm)
        return e(self.c_xz) + e(c_vz) - e(c_xvz) - e(self.c_z)


def permute_within(strata, v, rng):
    """Embaralha v dentro de cada estrato (preserva P(V | Z); quebra X-V)."""
    order = np.lexsort((rng.random(len(strata)), strata))  # agrupado por estrato, aleatório dentro
    base = np.argsort(strata, kind="stable")               # agrupado por estrato, ordem original
    out = np.empty_like(v)
    out[base] = v[order]
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(func, n, B=300, seed=0, alpha=0.05):
    """func(idx) -> float. Devolve (média, lo, hi) do bootstrap percentil."""
    rng = np.random.default_rng(seed)
    vals = np.array([func(rng.integers(0, n, n)) for _ in range(B)])
    return vals.mean(), np.quantile(vals, alpha / 2), np.quantile(vals, 1 - alpha / 2)


# --------------------------------------------------------------------------- #
# Fano
# --------------------------------------------------------------------------- #
def hb(p):
    """Entropia binária em bits."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fano_min_error(h_cond, m=2):
    """Menor Pe compatível com Fano: h(Pe) + Pe*log2(m-1) >= H(V|X).

    Para V binário (m=2) isso é Pe >= h^{-1}(H(V|X)).
    """
    pmax = (m - 1) / m
    if h_cond <= 1e-12:
        return 0.0
    if h_cond >= np.log2(m):
        return pmax
    f = lambda p: hb(p) + p * np.log2(max(m - 1, 1)) - h_cond
    return float(brentq(f, 1e-12, pmax))


def bayes_error_plugin(x_codes, v):
    """Erro do melhor classificador que só enxerga as células de x (plug-in)."""
    xc = encode(x_codes)
    v = np.asarray(v, dtype=np.int64)
    tab = np.bincount(xc * 2 + v, minlength=2 * (xc.max() + 1)).reshape(-1, 2)
    return 1.0 - tab.max(axis=1).sum() / tab.sum()


def bayes_error_cv(x_codes, v, folds=5, seed=0):
    """Mesmo classificador por célula, mas treinando e testando em partes diferentes."""
    xc = encode(x_codes)
    v = np.asarray(v, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(v))
    nc = xc.max() + 1
    err = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        tab = np.bincount(xc[tr] * 2 + v[tr], minlength=2 * nc).reshape(-1, 2)
        pred = tab.argmax(axis=1)  # célula sem treino -> classe 0 (empate); erro conservador
        err += np.sum(pred[xc[te]] != v[te])
    return err / len(v)
