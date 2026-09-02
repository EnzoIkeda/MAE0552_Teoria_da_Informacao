"""Estimacao com incerteza: reamostragem bootstrap para barras de erro.

Nao implementa quantidades de teoria da informacao -- recebe uma funcao
estatistica qualquer (por exemplo uma de `utils.info_theory`) e devolve uma
estimativa pontual com intervalo de confianca. Todos os topicos comparam
"estimado vs teorico" e precisam dessa nocao de incerteza (I vs N,
I(X; Yj) vs j, BER vs p, ...).
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(data, statistic, n_boot=2000, ci=95, generator=None, paired=True):
    """Intervalo de confianca bootstrap (metodo do percentil) para `statistic`.

    Parameters
    ----------
    data : array-like ou tupla/lista de array-likes
        A amostra. Se for uma tupla de varios arrays de mesmo tamanho
        (ex.: ``(x, y)``), cada reamostragem usa os MESMOS indices em todos
        eles (bootstrap pareado), preservando a dependencia entre as colunas.
    statistic : callable
        ``statistic(sample)`` ou ``statistic(*samples)`` -> float.
    n_boot : int
        Numero de reamostragens.
    ci : float
        Nivel do intervalo, em porcentagem (ex.: 95 -> IC de 95%).
    generator : numpy.random.Generator, opcional
        Gerador aleatorio (use um com seed fixo para reprodutibilidade).
    paired : bool
        Quando `data` e uma tupla, usa indices comuns entre os arrays.

    Returns
    -------
    (estimativa, baixo, alto) : tuple[float, float, float]
        Estimativa na amostra original e limites inferior/superior do IC.
    """
    g = generator if generator is not None else np.random.default_rng()

    if isinstance(data, (tuple, list)):
        arrays = [np.asarray(d) for d in data]
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("todos os arrays precisam ter o mesmo tamanho")
        est = float(statistic(*arrays))
        boot = np.empty(n_boot)
        for b in range(n_boot):
            if paired:
                idx = g.integers(0, n, n)
                boot[b] = statistic(*[a[idx] for a in arrays])
            else:
                boot[b] = statistic(*[a[g.integers(0, n, n)] for a in arrays])
    else:
        arr = np.asarray(data)
        n = len(arr)
        est = float(statistic(arr))
        boot = np.empty(n_boot)
        for b in range(n_boot):
            boot[b] = statistic(arr[g.integers(0, n, n)])

    alpha = (100 - ci) / 2
    low, high = np.percentile(boot, [alpha, 100 - alpha])
    return est, float(low), float(high)
