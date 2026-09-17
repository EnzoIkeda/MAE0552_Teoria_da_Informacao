"""Tópico 2: cadeia de canais ruidosos e "telefone sem fio".

X -> Y1 -> Y2 -> ... -> Yk, cada elo um canal binário simétrico independente.
A desigualdade de processamento de dados prevê I(X; Y1) >= I(X; Y2) >= ...

`compose_bsc_chain` guarda TODOS os estágios (não só o último), porque é preciso
I(X; Yj) para cada j. As informações mútuas saem de `utils.info_theory`.
"""

from __future__ import annotations

import numpy as np

from utils.simulation import bsc


def compose_bsc_chain(x_bits: np.ndarray, p_list, generator) -> list[np.ndarray]:
    """Propaga `x_bits` por canais BSC de parâmetros `p_list`.

    Retorna [X, Y1, ..., Yk] (k = len(p_list)), cada um array de 0/1 do mesmo
    tamanho de `x_bits`.
    """
    stages = [np.asarray(x_bits).astype(int)]
    for p in p_list:
        stages.append(bsc(stages[-1], p, generator))
    return stages


def class_conditional_gaussian(labels: np.ndarray, means, sigma: float, generator) -> np.ndarray:
    """Passo 7: "imagem" simplificada, vetor gaussiano cuja média depende da classe.

    `means` tem shape (n_classes, dim); retorna array (len(labels), dim).
    """
    means = np.asarray(means, dtype=float)
    base = means[np.asarray(labels).astype(int)]
    return base + sigma * generator.standard_normal(base.shape)


def compress_mean_blocks(x: np.ndarray, factor: int) -> np.ndarray:
    """Passo 7: compressão por média de blocos de coordenadas.

    Divide o último eixo de `x` em blocos de tamanho `factor` e substitui cada
    bloco pela média, reduzindo a dimensão por esse fator (a dimensão original
    precisa ser múltipla de `factor`).
    """
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    if dim % factor != 0:
        raise ValueError("a dimensao precisa ser multipla de `factor`")
    blocks = x.reshape(*x.shape[:-1], dim // factor, factor)
    return blocks.mean(axis=-1)


def nearest_mean_predict(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Passo 7: classifica cada linha de `x` pela média de classe mais próxima (distância)."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    dists = np.linalg.norm(x[:, None, :] - means[None, :, :], axis=-1)
    return np.argmin(dists, axis=1)
