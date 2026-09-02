"""Padronizacao visual dos graficos e salvamento de figuras.

Uso tipico no comeco do notebook::

    from utils.plotting import setup_estilo, salvar_fig
    setup_estilo()
    ...
    fig, ax = plt.subplots()
    ax.plot(ns, erros)
    salvar_fig("erro_i_vs_n")        # grava figuras/erro_i_vs_n.png
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

_ESTILO = {
    "figure.figsize": (7.0, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "legend.frameon": False,
}


def setup_estilo(overrides: dict | None = None) -> None:
    """Aplica o estilo padrao do projeto ao matplotlib (rcParams).

    `overrides` e um dicionario opcional de rcParams para ajustar pontualmente.
    """
    mpl.rcParams.update(_ESTILO)
    if overrides:
        mpl.rcParams.update(overrides)


def salvar_fig(nome, fig=None, pasta="figuras", formatos=("png",)) -> list[pathlib.Path]:
    """Salva a figura em ``pasta/<nome>.<fmt>`` e devolve os caminhos gravados.

    `nome` vem sem extensao. A pasta e criada se nao existir. Por padrao grava
    PNG; passe ``formatos=("png", "pdf")`` para incluir a versao vetorial no
    relatorio.
    """
    fig = fig if fig is not None else plt.gcf()
    destino = pathlib.Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for fmt in formatos:
        caminho = destino / f"{nome}.{fmt}"
        fig.savefig(caminho)
        caminhos.append(caminho)
    return caminhos
