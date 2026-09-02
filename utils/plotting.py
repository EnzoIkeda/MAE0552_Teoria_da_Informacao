"""Estilo padrao dos graficos, wrappers de seaborn e tabelas para os relatorios.

Uso tipico no comeco do notebook::

    from utils.plotting import setup_estilo, salvar_fig
    setup_estilo()
    ...
    fig, ax = plt.subplots()
    ax.plot(ns, erros)
    salvar_fig("erro_i_vs_n")        # grava figuras/erro_i_vs_n.png

Alem do estilo, o modulo oferece:

- graficos recorrentes nas quatro simulacoes, construidos sobre o seaborn:
  ``plot_estimativa_vs_n`` (curva estimada vs teorica com faixa de IC),
  ``plot_curvas`` (varias series de um DataFrame tidy), ``plot_pmf``
  (comparacao empirica vs teorica), ``plot_barras`` (H(X), H(Y), I(X; Y), ...),
  ``plot_entropia_binaria``, ``heatmap_matriz`` (tabela conjunta, matriz de
  transicao, matriz de confusao), ``plot_distribuicao`` e ``plot_ecdf``
  (conjunto tipico do AEP, taxa de crescimento de Kelly, ...);
- utilidades de tabela para o ``relatorio.md``: ``tabela``, ``tabela_conjunta``,
  ``tabela_comparacao``, ``formata_ic``, ``df_para_markdown`` e ``salvar_tabela``.
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .info_theory import binary_entropy

__all__ = [
    "setup_estilo",
    "salvar_fig",
    "cores",
    "plot_estimativa_vs_n",
    "plot_curvas",
    "plot_pmf",
    "plot_barras",
    "plot_entropia_binaria",
    "heatmap_matriz",
    "plot_distribuicao",
    "plot_ecdf",
    "tabela",
    "tabela_conjunta",
    "tabela_comparacao",
    "formata_ic",
    "df_para_markdown",
    "salvar_tabela",
]

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

#: Paleta categorica padrao (seaborn "deep"); use ``cores(n)`` para n cores.
PALETA = list(sns.color_palette("deep"))
#: Cor reservada para a referencia teorica / valor exato nas comparacoes.
COR_TEORICO = PALETA[3]


def setup_estilo(overrides: dict | None = None, usar_seaborn: bool = True) -> None:
    """Aplica o estilo padrao do projeto ao matplotlib/seaborn.

    Com ``usar_seaborn=True`` (padrao) define primeiro o tema do seaborn
    (``whitegrid`` + paleta ``deep``) e, por cima, os rcParams do projeto, que
    tem a palavra final. `overrides` e um dicionario opcional de rcParams para
    ajustar pontualmente.
    """
    if usar_seaborn:
        sns.set_theme(style="whitegrid", palette="deep", context="notebook")
    mpl.rcParams.update(_ESTILO)
    if overrides:
        mpl.rcParams.update(overrides)


def cores(n: int, paleta: str = "deep") -> list:
    """Lista de `n` cores distintas de uma paleta nomeada do seaborn."""
    return list(sns.color_palette(paleta, n))


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


# --------------------------------------------------------------------------- #
# Graficos                                                                     #
# --------------------------------------------------------------------------- #


def _eixo(ax):
    """Devolve ``(fig, ax)``; cria um par novo quando `ax` e None."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots()


def plot_estimativa_vs_n(
    ns,
    estimativas,
    ic_baixo=None,
    ic_alto=None,
    teorico=None,
    *,
    label="estimado",
    xlabel="N (amostras)",
    ylabel="bits",
    logx=True,
    ax=None,
):
    """Curva de uma quantidade estimada em funcao do tamanho de amostra.

    Serve para "estimado vs teorico" (I vs N, H vs N, BER vs N, ...). `ic_baixo`
    e `ic_alto` (por exemplo a saida de :func:`utils.estimation.bootstrap_ci`
    avaliada em cada N) viram uma faixa sombreada; `teorico` (escalar) vira uma
    linha horizontal de referencia. Retorna o ``ax``.
    """
    _, ax = _eixo(ax)
    ns = np.asarray(ns, dtype=float)
    estimativas = np.asarray(estimativas, dtype=float)
    sns.lineplot(x=ns, y=estimativas, marker="o", label=label, ax=ax, color=PALETA[0])
    if ic_baixo is not None and ic_alto is not None:
        ax.fill_between(
            ns,
            np.asarray(ic_baixo, dtype=float),
            np.asarray(ic_alto, dtype=float),
            alpha=0.2,
            color=PALETA[0],
            label="IC",
        )
    if teorico is not None:
        ax.axhline(float(teorico), ls="--", lw=1.5, color=COR_TEORICO, label="teorico", zorder=5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_curvas(
    data,
    x,
    y,
    hue=None,
    *,
    teorico=None,
    errorbar=None,
    marker="o",
    xlabel=None,
    ylabel=None,
    ax=None,
):
    """``sns.lineplot`` de um DataFrame tidy, com referencia teorica opcional.

    `data` e um DataFrame no formato longo (uma linha por observacao). `teorico`
    pode ser um escalar (linha horizontal) ou um callable ``f(x) -> y`` (curva
    de referencia sobre o dominio de `x`). `errorbar` e repassado ao seaborn
    (por exemplo ``("ci", 95)`` ou ``"sd"``). Retorna o ``ax``.
    """
    _, ax = _eixo(ax)
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, errorbar=errorbar, ax=ax)
    if callable(teorico):
        xs = np.linspace(float(np.min(data[x])), float(np.max(data[x])), 200)
        ax.plot(xs, [teorico(v) for v in xs], ls="--", lw=1.5, color=COR_TEORICO, label="teorico")
        ax.legend()
    elif teorico is not None:
        ax.axhline(float(teorico), ls="--", lw=1.5, color=COR_TEORICO, label="teorico", zorder=5)
        ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def plot_pmf(
    p,
    rotulos=None,
    *,
    comparar=None,
    nomes=("empirica", "teorica"),
    ylabel="probabilidade",
    ax=None,
):
    """Grafico de barras de uma pmf; com `comparar` mostra duas lado a lado.

    Util para conferir uma frequencia empirica contra a distribuicao usada na
    simulacao. `p` e `comparar` sao vetores de probabilidade de mesmo tamanho.
    Retorna o ``ax``.
    """
    _, ax = _eixo(ax)
    p = np.asarray(p, dtype=float).ravel()
    k = p.size
    if rotulos is not None:
        rotulos = [str(r) for r in rotulos]
    else:
        rotulos = [str(i) for i in range(k)]
    if comparar is None:
        df = pd.DataFrame({"simbolo": rotulos, ylabel: p})
        sns.barplot(data=df, x="simbolo", y=ylabel, ax=ax, color=PALETA[0])
    else:
        q = np.asarray(comparar, dtype=float).ravel()
        df = pd.DataFrame(
            {
                "simbolo": rotulos * 2,
                ylabel: np.concatenate([p, q]),
                "fonte": [nomes[0]] * k + [nomes[1]] * k,
            }
        )
        sns.barplot(data=df, x="simbolo", y=ylabel, hue="fonte", ax=ax)
    ax.set_xlabel("simbolo")
    return ax


def plot_barras(valores, *, ylabel="bits", anotar=True, ax=None):
    """Barras a partir de um dict ``{nome: valor}`` (H(X), H(Y), H(X, Y), I(X; Y), ...)."""
    _, ax = _eixo(ax)
    nomes = list(valores)
    alturas = [float(valores[n]) for n in nomes]
    sns.barplot(x=nomes, y=alturas, hue=nomes, legend=False, palette="deep", ax=ax)
    ax.set_ylabel(ylabel)
    if anotar:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    return ax


def plot_entropia_binaria(p_destaque=None, *, ax=None):
    """Curva ``H_b(p) = -p log2 p - (1 - p) log2 (1 - p)`` no intervalo [0, 1]."""
    _, ax = _eixo(ax)
    ps = np.linspace(0.0, 1.0, 501)
    sns.lineplot(x=ps, y=binary_entropy(ps), ax=ax, color=PALETA[0])
    if p_destaque is not None:
        p_destaque = float(p_destaque)
        ax.plot([p_destaque], [binary_entropy(p_destaque)], "o", color=COR_TEORICO)
        ax.axvline(p_destaque, ls=":", lw=1.0, color=COR_TEORICO)
    ax.set_xlabel("p")
    ax.set_ylabel("H_b(p)  [bits]")
    ax.set_ylim(0.0, 1.02)
    return ax


def heatmap_matriz(
    M,
    rotulos_x=None,
    rotulos_y=None,
    *,
    fmt=".2f",
    cmap="rocket",
    cbar_label=None,
    xlabel="Y",
    ylabel="X",
    ax=None,
):
    """Heatmap de uma matriz 2D: tabela conjunta, matriz de transicao, confusao.

    Convencao do projeto: eixo 0 (linhas) = X, eixo 1 (colunas) = Y. Retorna o ``ax``.
    """
    _, ax = _eixo(ax)
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M precisa ser uma matriz 2D")
    r, c = M.shape
    xt = list(rotulos_y) if rotulos_y is not None else list(range(c))
    yt = list(rotulos_x) if rotulos_x is not None else list(range(r))
    sns.heatmap(
        M,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        xticklabels=xt,
        yticklabels=yt,
        cbar_kws={"label": cbar_label} if cbar_label else None,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_distribuicao(
    amostras,
    *,
    referencia=None,
    bins="auto",
    kde=True,
    stat="density",
    xlabel=None,
    ax=None,
):
    """Histograma (com KDE) de um vetor de amostras.

    Feito para as distribuicoes empiricas das simulacoes: a variavel
    ``-1/n log2 p(x^n)`` do AEP, a taxa de crescimento no jogo de Kelly, o numero
    de erros por bloco no canal. `referencia` (escalar) vira uma linha vertical.
    Retorna o ``ax``.
    """
    _, ax = _eixo(ax)
    amostras = np.asarray(amostras, dtype=float).ravel()
    sns.histplot(amostras, bins=bins, kde=kde, stat=stat, ax=ax, color=PALETA[0])
    if referencia is not None:
        ax.axvline(float(referencia), ls="--", lw=1.5, color=COR_TEORICO, label="referencia")
        ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def plot_ecdf(amostras, *, referencia=None, xlabel=None, label=None, ax=None):
    """ECDF de um vetor de amostras (``sns.ecdfplot``), com referencia vertical opcional.

    Util para o AEP: sobrepor a ECDF de ``-1/n log2 p(x^n)`` para varios ``n`` e
    ver a massa concentrar em torno de ``H(X)`` (passe ``referencia=H``).
    """
    _, ax = _eixo(ax)
    amostras = np.asarray(amostras, dtype=float).ravel()
    sns.ecdfplot(x=amostras, ax=ax, label=label)
    if referencia is not None:
        ax.axvline(float(referencia), ls="--", lw=1.5, color=COR_TEORICO, label="referencia")
    if label or referencia is not None:
        ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


# --------------------------------------------------------------------------- #
# Tabelas e formatacao                                                         #
# --------------------------------------------------------------------------- #


def tabela(registros, *, ordem=None, indice=None) -> pd.DataFrame:
    """DataFrame a partir de `registros`, para exibir ou salvar resultados.

    `registros` pode ser uma lista de dicts (uma linha cada), um dict de colunas
    (``{coluna: [valores...]}``) ou um dict simples de escalares (uma linha).
    `ordem` reordena/seleciona colunas; `indice` define a coluna de indice.
    """
    if isinstance(registros, dict):
        escalar = all(np.isscalar(v) or isinstance(v, str) for v in registros.values())
        df = pd.DataFrame([registros]) if escalar else pd.DataFrame(registros)
    else:
        df = pd.DataFrame(list(registros))
    if ordem is not None:
        df = df[list(ordem)]
    if indice is not None:
        df = df.set_index(indice)
    return df


def tabela_conjunta(
    joint,
    rotulos_x=None,
    rotulos_y=None,
    *,
    normalizar=False,
    margens=True,
) -> pd.DataFrame:
    """Tabela conjunta 2D como DataFrame rotulado (X nas linhas, Y nas colunas).

    Com `normalizar=True` divide pelo total (probabilidades); com `margens=True`
    acrescenta a linha e a coluna ``total`` (marginais).
    """
    M = np.asarray(joint, dtype=float)
    if M.ndim != 2:
        raise ValueError("joint precisa ser uma tabela 2D")
    if normalizar:
        total = M.sum()
        if total <= 0:
            raise ValueError("soma nula: nao e possivel normalizar")
        M = M / total
    r, c = M.shape
    idx = list(rotulos_x) if rotulos_x is not None else [f"x{i}" for i in range(r)]
    col = list(rotulos_y) if rotulos_y is not None else [f"y{j}" for j in range(c)]
    df = pd.DataFrame(M, index=idx, columns=col)
    if margens:
        df["total"] = df.sum(axis=1)
        df.loc["total"] = df.sum(axis=0)
    return df


def tabela_comparacao(nomes, estimado, teorico, ic_baixo=None, ic_alto=None) -> pd.DataFrame:
    """Tabela "estimado vs teorico" com erro absoluto/relativo e cobertura do IC.

    Os argumentos sao sequencias alinhadas por posicao. Quando `ic_baixo` e
    `ic_alto` sao dados, inclui o IC formatado e uma coluna booleana dizendo se
    ele cobre o valor teorico.
    """
    nomes = list(nomes)
    est = np.asarray(estimado, dtype=float)
    teo = np.asarray(teorico, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        erro_rel = np.where(teo != 0.0, np.abs(est - teo) / np.abs(teo), np.nan)
    df = pd.DataFrame(
        {
            "quantidade": nomes,
            "estimado": est,
            "teorico": teo,
            "erro_abs": np.abs(est - teo),
            "erro_rel_%": 100.0 * erro_rel,
        }
    ).set_index("quantidade")
    if ic_baixo is not None and ic_alto is not None:
        lo = np.asarray(ic_baixo, dtype=float)
        hi = np.asarray(ic_alto, dtype=float)
        df["IC"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(lo, hi, strict=True)]
        df["cobre_teorico"] = (lo <= teo) & (teo <= hi)
    return df


def formata_ic(est, baixo, alto, casas=4) -> str:
    """Formata uma estimativa com IC como ``"0.1234 [0.1000, 0.1500]"``."""
    return f"{est:.{casas}f} [{baixo:.{casas}f}, {alto:.{casas}f}]"


def df_para_markdown(df, *, casas=4, indice=True) -> str:
    """DataFrame -> tabela em Markdown, pronta para colar no ``relatorio.md``.

    Implementacao propria (nao depende de ``tabulate``): floats sao formatados
    com `casas` decimais, booleanos viram ``sim``/``nao`` e o resto vai por
    ``str``.
    """

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "sim" if v else "nao"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{casas}f}"
        return str(v)

    cols = [str(c) for c in df.columns]
    cabecalho = ([df.index.name or ""] if indice else []) + cols
    linhas = [cabecalho, ["---"] * len(cabecalho)]
    for chave, linha in df.iterrows():
        celulas = ([str(chave)] if indice else []) + [fmt(linha[c]) for c in df.columns]
        linhas.append(celulas)
    return "\n".join("| " + " | ".join(cel) + " |" for cel in linhas)


def salvar_tabela(
    df,
    nome,
    *,
    pasta="dados",
    formatos=("csv", "md"),
    casas=4,
) -> list[pathlib.Path]:
    """Salva um DataFrame em ``pasta/<nome>.<fmt>`` (``csv``, ``md`` e/ou ``tex``).

    Espelha :func:`salvar_fig`. ``md`` usa :func:`df_para_markdown`. Devolve os
    caminhos gravados.
    """
    destino = pathlib.Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for fmt in formatos:
        caminho = destino / f"{nome}.{fmt}"
        if fmt == "csv":
            df.to_csv(caminho)
        elif fmt == "md":
            caminho.write_text(df_para_markdown(df, casas=casas) + "\n", encoding="utf-8")
        elif fmt == "tex":
            caminho.write_text(df.to_latex(float_format=f"%.{casas}f"), encoding="utf-8")
        else:
            raise ValueError(f"formato nao suportado: {fmt!r}")
        caminhos.append(caminho)
    return caminhos
