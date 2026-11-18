"""Parte 6 - Qual lane é a mais decisiva? (ouro, XP, CS e KDA por lane)

Cada partida tem 5 duelos de lane (top, jng, mid, bot, sup). Para cada um usamos a diferença azul - vermelho
(ouro, XP e CS já vêm assim no Oracle's Elixir; KDA = (abates + assistências) / max(mortes, 1) calculado por jogador).

KDA aos 15 min usa killsat15/assistsat15/deathsat15 (pré-resultado). KDA FINAL é calculado no fim da partida e carrega o
resultado (quem vence abate mais): serve só de comparação, como o "vazamento" das Partes 3 e 5.

Medidas (bits, limite inferior variacional por regressão logística em CV de 5 folds):
  1. I(variável da lane ; V) isolada.
  2. I(lane ; V | as outras 4 lanes): quanto se perde ao retirar a lane do modelo completo.
  3. Curva no tempo: I(ouro da lane no min t ; V), t = 10, 15, 20, 25 (amostra comum).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import it_utils as it

LANES = ["top", "jng", "mid", "bot", "sup"]
VARS15 = {"gd": "golddiffat15", "xd": "xpdiffat15", "cd": "csdiffat15", "kda": "kda15", "net": "net15"}


def _loss(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def _loss0(y, seed=0):
    p = np.full(len(y), y.mean())  # priori (diferença desprezível em 90 mil jogos)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def cols(var, lanes=LANES):
    return [f"{var}_{l}" for l in lanes]


def isoladas(w, seed=0):
    y = w.V.to_numpy()
    l0 = _loss0(y).mean()
    rows = []
    for l in LANES:
        r = dict(lane=l)
        for k, v in (("ouro", "golddiffat15"), ("XP", "xpdiffat15"), ("CS", "csdiffat15"),
                     ("KDA 15", "kda15"), ("(A+K-D) 15", "net15"),
                     ("[consequência] KDA final", "kdaF"), ("[consequência] (A+K-D) final", "netF")):
            r[k] = l0 - _loss(w[[f"{v}_{l}"]].to_numpy(float), y, seed).mean()
        # ouro por discretização, com correção por permutação (confere o limite variacional)
        z = it.qbin(w[f"golddiffat15_{l}"], 20)
        cmi = it.CondMI(z)
        rng = np.random.default_rng(seed)
        obs = cmi(y)
        nul = np.mean([cmi(rng.permutation(y)) for _ in range(50)])
        r["ouro (discretizado, 20 faixas)"] = obs - nul
        rows.append(r)
    return pd.DataFrame(rows)


def sem_a_lane(w, modos, seed=0):
    """I(lane ; V | outras 4) = perda do modelo sem a lane - perda do completo (por jogo, com IC)."""
    y = w.V.to_numpy()
    rows = []
    for nome, vs in modos.items():
        full = [f"{v}_{l}" for v in vs for l in LANES]
        lf = _loss(w[full].to_numpy(float), y, seed)
        l0 = _loss0(y)
        for l in LANES:
            rest = [f"{v}_{k}" for v in vs for k in LANES if k != l]
            d = _loss(w[rest].to_numpy(float), y, seed) - lf
            solo = [f"{v}_{l}" for v in vs]
            ls = _loss(w[solo].to_numpy(float), y, seed)
            rows.append(dict(modelo=nome, lane=l, I_completo=l0.mean() - lf.mean(),
                             I_isolada=l0.mean() - ls.mean(),
                             I_dado_outras=d.mean(), ic95=1.96 * d.std() / np.sqrt(len(d))))
    return pd.DataFrame(rows)


def coeficientes(w, seed=0):
    y = w.V.to_numpy()
    X = w[cols("golddiffat15")].to_numpy(float)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    c = m[-1].coef_[0]
    return pd.DataFrame(dict(lane=LANES, coef_por_dp=c, odds_ratio_por_dp=np.exp(c),
                             dp_ouro=X.std(axis=0)))


def tempo(w, seed=0):
    y = w.V.to_numpy()
    l0 = _loss0(y).mean()
    rows = []
    for t in (10, 15, 20, 25):
        for l in LANES:
            rows.append(dict(minuto=t, lane=l, I=l0 - _loss(w[[f"golddiffat{t}_{l}"]].to_numpy(float), y, seed).mean()))
        full = [f"golddiffat{t}_{l}" for l in LANES]
        rows.append(dict(minuto=t, lane="5 lanes juntas", I=l0 - _loss(w[full].to_numpy(float), y, seed).mean()))
    return pd.DataFrame(rows)


def run(w, out, seed=0):
    os.makedirs(out, exist_ok=True)
    need15 = cols("golddiffat15") + cols("xpdiffat15") + cols("csdiffat15") + cols("kda15") + cols("net15") + cols("kdaF") + cols("netF")
    s = w.dropna(subset=need15).reset_index(drop=True)
    print(f"  {len(s):,} partidas com as 5 lanes completas até o min 15")
    iso = isoladas(s, seed)
    iso.to_csv(os.path.join(out, "parte6_isoladas.csv"), index=False)
    print(iso.round(4).to_string())

    modos = {"ouro": ["golddiffat15"], "ouro+XP+CS": ["golddiffat15", "xpdiffat15", "csdiffat15"],
             "ouro+XP+CS+KDA": ["golddiffat15", "xpdiffat15", "csdiffat15", "kda15"]}
    so = sem_a_lane(s, modos, seed)
    so.to_csv(os.path.join(out, "parte6_sem_a_lane.csv"), index=False)
    print(so.round(4).to_string())

    co = coeficientes(s, seed)
    co.to_csv(os.path.join(out, "parte6_coeficientes.csv"), index=False)
    print(co.round(3).to_string())

    s25 = w.dropna(subset=sum([cols(f"golddiffat{t}") for t in (10, 15, 20, 25)], [])).reset_index(drop=True)
    print(f"  amostra comum até o min 25: {len(s25):,}")
    tp = tempo(s25, seed)
    tp.to_csv(os.path.join(out, "parte6_tempo.csv"), index=False)
    print(tp.pivot(index="lane", columns="minuto", values="I").round(4).to_string())

    _figura(iso, so, tp, out)
    return iso, so, co, tp


def _figura(iso, so, tp, out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    x = np.arange(len(LANES))
    ordem = ["ouro", "XP", "CS", "KDA 15"]
    for i, k in enumerate(ordem):
        ax[0].bar(x + (i - 1.5) * .2, iso[k], .2, label=k)
    ax[0].bar(x + 2.5 * .2 + .02, iso["[consequência] KDA final"], .2, color="gray", alpha=.5, label="KDA final (consequência)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(LANES)
    ax[0].set_ylabel("bits")
    ax[0].set_title("Informação isolada de cada lane sobre a vitória (min 15)", fontsize=10)
    ax[0].legend(fontsize=8)
    # (b) dado as outras 4 lanes
    m = so[so.modelo == "ouro+XP+CS+KDA"]
    g = so[so.modelo == "ouro"]
    ax[1].bar(x - .2, g.I_isolada, .4, alpha=.45, color="C0", label="ouro, lane sozinha")
    ax[1].bar(x + .2, g.I_dado_outras, .4, yerr=g.ic95, capsize=3, color="C0", label="ouro, dado as outras 4 lanes")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(LANES)
    ax[1].set_ylabel("bits")
    ax[1].set_title("Isolada vs. depois de conhecer as outras 4 lanes (ouro, min 15)", fontsize=10)
    ax[1].legend(fontsize=8)
    # (c) tempo
    for l in LANES:
        d = tp[tp.lane == l]
        ax[2].plot(d.minuto, d.I, marker="o", label=l)
    d = tp[tp.lane == "5 lanes juntas"]
    ax[2].plot(d.minuto, d.I, "k--", marker="s", label="5 lanes juntas")
    ax[2].set_xticks([10, 15, 20, 25])
    ax[2].set_xlabel("minuto")
    ax[2].set_ylabel("bits")
    ax[2].set_title("Informação do ouro de cada lane no tempo (amostra comum, min 25)", fontsize=10)
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.25)
    plt.tight_layout()
    _paineis(fig, ax, out, 'parte6')
    plt.savefig(os.path.join(out, "parte6_lanes.png"), dpi=130)
    plt.close()


def _paineis(fig, ax, out, prefixo):
    """Salva cada painel como PNG separado (para relatórios em página retrato)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for i, a in enumerate(ax):
        bb = a.get_tightbbox(r).expanded(1.03, 1.03).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(out, f"{prefixo}_painel{i + 1}.png"), dpi=170, bbox_inches=bb)
