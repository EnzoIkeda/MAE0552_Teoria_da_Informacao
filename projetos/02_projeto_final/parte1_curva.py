"""Parte 1 - Quanto a diferença de ouro no minuto t informa sobre a vitória?

I(GD_t ; V) para t = 10, 15, 20, 25, com IC por bootstrap, correção de
Miller-Madow e checagem cruzada com um limite inferior variacional
(I >= H(V) - log-loss de uma regressão logística em validação cruzada).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import it_utils as it
from data import MINUTES


def _logloss_bits(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log2(p) + (1 - y) * np.log2(1 - p))))


def estimate(x, v, k, B, seed=0):
    """Todas as quantidades para uma amostra (x contínuo, v binário)."""
    xb = it.qbin(x, k)
    hv = it.entropy(v)
    i_plug = it.mutual_info(xb, v)
    i_mm = it.mutual_info(xb, v, mm=True)
    _, lo, hi = it.bootstrap_ci(lambda idx: it.mutual_info(xb[idx], v[idx]), len(v), B=B, seed=seed)
    h_cond = hv - i_plug
    # limite inferior variacional (cross-check independente da discretização)
    X = (np.asarray(x, float) / 1000.0).reshape(-1, 1)
    p = cross_val_predict(LogisticRegression(), X, v,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    i_var = hv - _logloss_bits(v, p)
    return dict(N=len(v), H_V=hv, I_plugin=i_plug, I_MM=i_mm, I_lo=lo, I_hi=hi,
                H_V_dado_X=h_cond, frac_resolvida=i_plug / hv,
                I_logit_cv=i_var,
                erro_min_fano=it.fano_min_error(h_cond),
                erro_bayes_plugin=it.bayes_error_plugin(xb, v),
                erro_base=min(v.mean(), 1 - v.mean()))


def minuto_decidido(minutos, frac, limiar):
    """Primeiro minuto (interpolado linearmente) em que a fração resolvida >= limiar."""
    if frac[0] >= limiar:
        return f"<= {minutos[0]}"
    for i in range(1, len(minutos)):
        if frac[i] >= limiar:
            t = minutos[i - 1] + (limiar - frac[i - 1]) / (frac[i] - frac[i - 1]) * (minutos[i] - minutos[i - 1])
            return f"{t:.1f}"
    return f"> {minutos[-1]}"


def exclusoes(games, out):
    """Por que partidas saem das Partes 1 e 3: falta de dados (liga parcial) ou duração do jogo.

    Sem golddiffat10 não é duração (nenhum jogo profissional acaba antes do min 10): é dado ausente.
    Com min 10 mas sem min 25, a causa esperada é o jogo ter acabado antes do min 25.
    """
    g = games.assign(
        sem_dados_min10=games.gd10.isna(),
        sem_min25_curto=games.gd10.notna() & games.gd25.isna() & (games.gamelength_min < 25),
        sem_min25_outro=games.gd10.notna() & games.gd25.isna() & ~(games.gamelength_min < 25),
    )
    tab = (g.groupby(["league", "completude"], dropna=False)
             .agg(partidas=("V", "size"), sem_dados_min10=("sem_dados_min10", "sum"),
                  sem_min25_curto=("sem_min25_curto", "sum"), sem_min25_outro=("sem_min25_outro", "sum"))
             .sort_values("sem_dados_min10", ascending=False))
    tab.to_csv(os.path.join(out, "parte1_exclusoes_por_liga.csv"))
    tot = tab.sum()
    print(f"\nExclusões: {int(tot.partidas):,} partidas; sem dados no min 10: {int(tot.sem_dados_min10):,} "
          f"(dado ausente, não duração); sem min 25 por jogo curto: {int(tot.sem_min25_curto):,}; "
          f"sem min 25 por outro motivo: {int(tot.sem_min25_outro):,}")
    print("Ligas que mais perdem partidas por dado ausente:")
    print(tab[tab.sem_dados_min10 > 0].head(10).to_string())
    return tab


def run(games, out, bins=20, B=300, limiar=0.25, seed=0):
    os.makedirs(out, exist_ok=True)
    cols = [f"gd{m}" for m in MINUTES]
    exclusoes(games, out)

    # amostra comum: só jogos que chegaram ao min 25 (comparação justa entre t)
    comum = games.dropna(subset=cols).reset_index(drop=True)
    rows = []
    for m in MINUTES:
        r = estimate(comum[f"gd{m}"].to_numpy(), comum.V.to_numpy(), bins, B, seed)
        rows.append(dict(minuto=m, amostra="comum (jogos >= 25 min)", **r))
    # amostra completa por t: todo jogo que chegou ao minuto t (viés de seleção!)
    for m in MINUTES:
        s = games.dropna(subset=[f"gd{m}"])
        r = estimate(s[f"gd{m}"].to_numpy(), s.V.to_numpy(), bins, B, seed)
        rows.append(dict(minuto=m, amostra="completa por t", **r))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "parte1_curva.csv"), index=False)

    # sensibilidade ao número de faixas (amostra comum)
    sens = []
    for k in (4, 6, 8, 10, 15, 20, 30, 40):
        for m in MINUTES:
            xb = it.qbin(comum[f"gd{m}"], k)
            sens.append(dict(K=k, minuto=m, I_plugin=it.mutual_info(xb, comum.V),
                             I_MM=it.mutual_info(xb, comum.V, mm=True)))
    pd.DataFrame(sens).to_csv(os.path.join(out, "parte1_sensibilidade_K.csv"), index=False)

    c = res[res.amostra.str.startswith("comum")]
    f = res[res.amostra == "completa por t"]
    dec = minuto_decidido(list(c.minuto), list(c.frac_resolvida), limiar)

    # ---------------- figura ----------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    a = ax[0]
    a.errorbar(c.minuto, c.I_plugin, yerr=[c.I_plugin - c.I_lo, c.I_hi - c.I_plugin],
               marker="o", capsize=4, lw=2, label="plug-in (IC 95% bootstrap)")
    a.plot(c.minuto, c.I_MM, "s--", label="Miller-Madow")
    a.plot(c.minuto, c.I_logit_cv, "^:", label="limite inferior (logística, CV)")
    a.plot(f.minuto, f.I_plugin, "x-.", color="gray", label="amostra completa por t")
    a.set_xlabel("minuto t"); a.set_ylabel("I(dif. de ouro em t ; V)  [bits]")
    a.set_title("Informação sobre a vitória"); a.set_xticks(MINUTES); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    a.plot(c.minuto, c.H_V_dado_X, "o-", lw=2, label="H(V | ouro em t)")
    a.axhline(c.H_V.iloc[0], color="k", ls="--", label=f"H(V) = {c.H_V.iloc[0]:.3f} bits")
    a.axhline(c.H_V.iloc[0] * (1 - limiar), color="r", ls=":", label=f"{limiar:.0%} da incerteza resolvida")
    for x_, y_, fr in zip(c.minuto, c.H_V_dado_X, c.frac_resolvida):
        a.annotate(f"{fr:.0%}", (x_, y_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a.set_ylim(0, 1.05); a.set_xlabel("minuto t"); a.set_ylabel("bits")
    a.set_title(f"Incerteza restante (decidido ~ min {dec})"); a.set_xticks(MINUTES)
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2]
    for m in MINUTES:
        xb = it.qbin(comum[f"gd{m}"], 20)
        g = comum.assign(b=xb).groupby("b").agg(gd=(f"gd{m}", "mean"), w=("V", "mean"))
        a.plot(g.gd / 1000, g.w, "o-", ms=3, label=f"min {m}")
    a.axhline(.5, color="gray", lw=.8); a.axvline(0, color="gray", lw=.8)
    a.set_xlabel("diferença de ouro (azul - vermelho) [mil]"); a.set_ylabel("P(azul vence)")
    a.set_title("P(V | diferença de ouro)"); a.legend(fontsize=8); a.grid(alpha=.3)

    fig.suptitle(f"Parte 1 - N = {len(comum):,} partidas na amostra comum", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte1_curva.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== PARTE 1 (amostra comum) ===")
    print(c[["minuto", "N", "H_V", "I_plugin", "I_lo", "I_hi", "I_MM", "I_logit_cv",
             "H_V_dado_X", "frac_resolvida", "erro_min_fano", "erro_bayes_plugin"]]
          .round(4).to_string(index=False))
    for lim in sorted({limiar, 0.25, 0.5}):
        print(f"Minuto em que {lim:.0%} da incerteza foi resolvida: "
              f"{minuto_decidido(list(c.minuto), list(c.frac_resolvida), lim)}")
    print("\nAmostra completa por t (viés de seleção):")
    print(f[["minuto", "N", "I_plugin", "H_V_dado_X", "frac_resolvida"]].round(4).to_string(index=False))
    return res
