"""Parte 2c - Detalhando os picks: quem mais contribui para I(campeão; V) e por quê.

Decomposição exata:  I(X;V) = sum_c p(c) * D( P(V|c) || P(V) ),  em bits.
Cada campeão contribui com (frequência do pick) x (distância de sua win rate a 50%, em bits).
Também checa: piso de ruído, estabilidade (metades aleatórias e temporais),
ajuste por time-ano e win rate por posição.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import it_utils as it
from parte2_draft import _units, CONTEXTOS


def _half_table(p, mask):
    return p[mask].groupby("champion").V.agg(["size", "mean"])


def run(p, out, top=10, min_half=250, min_role=300, seed=0):
    os.makedirs(out, exist_ok=True)
    N = len(p)
    pv = p.V.mean()
    n_unit = p.groupby(["gameid", "side"]).ngroups
    u_code, units = _units(p)

    # ---- contribuição de cada campeão (plug-in, marginal) ----
    g = p.groupby("champion").V.agg(n="size", wr="mean")
    w = g.wr.clip(1e-9, 1 - 1e-9)
    g["kl_bits"] = w * np.log2(w / pv) + (1 - w) * np.log2((1 - w) / (1 - pv))
    g["contrib_mbit"] = g.n / N * g.kl_bits * 1e3
    total = g.contrib_mbit.sum()
    g["pct_do_total"] = 100 * g.contrib_mbit / total
    g["pickrate_pct"] = 100 * g.n / n_unit
    g = g.sort_values("contrib_mbit", ascending=False)
    piso = 1 / (2 * N * np.log(2)) * 1e3   # ruído esperado por campeão (mbit)

    # ---- ajuste por time-ano (leave-one-out) ----
    v_u = units.V.to_numpy().astype(float)
    zu = it.encode(*[units[c] for c in CONTEXTOS["time-ano + lado (extra)"]])
    s, n = np.bincount(zu, weights=v_u), np.bincount(zu)
    esp = np.where(n[zu] > 1, (s[zu] - v_u) / np.maximum(n[zu] - 1, 1), v_u.mean())[u_code]
    p = p.assign(esp=esp)
    adj = (p.groupby("champion").V.mean() - p.groupby("champion").esp.mean()) * 100
    g["adj_pp"] = adj
    g["z_raw"] = (g.wr - .5) / np.sqrt(g.wr * (1 - g.wr) / g.n)
    g["z_adj"] = (g.adj_pp / 100) / np.sqrt(g.wr * (1 - g.wr) / g.n)

    # ---- estabilidade: metades aleatórias e temporais ----
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 2, len(units))[u_code]
    rel = _half_table(p, half == 0).join(_half_table(p, half == 1), lsuffix="_A", rsuffix="_B").dropna()
    rel = rel[(rel.size_A >= min_half) & (rel.size_B >= min_half)]
    yr = p.year.to_numpy()
    tem = _half_table(p, yr <= 2023).join(_half_table(p, yr >= 2024), lsuffix="_A", rsuffix="_B").dropna()
    tem = tem[(tem.size_A >= min_half) & (tem.size_B >= min_half)]
    r_rand = np.corrcoef(rel.mean_A, rel.mean_B)[0, 1]
    r_temp = np.corrcoef(tem.mean_A, tem.mean_B)[0, 1]
    same_sign = np.mean(np.sign(rel.mean_A - .5) == np.sign(rel.mean_B - .5))
    g["wr_ate2023"] = tem.mean_A * 100
    g["wr_2024mais"] = tem.mean_B * 100
    g.to_csv(os.path.join(out, "parte2c_contribuicoes.csv"))

    # ---- posições dos campeões do topo ----
    rows = []
    for c in g.head(top).index:
        d = p[p.champion == c].groupby("position").V.agg(["size", "mean"])
        d = d[d["size"] >= min_role].sort_values("size", ascending=False)
        for pos, r in d.iterrows():
            rows.append(dict(campeao=c, posicao=pos, n=int(r["size"]), winrate_pct=100 * r["mean"],
                             ep_pp=100 * np.sqrt(r["mean"] * (1 - r["mean"]) / r["size"])))
    pos_tab = pd.DataFrame(rows)
    pos_tab.to_csv(os.path.join(out, "parte2c_posicoes.csv"), index=False)

    # ---- figura ----
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    a = ax[0]
    t15 = g.head(15)[::-1]
    a.barh(t15.index, t15.contrib_mbit, color=np.where(t15.wr > .5, "C0", "C1"), alpha=.85)
    for i, (c, r) in enumerate(t15.iterrows()):
        a.text(r.contrib_mbit + .001, i, f"{100*r.wr:.1f}%", va="center", fontsize=8)
    a.set_xlabel("contribuição para I(campeão; V) [milibits]")
    a.set_title(f"Top 15 contribuições (azul: win rate > 50%; laranja: < 50%)\npiso de ruído por campeão ≈ {piso:.4f} mbit")
    a.grid(alpha=.3, axis="x")
    topset = set(g.head(top).index)
    for a, d, titulo, r, lx, ly in (
            (ax[1], rel, "Estabilidade: metades aleatórias", r_rand, "metade A", "metade B"),
            (ax[2], tem, "Estabilidade: até 2023 vs 2024+", r_temp, "até 2023", "2024 em diante")):
        x, y = (d.mean_A - .5) * 100, (d.mean_B - .5) * 100
        isTop = d.index.isin(topset)
        a.scatter(x[~isTop], y[~isTop], s=12, alpha=.5)
        a.scatter(x[isTop], y[isTop], s=28, color="C3", label=f"top {top} em contribuição")
        for c in d.index[isTop]:
            a.annotate(c, (x[c], y[c]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = max(abs(x).max(), abs(y).max()) * 1.05
        a.plot([-lim, lim], [-lim, lim], "k--", lw=.8)
        a.axhline(0, color="gray", lw=.6); a.axvline(0, color="gray", lw=.6)
        a.set_xlabel(f"win rate - 50% [p.p.] ({lx})"); a.set_ylabel(f"win rate - 50% [p.p.] ({ly})")
        a.set_title(f"{titulo} (r = {r:.2f}, {len(d)} campeões)"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte2c_picks_detalhe.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Soma das contribuições = {total:.4f} mbit (= I plug-in marginal). Top 10 = {g.contrib_mbit.head(10).sum()/total:.1%}, "
          f"top 20 = {g.contrib_mbit.head(20).sum()/total:.1%}. Piso de ruído por campeão = {piso:.4f} mbit.")
    print(f"Confiabilidade: metades aleatórias r = {r_rand:.2f} ({len(rel)} campeões, mesmo sinal em {same_sign:.0%}); "
          f"temporal r = {r_temp:.2f} ({len(tem)} campeões)")
    show = g.head(top)[["n", "pickrate_pct", "wr", "contrib_mbit", "pct_do_total", "adj_pp", "z_raw", "z_adj", "wr_ate2023", "wr_2024mais"]].copy()
    show["wr"] *= 100
    print(show.round(2).to_string())
    print(pos_tab.round(1).to_string(index=False))
    resumo = dict(r_rand=r_rand, r_temp=r_temp, same_sign=same_sign, piso=piso, total=total,
                            n_rand=len(rel), n_temp=len(tem), top10_share=g.contrib_mbit.head(10).sum() / total,
                  top20_share=g.contrib_mbit.head(20).sum() / total, n_unit=n_unit)
    pd.Series(resumo).to_csv(os.path.join(out, "parte2c_resumo.csv"), header=["valor"])
    return g, pos_tab, resumo
