"""Parte 2b - Aprofundando os campeões: ruído vs força real, ajuste por time, heterogeneidade por patch."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import it_utils as it
from parte2_draft import _units, CONTEXTOS


def eb_shrink(wr, se):
    """Empirical Bayes (método dos momentos): tau = desvio-padrão da força REAL entre campeões."""
    mu = np.average(wr, weights=1 / se**2)
    tau2 = max(np.var(wr, ddof=1) - np.mean(se**2), 0.0)
    w = tau2 / (tau2 + se**2) if tau2 > 0 else np.zeros_like(se)
    return mu + w * (wr - mu), np.sqrt(tau2), w


def run(p, out, min_picks=500, min_patch=40):
    os.makedirs(out, exist_ok=True)
    u_code, units = _units(p)
    v_u = units.V.to_numpy().astype(float)
    zu = it.encode(*[units[c] for c in CONTEXTOS["time-ano + lado (extra)"]])
    s, n = np.bincount(zu, weights=v_u), np.bincount(zu)
    esp = np.where(n[zu] > 1, (s[zu] - v_u) / np.maximum(n[zu] - 1, 1), v_u.mean())[u_code]
    d = p.assign(esp=esp)
    g = d.groupby("champion").agg(n=("V", "size"), wr=("V", "mean"), esp=("esp", "mean"))
    g = g[g.n >= min_picks].copy()
    g["adj"] = g.wr - g.esp
    g["se"] = np.sqrt(g.wr * (1 - g.wr) / g.n)
    g["z_raw"] = (g.wr - .5) / g.se
    g["z_adj"] = g.adj / g.se

    # 1) quanto da dispersão é ruído?
    _, tau_raw, _ = eb_shrink(g.wr.to_numpy(), g.se.to_numpy())
    sh_adj, tau_adj, w_adj = eb_shrink(g.adj.to_numpy() + .5, g.se.to_numpy())
    g["adj_shrunk"] = sh_adj - .5
    sd_obs = g.wr.std(ddof=1)
    # 2) quem muda ao ajustar pelo time
    g["delta"] = g.adj - (g.wr - .5)
    # 3) heterogeneidade por patch (Cochran Q, patches com >= min_patch picks)
    rows = []
    for c in g.index:
        h = d[d.champion == c].groupby("patch").V.agg(["size", "mean"])
        h = h[h["size"] >= min_patch]
        if len(h) < 4:
            continue
        pbar = np.average(h["mean"], weights=h["size"])
        q = np.sum(h["size"] * (h["mean"] - pbar) ** 2) / (pbar * (1 - pbar))
        dfq = len(h) - 1
        rows.append(dict(champion=c, n_patches=len(h), Q=q, df=dfq, p=stats.chi2.sf(q, dfq),
                         excesso=(q - dfq) / dfq))
    het = pd.DataFrame(rows).set_index("champion").sort_values("excesso", ascending=False)
    g.join(het).to_csv(os.path.join(out, "parte2b_campeoes.csv"))
    het.to_csv(os.path.join(out, "parte2b_heterogeneidade_patch.csv"))

    print(f"Campeões com >= {min_picks} picks: {len(g)}")
    print(f"Desvio-padrão OBSERVADO da win rate entre campeões: {sd_obs*100:.2f} p.p.")
    print(f"  ruído amostral médio (SE): {np.sqrt((g.se**2).mean())*100:.2f} p.p.")
    print(f"  força REAL estimada (tau, EB) bruta: {tau_raw*100:.2f} p.p. | ajustada por time-ano: {tau_adj*100:.2f} p.p.")
    print(f"|z|>2 bruto: {(g.z_raw.abs()>2).sum()} | |z|>2 ajustado: {(g.z_adj.abs()>2).sum()}  (esperado por acaso: {0.0455*len(g):.1f})")
    print("\nMaiores mudanças ao ajustar por time-ano (p.p.):")
    t = g.reindex(g.delta.abs().sort_values(ascending=False).index).head(8)
    print((t[["n", "wr", "esp", "adj", "delta"]].assign(wr=lambda x: (x.wr-.5)*100, esp=lambda x: (x.esp-.5)*100,
          adj=lambda x: x.adj*100, delta=lambda x: x.delta*100)).round(2).to_string())
    print(f"\nHeterogeneidade por patch: {(het.p<0.01).sum()} de {len(het)} campeões com p<0.01")
    print(het.head(8).round(3).to_string())

    # ---------------- figura ----------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    a = ax[0]
    a.hist(g.z_raw, bins=30, alpha=.6, density=True, label="z bruto")
    a.hist(g.z_adj, bins=30, alpha=.6, density=True, label="z ajustado (time-ano)")
    xs = np.linspace(-6, 6, 200); a.plot(xs, stats.norm.pdf(xs), "k--", label="N(0,1): só ruído")
    a.set_title("Os campeões são mais dispersos que o acaso?"); a.set_xlabel("z = desvio / erro-padrão"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    lim = 9
    a.errorbar((g.wr - .5) * 100, g.adj_shrunk * 100, xerr=1.96 * g.se * 100, fmt="o", ms=3, alpha=.5, elinewidth=.4,
               label="ajustada + encolhida (EB)")
    a.plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    for c in g.reindex(g.adj_shrunk.abs().sort_values(ascending=False).index).head(8).index:
        a.annotate(c, ((g.wr[c] - .5) * 100, g.adj_shrunk[c] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("win rate bruta - 50% [p.p.]"); a.set_ylabel("estimativa final [p.p.]")
    a.set_title(f"Encolhimento: força real ~ {tau_adj*100:.1f} p.p.\n(observado {sd_obs*100:.1f} p.p.)"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[2]
    top = het[het.n_patches >= 15].head(3).index
    for c in top:
        h = d[d.champion == c].groupby("patch").V.agg(["size", "mean"])
        h = h[h["size"] >= min_patch]
        h = h.reindex(sorted(h.index, key=lambda s: [int(x) if x.isdigit() else 0 for x in s.split(".")]))
        se = np.sqrt(h["mean"] * (1 - h["mean"]) / h["size"])
        a.errorbar(range(len(h)), (h["mean"] - .5) * 100, yerr=1.96 * se * 100, marker="o", ms=3, lw=1, capsize=2, label=c, alpha=.8)
    a.axhline(0, color="gray", lw=.8)
    a.set_xlabel("patches (ordem cronológica)"); a.set_ylabel("win rate - 50% [p.p.]")
    a.set_title("Campeões cuja força mais varia entre patches"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte2b_campeoes_extra.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return g, het
