"""Parte 2 - Um campeão parece forte porque é bom ou por causa do contexto?

Compara I(Campeão ; V) com I(Campeão ; V | Z) para vários contextos Z.
Unidade de análise: cada pick (partida, time, campeão); V = vitória do time.

Como há ~170 campeões e muitos estratos, o estimador plug-in é MUITO enviesado
para cima. Reportamos três versões:
  - plug-in (ingênuo)
  - Miller-Madow (correção de 1a ordem; falha quando os estratos são esparsos)
  - plug-in menos a média sob a hipótese nula, obtida embaralhando V dentro de
    cada estrato Z (teste de permutação; a correção mais robusta aqui)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import it_utils as it


# --------------------------------------------------------------------------- #
def _units(p):
    """Tabela de unidades (partida, lado): V e atributos de contexto."""
    u_code = it.encode(p.gameid, p.side)
    first = pd.Series(np.arange(len(p))).groupby(u_code).first().to_numpy()
    units = p.iloc[first][["gameid", "V", "side", "patch", "league", "team_year"]].reset_index(drop=True)
    return u_code, units


CONTEXTOS = {
    "nenhum (marginal)": None,
    "lado": ["side"],
    "patch": ["patch"],
    "liga": ["league"],
    "patch + liga + lado": ["patch", "league", "side"],
    "time-ano + lado (extra)": ["team_year", "side"],
}


def _permutador(units, zu, cols):
    """Devolve f(rng) -> V embaralhado por unidade (partida, lado), sob H0: campeão independente de V dado Z.

    Contexto SEM o lado (nenhum, patch, liga): as duas unidades de uma partida estão no mesmo estrato e
    têm V opostos. Embaralhar unidade por unidade quebraria esse pareamento: o nulo ficaria ~5 graus de
    liberdade por estrato abaixo do real e a informação corrigida sairia inflada (~1 mbit em "patch").
    Por isso embaralha-se o resultado do lado azul entre PARTIDAS do mesmo estrato, e o vermelho é 1 - azul.
    Contexto COM o lado: as unidades de um estrato são de partidas diferentes, e embaralhar unidades dentro
    do estrato já é exato.
    """
    v_u = units.V.to_numpy()
    strata_u = np.zeros(len(units), dtype=np.int64) if zu is None else zu
    por_unidade = lambda rng: it.permute_within(strata_u, v_u, rng)
    if cols is not None and "side" in cols:
        return por_unidade
    gid = it.encode(units.gameid)
    azul = (units.side == "Blue").to_numpy()
    G = int(gid.max()) + 1
    pareado = (np.bincount(gid, minlength=G) == 2) & (np.bincount(gid[azul], minlength=G) == 1)
    um_vencedor = np.bincount(gid, weights=v_u, minlength=G) == 1
    ok = pareado & um_vencedor               # há erros na base (ex.: partidas com os dois lados marcados como derrota)
    if (~ok).any():
        print(f"  {int((~ok).sum())} partida(s) sem vencedor único ficam fora do embaralhamento (V mantido)")
    vb = np.zeros(G, dtype=v_u.dtype)
    zg = np.zeros(G, dtype=np.int64)
    vb[gid[azul]] = v_u[azul]
    zg[gid[azul]] = strata_u[azul]          # contexto da partida (igual nos dois lados)
    jogos_ok = np.where(ok)[0]
    unid_ok = ok[gid]

    def por_partida(rng):
        pv = vb.copy()
        pv[jogos_ok] = it.permute_within(zg[jogos_ok], vb[jogos_ok], rng)
        return np.where(unid_ok, np.where(azul, pv[gid], 1 - pv[gid]), v_u)
    return por_partida


def _info_table(p, B, seed, u_code, units):
    x = it.encode(p.champion)
    rng = np.random.default_rng(seed)
    rows = []
    for nome, cols in CONTEXTOS.items():
        zu = None if cols is None else it.encode(*[units[c] for c in cols])
        zp = None if zu is None else zu[u_code]
        cmi = it.CondMI(x, zp)
        obs, obs_mm = cmi(p.V.to_numpy()), cmi(p.V.to_numpy(), mm=True)
        embaralha = _permutador(units, zu, cols)
        nulls = np.empty(B)
        for b in range(B):
            nulls[b] = cmi(embaralha(rng)[u_code])
        rows.append(dict(
            contexto=nome,
            n_estratos=1 if zu is None else int(zu.max()) + 1,
            I_plugin=obs, I_MM=obs_mm,
            I_nulo_media=nulls.mean(), I_nulo_p95=np.quantile(nulls, .95),
            I_corrigida_perm=obs - nulls.mean(),
            p_valor=(1 + np.sum(nulls >= obs)) / (B + 1),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _meias_amostras(p, n_meias, perm, seed):
    """Estimativas corrigidas em meias-amostras, para o erro-padrão (jackknife delete-n/2).

    Sorteia metade das PARTIDAS sem reposição (os dois lados juntos) e, na metade, recalcula o plug-in
    e o nulo por permutação. Sem reposição não há linhas duplicadas, então a correção por permutação
    continua válida em cada metade (o bootstrap com reposição duplicaria picks e inflaria o plug-in).
    Com d = n/2 o fator (n - d)/d vale 1: a variância entre metades já estima a variância do
    estimador com n completo. O ruído de Monte Carlo do nulo (perm finito) só torna o IC mais largo.
    """
    rng = np.random.default_rng(seed + 101)
    jogos = pd.unique(p.gameid)
    nomes = list(CONTEXTOS)
    est = np.empty((n_meias, len(nomes)))
    for s in range(n_meias):
        escolhidos = rng.choice(jogos, len(jogos) // 2, replace=False)
        sub = p[p.gameid.isin(escolhidos)].reset_index(drop=True)
        uc, un = _units(sub)
        x = it.encode(sub.champion)
        for j, cols in enumerate(CONTEXTOS.values()):
            zu = None if cols is None else it.encode(*[un[c] for c in cols])
            cmi = it.CondMI(x, None if zu is None else zu[uc])
            embaralha = _permutador(un, zu, cols)
            nulo = np.mean([cmi(embaralha(rng)[uc]) for _ in range(perm)])
            est[s, j] = cmi(sub.V.to_numpy()) - nulo
    return pd.DataFrame(est, columns=nomes)


def _bias_curve(p, u_code, units, B, seed, fracs=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0)):
    """Como o viés some quando N cresce (I plug-in, MM e corrigida por permutação)."""
    rng = np.random.default_rng(seed)
    jogos = pd.unique(units.gameid)
    rows = []
    for ctx in ("nenhum (marginal)", "patch + liga + lado"):
        cols = CONTEXTOS[ctx]
        for f in fracs:
            # sorteia PARTIDAS (os dois lados juntos), para o nulo pareado continuar valendo
            keep = rng.choice(jogos, int(f * len(jogos)), replace=False)
            sub = p[p.gameid.isin(keep)].reset_index(drop=True)
            uc, un = _units(sub)  # unidades da subamostra
            zu = None if cols is None else it.encode(*[un[c] for c in cols])
            zp = None if zu is None else zu[uc]
            cmi = it.CondMI(it.encode(sub.champion), zp)
            obs, obs_mm = cmi(sub.V.to_numpy()), cmi(sub.V.to_numpy(), mm=True)
            embaralha = _permutador(un, zu, cols)
            nul = np.mean([cmi(embaralha(rng)[uc]) for _ in range(B)])
            rows.append(dict(contexto=ctx, frac=f, n_picks=len(sub), I_plugin=obs, I_MM=obs_mm,
                             I_nulo_media=nul, I_corrigida_perm=obs - nul))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _per_champion(p, u_code, units, min_picks):
    """Win rate bruta vs. win rate esperada dada o contexto (leave-one-out por unidade)."""
    v_u = units.V.to_numpy().astype(float)
    out = {}
    for nome in ("patch + liga + lado", "time-ano + lado (extra)"):
        zu = it.encode(*[units[c] for c in CONTEXTOS[nome]])
        s = np.bincount(zu, weights=v_u)
        n = np.bincount(zu)
        loo = np.where(n[zu] > 1, (s[zu] - v_u) / np.maximum(n[zu] - 1, 1), v_u.mean())
        out[nome] = loo[u_code]  # esperado por pick
    d = pd.DataFrame({"champion": p.champion.to_numpy(), "V": p.V.to_numpy(),
                      **{f"esp_{k}": v for k, v in out.items()}})
    g = d.groupby("champion").agg(n=("V", "size"), winrate=("V", "mean"),
                                  **{f"esp_{k}": (f"esp_{k}", "mean") for k in out})
    g = g[g.n >= min_picks].copy()
    for k in out:
        g[f"ajust_{k}"] = g.winrate - g[f"esp_{k}"]
    g["se"] = np.sqrt(g.winrate * (1 - g.winrate) / g.n)
    # contribuição de cada campeão para I(X;V):  p(c) * D( P(V|c) || P(V) )
    pv = p.V.mean()
    wr = np.clip(g.winrate, 1e-9, 1 - 1e-9)
    kl = wr * np.log2(wr / pv) + (1 - wr) * np.log2((1 - wr) / (1 - pv))
    g["contrib_bits"] = g.n / len(p) * kl
    return g.sort_values("contrib_bits", ascending=False)


# --------------------------------------------------------------------------- #
def run(picks, out, B=100, seed=0, min_picks=500, bias_B=10, n_meias=50, perm_meias=10):
    os.makedirs(out, exist_ok=True)
    u_code, units = _units(picks)
    print(f"  {len(picks):,} picks, {len(units):,} (partida, lado), {picks.champion.nunique()} campeões")

    tab = _info_table(picks, B, seed, u_code, units)
    # IC 95% da informação corrigida e diferenças entre contextos (pareadas na mesma meia-amostra)
    tab["I_corrigida_ep"] = np.nan
    tab["I_corrigida_media_meias"] = np.nan
    if n_meias > 1:
        print(f"  IC por meias-amostras ({n_meias} metades x {perm_meias} permutações)...")
        meias = _meias_amostras(picks, n_meias, perm_meias, seed)
        tab["I_corrigida_ep"] = tab.contexto.map(meias.std(ddof=1)).to_numpy()
        # diagnóstico de viés: se a estimativa com n/2 for sistematicamente menor que com n, o estimador
        # ainda cresce com a amostra (células esparsas "escondem" o sinal) e o valor com n é subestimado
        tab["I_corrigida_media_meias"] = tab.contexto.map(meias.mean()).to_numpy()
        base = tab.set_index("contexto").I_corrigida_perm
        dif = meias.sub(meias["nenhum (marginal)"], axis=0)
        difs = pd.DataFrame(dict(contexto=list(CONTEXTOS),
                                 dif_vs_marginal=[base[c] - base["nenhum (marginal)"] for c in CONTEXTOS],
                                 ep=dif.std(ddof=1).to_numpy()))
        difs["IC95_lo"] = difs.dif_vs_marginal - 1.96 * difs.ep
        difs["IC95_hi"] = difs.dif_vs_marginal + 1.96 * difs.ep
        difs.to_csv(os.path.join(out, "parte2_diferenca_vs_marginal.csv"), index=False)
    tab["IC95_lo"] = tab.I_corrigida_perm - 1.96 * tab.I_corrigida_ep
    tab["IC95_hi"] = tab.I_corrigida_perm + 1.96 * tab.I_corrigida_ep
    tab.to_csv(os.path.join(out, "parte2_informacao_campeao.csv"), index=False)
    bias = _bias_curve(picks, u_code, units, bias_B, seed)
    bias.to_csv(os.path.join(out, "parte2_vies_vs_N.csv"), index=False)
    champ = _per_champion(picks, u_code, units, min_picks)
    champ.to_csv(os.path.join(out, "parte2_por_campeao.csv"))

    # ---------------- figura 1: bits por contexto ----------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    a = ax[0]
    idx = np.arange(len(tab)); w = .2
    a.bar(idx - 1.5 * w, tab.I_plugin * 1e3, w, color="C0", label="plug-in")
    a.bar(idx - .5 * w, tab.I_MM * 1e3, w, color="C1", label="Miller-Madow")
    a.bar(idx + .5 * w, tab.I_nulo_media * 1e3, w, color="gray", label="ruído esperado (nulo por permutação)")
    a.bar(idx + 1.5 * w, tab.I_corrigida_perm.clip(lower=0) * 1e3, w, color="C3", label="corrigida = plug-in - nulo (IC 95%)")
    # IC centrado no valor real (a barra é cortada em zero, o intervalo não)
    a.errorbar(idx + 1.5 * w, tab.I_corrigida_perm * 1e3, yerr=1.96 * tab.I_corrigida_ep.fillna(0) * 1e3,
               fmt="none", ecolor="k", capsize=2)
    a.set_yscale("symlog", linthresh=1)
    a.set_xticks(idx); a.set_xticklabels(tab.contexto, rotation=25, ha="right")
    a.set_ylabel("milibits (escala symlog)"); a.set_title("I(campeão ; V | contexto)")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="y")

    a = ax[1]
    cores = {"I_plugin": "C0", "I_MM": "C1", "I_corrigida_perm": "C3"}
    nomes = {"I_plugin": "plug-in", "I_MM": "Miller-Madow", "I_corrigida_perm": "corrigida (permutação)"}
    marc = {"nenhum (marginal)": ("o", "-"), "patch + liga + lado": ("s", "--")}
    for ctx, (mk, ls) in marc.items():
        b = bias[bias.contexto == ctx]
        for col, cor in cores.items():
            a.plot(b.n_picks, b[col].clip(lower=0) * 1e3, marker=mk, ls=ls, color=cor, ms=5)
    for col, cor in cores.items():
        a.plot([], [], color=cor, label=nomes[col])
    for ctx, (mk, ls) in marc.items():
        a.plot([], [], color="k", marker=mk, ls=ls, label=f"contexto: {ctx}")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=1); a.set_ylim(bottom=0)
    a.set_xlabel("nº de picks na amostra"); a.set_ylabel("milibits (escala symlog)")
    a.set_title("O viés do plug-in só diminui com N muito grande"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte2_informacao.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figura 2: win rate bruta vs ajustada ----------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for a, nome in zip(ax, ("patch + liga + lado", "time-ano + lado (extra)")):
        x = (champ.winrate - .5) * 100
        y = champ[f"ajust_{nome}"] * 100
        a.errorbar(x, y, xerr=1.96 * champ.se * 100, fmt="o", ms=3, alpha=.6, elinewidth=.5)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.1
        a.plot([-lim, lim], [-lim, lim], "k--", lw=.8, label="y = x (contexto não explica nada)")
        a.axhline(0, color="gray", lw=.6); a.axvline(0, color="gray", lw=.6)
        for c in champ.reindex((champ.winrate - .5).abs().sort_values(ascending=False).index).head(8).index:
            a.annotate(c, (x[c], y[c]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        corr = np.corrcoef(x, y)[0, 1]
        a.set_title(f"Contexto = {nome}  (r = {corr:.2f})")
        a.set_xlabel("win rate bruta - 50% [p.p.]"); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_ylabel("win rate - esperada dado o contexto [p.p.]")
    fig.suptitle(f"Campeões com >= {min_picks} picks: o contexto 'come' parte da força aparente?")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte2_campeoes.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== PARTE 2 (milibits) ===")
    show = tab.copy()
    for c in ("I_plugin", "I_MM", "I_nulo_media", "I_nulo_p95", "I_corrigida_perm", "I_corrigida_ep",
              "IC95_lo", "IC95_hi", "I_corrigida_media_meias"):
        show[c] = show[c] * 1e3
    print(show.round(3).to_string(index=False))
    if n_meias > 1:
        print("\nDiferença contra o marginal (mbit, IC 95% pareado por meia-amostra):")
        print((difs.set_index("contexto") * 1e3).round(3).to_string())
    print("\nTop 10 campeões por contribuição para I(X;V) [milibits]:")
    t = champ.head(10).copy(); t["contrib_bits"] *= 1e3
    print(t[["n", "winrate", "contrib_bits"]].round(4).to_string())
    return tab, bias, champ
