"""Parte 7 - A T1 em bits: favorito, azarão e a surpresa das séries do Mundial (2022-2025).

Pergunta: as campanhas da T1 no Mundial foram surpreendentes? Quanto, em bits?

1. Força ANTES do jogo: Elo cronológico por ano (todos começam em 1500 em janeiro; K = 20), com as principais ligas,
   MSI e Mundial. Só entram jogos anteriores ao corte, então a força não usa o resultado que se quer prever
   (diferente do "time-ano" das Partes 2 e 4, que usa o ano inteiro).
2. Chance da série: com p = P(vencer um jogo) pelo Elo e jogos independentes, P(vencer uma md5) =
   sum_k C(2+k, k) p^3 (1-p)^k, k = 0..2.
3. Surpresa (autoinformação) do resultado observado: -log2 P(resultado), em bits. A entropia H(P) da série é a
   surpresa MÉDIA que se esperava antes do jogo; surpresa >> H(P) = zebra. Para eventos independentes as surpresas
   se somam: a surpresa de uma campanha é a soma das surpresas das séries.
4. Volcano plot (descritivo): KDA de cada jogador na temporada regular da LCK contra os demais da mesma posição.
   O KDA final é em boa parte consequência das vitórias (Parte 6), então o painel descreve a temporada, não prevê.

Atenção: no Oracle's Elixir a liga "WLDs" inclui as SELETIVAS regionais (ex.: LCK, setembro de 2024). O início do
Mundial propriamente dito é o primeiro jogo da "WLDs" entre times de ligas diferentes.
"""
import os
from math import comb, log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LIGAS_ELO = ["LCK", "LPL", "LEC", "LCS", "MSI", "WLDs", "LTA", "LTA N", "LTA S", "LCP", "PCS", "VCS", "CBLOL", "LJL", "LLA"]
INTERNACIONAIS = {"MSI", "WLDs", "EWC"}
ANOS = (2022, 2023, 2024, 2025)
TIME = "T1"
# rival em destaque no volcano plot de cada ano (o rival da narrativa)
DESTAQUE = {2022: ["Kiwoom DRX"], 2023: ["Gen.G"], 2024: ["Gen.G"], 2025: ["Gen.G", "KT Rolster"]}
FASE = {(2022, "Royal Never Give Up"): "quartas", (2022, "JD Gaming"): "semi", (2022, "Kiwoom DRX"): "final",
        (2023, "LNG Esports"): "quartas", (2023, "JD Gaming"): "semi", (2023, "Weibo Gaming"): "final",
        (2024, "Dplus Kia"): "seletiva LCK", (2024, "KT Rolster"): "seletiva LCK", (2024, "Top Esports"): "quartas",
        (2024, "Gen.G"): "semi", (2024, "Bilibili Gaming"): "final",
        (2025, "Invictus Gaming"): "play-in", (2025, "Anyone's Legend"): "quartas", (2025, "Top Esports"): "semi",
        (2025, "KT Rolster"): "final"}


# --------------------------------------------------------------------------- #
def _times(raw):
    t = raw[(raw.position == "team") & raw.result.notna() & raw.date.notna()][
        ["gameid", "date", "year", "league", "teamname", "side", "result"]].copy()
    t["date"] = pd.to_datetime(t.date)
    return t


def _jogos(T, y, ligas=LIGAS_ELO):
    """Uma linha por jogo (azul x vermelho), em ordem cronológica."""
    t = T[(T.year == y) & T.league.isin(ligas)]
    b = t[t.side == "Blue"].set_index("gameid")
    r = t[t.side == "Red"].set_index("gameid")
    g = b.join(r, lsuffix="_b", rsuffix="_r", how="inner").reset_index()
    g["date"] = g.date_b
    return g.sort_values("date").reset_index(drop=True)


def elo(g, ate=None, K=20):
    """Elo depois de todos os jogos de g ANTERIORES a 'ate' (exclusivo)."""
    E = {}
    for a, b, rb, d in zip(g.teamname_b, g.teamname_r, g.result_b, g.date):
        if ate is not None and d >= ate:
            break
        ea, eb = E.get(a, 1500.0), E.get(b, 1500.0)
        pa = 1 / (1 + 10 ** ((eb - ea) / 400))
        E[a] = ea + K * (rb - pa)
        E[b] = eb - K * (rb - pa)
    return E


def _casas(T):
    """Liga de origem de cada (ano, time): a liga não internacional em que o time mais jogou no ano."""
    t = T[~T.league.isin(INTERNACIONAIS)]
    return t.groupby(["year", "teamname"]).league.agg(lambda s: s.value_counts().idxmax()).to_dict()


def elo_continuo(T, ate, K=20, retencao=0.75, desde=2014):
    """Elo contínuo desde 2014, só com jogos anteriores a 'ate'.

    Na virada do ano cada time regride em direção à MÉDIA DA SUA LIGA (mantém 'retencao' da distância), e um time
    novo entra com a média atual da sua liga. Assim a força das ligas, aprendida nos jogos de MSI e Mundial, se
    acumula entre os anos. O Elo que recomeça em 1500 todo ano (elo()) não compara ligas: um time que domina uma
    liga fraca fica com Elo alto, e as ligas só se encontram nos poucos jogos internacionais.
    """
    casa = _casas(T)
    t = T[T.league.isin(LIGAS_ELO) & (T.year >= desde) & (T.year <= ate.year)]
    b = t[t.side == "Blue"].set_index("gameid")
    r = t[t.side == "Red"].set_index("gameid")
    g = b.join(r, lsuffix="_b", rsuffix="_r", how="inner").sort_values("date_b")
    E, ano = {}, None

    def media_liga(y, lg):
        v = [e for k, e in E.items() if casa.get((y, k)) == lg]
        return float(np.mean(v)) if v else 1500.0

    for a, bb, rb, d, y in zip(g.teamname_b, g.teamname_r, g.result_b, g.date_b, g.year_b):
        if d >= ate:
            break
        if y != ano:
            if ano is not None:                      # virada do ano: regride para a média da liga do ano anterior
                medias = {}
                for k in E:
                    lg = casa.get((ano, k))
                    if lg is not None and lg not in medias:
                        medias[lg] = media_liga(ano, lg)
                for k in E:
                    lg = casa.get((ano, k))
                    if lg in medias:
                        E[k] = medias[lg] + retencao * (E[k] - medias[lg])
            ano = y
        for k in (a, bb):
            if k not in E:
                E[k] = media_liga(y, casa.get((y, k)))
        pa = 1 / (1 + 10 ** ((E[bb] - E[a]) / 400))
        E[a] += K * (rb - pa)
        E[bb] -= K * (rb - pa)
    return E


def p_jogo(ea, eb):
    return 1 / (1 + 10 ** ((eb - ea) / 400))


def p_serie(p, n):
    """P(vencer uma melhor-de-n), jogos independentes com P(vencer um jogo) = p."""
    v = n // 2 + 1
    return sum(comb(v - 1 + k, k) * p ** v * (1 - p) ** k for k in range(v))


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * log2(p) + (1 - p) * log2(1 - p))


def inicio_mundial(T, y):
    """Primeiro jogo da 'WLDs' entre times de ligas de origem diferentes (exclui as seletivas regionais)."""
    casa = (T[(T.year == y) & ~T.league.isin(INTERNACIONAIS)]
            .groupby("teamname").league.agg(lambda s: s.value_counts().idxmax()))
    g = _jogos(T, y, ["WLDs"])
    diff = g.teamname_b.map(casa).fillna("?") != g.teamname_r.map(casa).fillna("!")
    return g.loc[diff, "date"].min()


def series_do_time(T, y, time=TIME, gap_h=8):
    """Séries do time na 'WLDs' do ano: jogos seguidos contra o mesmo rival (intervalo < gap_h horas)."""
    g = _jogos(T, y, ["WLDs"])
    g = g[(g.teamname_b == time) | (g.teamname_r == time)].copy()
    g["rival"] = np.where(g.teamname_b == time, g.teamname_r, g.teamname_b)
    g["venceu"] = np.where(g.teamname_b == time, g.result_b, 1 - g.result_b)
    out = []
    for rv, s in g.groupby("rival"):
        s = s.sort_values("date")
        bloco = (s.date.diff() > pd.Timedelta(hours=gap_h)).cumsum()
        for _, x in s.groupby(bloco):
            vt, vr = int(x.venceu.sum()), int(len(x) - x.venceu.sum())
            v = max(vt, vr)
            out.append(dict(ano=y, rival=rv, inicio=x.date.min(), jogos=len(x), placar=f"{vt}-{vr}",
                            n_md=2 * v - 1, time_venceu=vt > vr))
    return pd.DataFrame(out).sort_values("inicio").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def participantes_mundial(T, y):
    g = _jogos(T, y, ["WLDs"])
    g = g[g.date >= inicio_mundial(T, y)]
    return set(g.teamname_b) | set(g.teamname_r)


def surpresas(T):
    """Séries eliminatórias da T1. Método principal: Elo contínuo; comparação: Elo que recomeça todo ano."""
    linhas, rank = [], []
    for y in ANOS:
        g = _jogos(T, y)
        ini = inicio_mundial(T, y)
        E0 = elo_continuo(T, ini)
        part = participantes_mundial(T, y)
        top = sorted(((k, v) for k, v in E0.items() if k in part), key=lambda kv: -kv[1])
        for pos, (k, v) in enumerate(top, 1):
            rank.append(dict(ano=y, posicao=pos, time=k, elo_pre_mundial=round(v), participantes=len(top)))
        for _, s in series_do_time(T, y).iterrows():
            if s.n_md < 5:          # md1/md3 da fase suíça ficam fora: o foco são as séries eliminatórias
                continue
            for momento, corte in (("antes do Mundial", ini), ("antes da série", s.inicio)):
                if momento == "antes do Mundial" and s.inicio < ini:
                    continue        # seletiva regional: acontece antes do Mundial
                E = elo_continuo(T, corte)
                Ea = elo(g, ate=corte)
                ea, eb = E.get(TIME, 1500.0), E.get(s.rival, 1500.0)
                P = p_serie(p_jogo(ea, eb), s.n_md)
                P_anual = p_serie(p_jogo(Ea.get(TIME, 1500.0), Ea.get(s.rival, 1500.0)), s.n_md)
                pobs = P if s.time_venceu else 1 - P
                pobs_a = P_anual if s.time_venceu else 1 - P_anual
                linhas.append(dict(ano=y, fase=FASE.get((y, s.rival), "?"), rival=s.rival, momento=momento,
                                   placar=s.placar, elo_T1=round(ea), elo_rival=round(eb), p_jogo=p_jogo(ea, eb),
                                   p_serie_T1=P, T1_venceu=bool(s.time_venceu), surpresa_bits=-log2(pobs),
                                   entropia_bits=h2(P), p_serie_T1_elo_anual=P_anual,
                                   surpresa_bits_elo_anual=-log2(pobs_a), inicio_serie=s.inicio))
    return pd.DataFrame(linhas), pd.DataFrame(rank)


def campanhas(sp):
    """Surpresa acumulada das séries eliminatórias no Mundial (soma das surpresas 'antes da série')."""
    s = sp[(sp.momento == "antes da série") & ~sp.fase.str.startswith("seletiva")]
    return (s.groupby("ano")
             .agg(series=("rival", "size"), surpresa_total_bits=("surpresa_bits", "sum"),
                  entropia_total_bits=("entropia_bits", "sum"),
                  campea=("T1_venceu", "all"))
             .assign(p_do_que_aconteceu=lambda d: 2.0 ** -d.surpresa_total_bits)   # surpresas somam = probabilidades multiplicam
             .reset_index())


# --------------------------------------------------------------------------- #
def volcano(raw):
    P = raw[(raw.position != "team") & (raw.league == "LCK") & raw.playername.notna() & (raw.playoffs == 0)
            & raw.year.isin(ANOS)].copy()
    P["kda"] = (P.kills + P.assists) / P.deaths.clip(lower=1)
    out = []
    for y in ANOS:
        d = P[P.year == y]
        rows = []
        for (pl, pos, tm), g in d.groupby(["playername", "position", "teamname"]):
            if len(g) < 30:
                continue
            rest = d[(d.position == pos) & (d.playername != pl)].kda
            diff = g.kda.mean() - rest.mean()
            sd = np.sqrt((g.kda.var() + rest.var()) / 2)
            rows.append(dict(ano=y, jogador=pl, pos=pos, time=tm, n=len(g), kda=g.kda.mean(), dif=diff,
                             d=diff / sd, p=stats.mannwhitneyu(g.kda, rest).pvalue))
        r = pd.DataFrame(rows)
        r["nlp"] = -np.log10(r.p.clip(lower=1e-300))
        r["bonferroni"] = r.p < 0.05 / len(r)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def resumo_volcano(V):
    rows = []
    for y in ANOS:
        r = V[V.ano == y]
        for tm in [TIME] + DESTAQUE[y]:
            s = r[r.time == tm]
            rows.append(dict(ano=y, time=tm, jogadores=len(s), d_medio=s.d.mean(),
                             acima_bonferroni=int((s.bonferroni & (s.d > 0)).sum()),
                             abaixo_de_zero=int((s.d < 0).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _figuras(sp, camp, V, out):
    # (a) surpresa das séries eliminatórias
    s = sp[sp.momento == "antes da série"].reset_index(drop=True)
    sigla = {"Royal Never Give Up": "RNG", "JD Gaming": "JDG", "Kiwoom DRX": "DRX", "LNG Esports": "LNG",
             "Weibo Gaming": "WBG", "Dplus Kia": "DK", "KT Rolster": "KT", "Top Esports": "TES", "Gen.G": "GEN",
             "Bilibili Gaming": "BLG", "Invictus Gaming": "IG", "Anyone's Legend": "AL"}
    fase = {"quartas": "QF", "semi": "SF", "final": "F", "seletiva LCK": "sel.", "play-in": "PI"}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(s))
    cor = np.where(s.T1_venceu, "#2b6cb0", "#c53030")
    ax.bar(x, s.surpresa_bits, color=cor, alpha=.9, label="surpresa do resultado, −log₂ P(resultado)")
    ax.scatter(x, s.entropia_bits, color="k", marker="_", s=400, zorder=3, label="surpresa média esperada, H(P)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{fase.get(f, f)} {sigla.get(r, r)}\n{p}" for a, f, r, p in zip(s.ano, s.fase, s.rival, s.placar)],
                       fontsize=8)
    for xi, (pp, b) in enumerate(zip(s.p_serie_T1, s.surpresa_bits)):
        ax.text(xi, b + .05, f"{pp:.0%}", ha="center", fontsize=8)
    ax.set_xlabel("ano, fase (sel. = seletiva, PI = play-in, QF = quartas, SF = semi, F = final), rival e placar da T1; "
                  "acima da barra: P(T1 vencer) antes da série", fontsize=8)
    ax.set_ylabel("bits")
    ax.set_title("T1 no Mundial: surpresa de cada série eliminatória (azul = T1 venceu; vermelho = perdeu)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "parte7_surpresa_series.png"), dpi=150)
    plt.close()

    # (b) volcano 2x2
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    cores = {TIME: "#d62728", "Kiwoom DRX": "#1f77b4", "Gen.G": "#ff7f0e", "KT Rolster": "#6b46c1"}
    for a, y in zip(axs.ravel(), ANOS):
        r = V[V.ano == y]
        a.scatter(r.d, r.nlp, c="#bbbbbb", s=24)
        for tm in [TIME] + DESTAQUE[y]:
            q = r[r.time == tm]
            a.scatter(q.d, q.nlp, c=cores.get(tm, "k"), s=60, label=tm, edgecolor="k")
            for _, z in q.iterrows():
                a.annotate(z.jogador, (z.d, z.nlp), xytext=(4, 3), textcoords="offset points", fontsize=8)
        a.axhline(-np.log10(0.05 / len(r)), ls="--", c="gray", lw=.8)
        a.axvline(0, c="gray", lw=.8)
        a.set_title(f"LCK {y} (temporada regular)", fontsize=10)
        a.set_xlabel("KDA do jogador − KDA da posição (em desvios-padrão)", fontsize=8)
        a.set_ylabel("−log10(p)  [Mann-Whitney]", fontsize=8)
        a.legend(fontsize=8)
    fig.suptitle("Volcano plot: KDA de cada jogador contra os demais da mesma posição (tracejado = Bonferroni)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "parte7_volcano_kda.png"), dpi=150)
    plt.close()


def run(raw, out):
    os.makedirs(out, exist_ok=True)
    T = _times(raw)
    sp, rank = surpresas(T)
    camp = campanhas(sp)
    V = volcano(raw)
    rv = resumo_volcano(V)
    sp.drop(columns="inicio_serie").to_csv(os.path.join(out, "parte7_surpresa_series.csv"), index=False)
    rank.to_csv(os.path.join(out, "parte7_elo_pre_mundial.csv"), index=False)
    camp.to_csv(os.path.join(out, "parte7_campanhas.csv"), index=False)
    V.to_csv(os.path.join(out, "parte7_volcano_kda.csv"), index=False)
    rv.to_csv(os.path.join(out, "parte7_volcano_resumo.csv"), index=False)
    _figuras(sp, camp, V, out)

    print("Início do Mundial (1º jogo entre ligas diferentes):",
          {y: str(inicio_mundial(T, y).date()) for y in ANOS})
    print("\nSéries eliminatórias da T1 (surpresa em bits):")
    print(sp.drop(columns="inicio_serie").round(3).to_string(index=False))
    print("\nCampanhas:")
    print(camp.round(3).to_string(index=False))
    print("\nElo contínuo antes do Mundial (top 8 entre os participantes):")
    print(rank[rank.posicao <= 8].pivot(index="posicao", columns="ano", values="time").to_string())
    print("Posição da T1:", rank[rank.time == TIME][["ano", "posicao", "participantes", "elo_pre_mundial"]].to_string(index=False))
    print("\nVolcano (resumo por time):")
    print(rv.round(3).to_string(index=False))
    return sp, camp, rank, V, rv
