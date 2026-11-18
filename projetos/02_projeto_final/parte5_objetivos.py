"""Parte 5 - Objetivos (first blood, 1ª torre, 1º dragão, arauto, barão...) informam sobre a vitória?

A base (Oracle's Elixir) NÃO tem timestamp dos objetivos. Por isso as variáveis são separadas por quando existem:

  * INICIAIS (valem para prever no min 15):  first blood, 1º dragão, 1º arauto*, 1ª torre
        (*arauto: min 8 até 2023, min 14 a partir de 2024 -> é tardio na era nova)
  * TARDIAS / AMBÍGUAS (ocorrem muitas vezes depois do min 15): 1ª torre do meio, 1 a 3 torres, 1º barão
  * PÓS-FATO (totais do fim do jogo, "vazamento"): dragões, arautos, barões, torres, inibidores
  * Placas de torre (2022+) e vermes do vazio (2024+): saturam até ~min 14-15, então são quase variáveis do min 15.

Análises (min 15, mesma amostra para todas):
  1. I(objetivo ; V) marginal, com correção por permutação.
  2. I(objetivo ; V | faixa de ouro) exato, com permutação (o que o objetivo acrescenta depois do ouro).
  3. Regra da cadeia por log-loss (gradient boosting, 5-fold CV), na ordem "ouro primeiro" e "objetivos primeiro".
  4. O que muda se entrarem os totais do FIM do jogo (vazamento).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import it_utils as it

BASE15 = ["gd15", "xd15", "cd15", "kd15"]
INICIAIS = ["fb", "fd", "fh", "ft"]
TARDIAS = ["fmt", "f3t", "fbar"]
NOME = {"gd15": "dif. de ouro (min 15)", "xd15": "dif. de XP", "cd15": "dif. de CS", "kd15": "dif. de abates",
        "fb": "first blood", "fd": "1º dragão", "fh": "1º arauto", "ft": "1ª torre",
        "fmt": "1ª torre do meio", "f3t": "1ª a derrubar 3 torres", "fbar": "1º barão",
        "pl": "placas de torre (dif.)", "gr": "vermes do vazio (dif.)",
        "dd": "dragões no fim (dif.)", "hd": "arautos no fim (dif.)", "bd": "barões no fim (dif.)",
        "td": "torres no fim (dif.)", "id": "inibidores no fim (dif.)"}
GRUPO = {**{k: "inicial" for k in INICIAIS}, **{k: "tardia" for k in TARDIAS}}


def _hgb(seed):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def _cv_loss(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_hgb(seed), X, y, cv=cv, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def _loss0(y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(DummyClassifier(strategy="prior"), np.zeros((len(y), 1)), y, cv=cv,
                          method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def _taxa_quem_pega(o, f):
    """P(vence | fez o objetivo primeiro), juntando os dois lados."""
    s = o[o[f].notna()]
    n1 = s[f] == 1
    return float(((s.V[n1] == 1).sum() + (s.V[~n1] == 0).sum()) / len(s))


# --------------------------------------------------------------------------- #
def marginais(o, B, seed):
    """I(objetivo; V) marginal, em cada amostra em que o objetivo existe (uso máximo dos dados)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in INICIAIS + TARDIAS:
        s = o[o[f].notna()]
        y, x = s.V.to_numpy(), s[f].to_numpy()
        cmi = it.CondMI(x)
        obs = cmi(y)
        nul = np.array([cmi(rng.permutation(y)) for _ in range(B)])
        rows.append(dict(objetivo=NOME[f], grupo=GRUPO[f], N=len(s), P_vence_se_faz_primeiro=_taxa_quem_pega(o, f),
                         H_V=it.entropy(y), I_plugin_mbit=obs * 1e3, I_nulo_mbit=nul.mean() * 1e3,
                         I_corrigida_mbit=(obs - nul.mean()) * 1e3,
                         p_valor=(1 + np.sum(nul >= obs)) / (B + 1)))
    return pd.DataFrame(rows)


def dado_ouro(s, bins, B, seed, rotulo="todas"):
    """I(objetivo; V) e I(objetivo; V | faixa de ouro no min 15), exatos, com permutação dentro das faixas."""
    rng = np.random.default_rng(seed)
    y = s.V.to_numpy()
    z = it.qbin(s.gd15, bins)
    rows = []
    # referência: o próprio ouro
    cg = it.CondMI(z)
    og = cg(y)
    ng = np.array([cg(rng.permutation(y)) for _ in range(B)])
    rows.append(dict(era=rotulo, objetivo="[ref] ouro min 15 (%d faixas)" % bins, N=len(s), I_marg_mbit=(og - ng.mean()) * 1e3,
                     I_dado_ouro_mbit=np.nan, I_dado_ouro_corr_mbit=np.nan, p_valor=np.nan, fracao_que_resta=np.nan))
    for f in INICIAIS + TARDIAS:
        x = s[f].to_numpy()
        c0, c1 = it.CondMI(x), it.CondMI(x, z)
        o0, o1 = c0(y), c1(y)
        n0 = np.array([c0(rng.permutation(y)) for _ in range(B)])
        n1 = np.array([c1(it.permute_within(z, y, rng)) for _ in range(B)])
        rows.append(dict(era=rotulo, objetivo=NOME[f], N=len(s), I_marg_mbit=(o0 - n0.mean()) * 1e3,
                         I_dado_ouro_mbit=o1 * 1e3, I_dado_ouro_corr_mbit=(o1 - n1.mean()) * 1e3,
                         p_valor=(1 + np.sum(n1 >= o1)) / (B + 1),
                         fracao_que_resta=(o1 - n1.mean()) / (o0 - n0.mean())))
    return pd.DataFrame(rows)


def cadeia(s, blocos, seed):
    """blocos: lista de (nome, [colunas]). Ganho incremental I(bloco; V | anteriores) por queda da log-loss (bits)."""
    y = s.V.to_numpy()
    l0 = _loss0(y, seed)
    prev, acum, rows = l0, [], []
    for nome, cols in blocos:
        acum = acum + cols
        l_solo = _cv_loss(s[cols].to_numpy(float), y, seed)
        l_now = _cv_loss(s[acum].to_numpy(float), y, seed)
        d = prev - l_now
        rows.append(dict(bloco=nome, I_isolada=l0.mean() - l_solo.mean(), ganho_incremental=d.mean(),
                         ic95=1.96 * d.std() / np.sqrt(len(d)), I_acumulada=l0.mean() - l_now.mean()))
        prev = l_now
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def run(o, out, B=100, seed=0, bins=10):
    os.makedirs(out, exist_ok=True)
    # amostra comum: chegou ao min 15 e tem todos os 'first*'
    cols = BASE15 + INICIAIS + TARDIAS
    s = o.dropna(subset=cols).reset_index(drop=True)
    for c in ("dd", "hd", "bd", "td", "id"):
        s[c] = s[c].fillna(0)
    print(f"  amostra comum (min 15 + todos os first*): {len(s):,} partidas ({s.year.min()}-{s.year.max()})")

    # 1) marginais (amostra máxima por objetivo)
    m = marginais(o, B, seed)
    m.to_csv(os.path.join(out, "parte5_marginais.csv"), index=False)
    print(m.round(4).to_string())

    # 2) dado o ouro
    d_all = dado_ouro(s, bins, B, seed, "2014-2026")
    d_old = dado_ouro(s[s.year <= 2023].reset_index(drop=True), bins, B, seed, "2014-2023")
    d_new = dado_ouro(s[s.year >= 2024].reset_index(drop=True), bins, B, seed, "2024+")
    dg = pd.concat([d_all, d_old, d_new], ignore_index=True)
    dg.to_csv(os.path.join(out, "parte5_dado_ouro.csv"), index=False)
    print(dg.round(3).to_string())

    # 3) cadeias
    ordem_ouro = [("dif. de ouro", ["gd15"]), ("+ XP, CS, abates", ["xd15", "cd15", "kd15"]),
                  ("+ first blood", ["fb"]), ("+ 1º dragão", ["fd"]), ("+ 1º arauto", ["fh"]), ("+ 1ª torre", ["ft"]),
                  ("[tardia] + 1ª torre do meio", ["fmt"]), ("[tardia] + 1ª a 3 torres", ["f3t"]),
                  ("[tardia] + 1º barão", ["fbar"])]
    c1 = cadeia(s, ordem_ouro, seed)
    c1.insert(0, "ordem", "ouro primeiro")
    ordem_obj = [("first blood", ["fb"]), ("+ 1º dragão", ["fd"]), ("+ 1º arauto", ["fh"]), ("+ 1ª torre", ["ft"]),
                 ("+ dif. de ouro", ["gd15"]), ("+ XP, CS, abates", ["xd15", "cd15", "kd15"])]
    c2 = cadeia(s, ordem_obj, seed)
    c2.insert(0, "ordem", "objetivos iniciais primeiro")
    ch = pd.concat([c1, c2], ignore_index=True)
    ch.to_csv(os.path.join(out, "parte5_cadeia.csv"), index=False)
    print(ch.round(4).to_string())

    # 3b) placas (2022-2025) e vermes (2024+): variáveis que fecham até ~min 14-15
    # Em 2026 a coluna turretplates muda de escala (média ~25 por time, máximo 45, contra no máximo 15 até 2025:
    # 3 torres externas x 5 placas). Não é mais "placas até o min 14", então 2026 fica fora da análise das placas.
    rec = []
    for nome, col, ano, ate in (("placas de torre", "pl", 2022, 2025), ("vermes do vazio", "gr", 2024, 2100)):
        r = o.dropna(subset=BASE15 + [col]).query("year >= @ano and year <= @ate").reset_index(drop=True)
        cc = cadeia(r, [("dif. de ouro", ["gd15"]), ("+ XP, CS, abates", ["xd15", "cd15", "kd15"]),
                        (f"+ {nome}", [col])], seed)
        cc.insert(0, "amostra", (f"{ano}–{ate}" if ate < 2100 else f"{ano}+") + f" (N={len(r):,})")
        rec.append(cc)
    rc = pd.concat(rec, ignore_index=True)
    rc.to_csv(os.path.join(out, "parte5_cadeia_placas_vermes.csv"), index=False)
    print(rc.round(4).to_string())

    # 4) consequência: totais do fim do jogo
    estado = ("estado aos 15 min (ouro, XP, CS, abates + objetivos iniciais)", BASE15 + INICIAIS)
    pos = [(NOME[k], [k]) for k in ("dd", "hd", "bd", "td", "id")]
    y = s.V.to_numpy()
    l0 = _loss0(y, seed)
    prev = _cv_loss(s[estado[1]].to_numpy(float), y, seed)
    acum, rows = list(estado[1]), [dict(bloco=estado[0], I_isolada=np.nan, ganho_incremental=l0.mean() - prev.mean(),
                                        ic95=np.nan, I_acumulada=l0.mean() - prev.mean())]
    for nome, c in pos:
        acum = acum + c
        l_solo = _cv_loss(s[c].to_numpy(float), y, seed)
        l_now = _cv_loss(s[acum].to_numpy(float), y, seed)
        d = prev - l_now
        rows.append(dict(bloco="+ [pós-fato] " + nome, I_isolada=l0.mean() - l_solo.mean(), ganho_incremental=d.mean(),
                         ic95=1.96 * d.std() / np.sqrt(len(d)), I_acumulada=l0.mean() - l_now.mean()))
        prev = l_now
    cq = pd.DataFrame(rows)
    cq.to_csv(os.path.join(out, "parte5_posfato.csv"), index=False)
    print(cq.round(4).to_string())

    _figura(m, d_all, c1, c2, cq, s, out)
    return m, dg, ch, rc, cq


def _figura(m, d_all, c1, c2, cq, s, out):
    hv = it.entropy(s.V.to_numpy())
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    # (a) marginal vs dado o ouro
    a = d_all[~d_all.objetivo.str.startswith("[ref]")].copy()
    x = np.arange(len(a))
    cor = ["C0" if k in [NOME[i] for i in INICIAIS] else "C1" for k in a.objetivo]
    ax[0].bar(x - .2, a.I_marg_mbit, .4, color=cor, alpha=.45, label="isolada I(obj; V)")
    ax[0].bar(x + .2, a.I_dado_ouro_corr_mbit, .4, color=cor, label="dado o ouro I(obj; V | ouro)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(a.objetivo, rotation=35, ha="right", fontsize=8)
    ref = d_all[d_all.objetivo.str.startswith("[ref]")].I_marg_mbit.iloc[0]
    ax[0].axhline(ref, color="k", ls=":", lw=1)
    ax[0].text(len(a) - .5, ref, f"ouro sozinho = {ref:.0f} mbit", ha="right", va="bottom", fontsize=8)
    ax[0].set_ylabel("milibits (corrigido por permutação)")
    ax[0].set_title("Objetivo isolado vs. depois de conhecer o ouro\n(azul = iniciais; laranja = tardias/ambíguas)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    # (b) cascata ouro primeiro
    c = c1
    cores = ["C0" if not b.startswith("[tardia]") else "C3" for b in c.bloco]
    bot = np.r_[0, c.ganho_incremental.cumsum().to_numpy()[:-1]]
    ax[1].bar(range(len(c)), c.ganho_incremental, bottom=bot, color=cores, yerr=c.ic95, capsize=2)
    ax[1].axhline(hv, color="k", ls="--", lw=1)
    ax[1].text(len(c) - .5, hv, f"H(V) = {hv:.3f}", ha="right", va="bottom", fontsize=8)
    ax[1].set_xticks(range(len(c)))
    ax[1].set_xticklabels(c.bloco, rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("bits")
    ax[1].set_title("Regra da cadeia no min 15 (ouro primeiro)\nvermelho = pode ocorrer depois do min 15", fontsize=10)
    # (c) pós-fato
    q = cq
    ax[2].bar(range(len(q)), q.ganho_incremental, bottom=np.r_[0, q.ganho_incremental.cumsum().to_numpy()[:-1]],
              color=["C0"] + ["C3"] * (len(q) - 1))
    ax[2].axhline(hv, color="k", ls="--", lw=1)
    ax[2].set_xticks(range(len(q)))
    ax[2].set_xticklabels(["estado\nmin 15"] + [b.replace("+ [pós-fato] ", "") for b in q.bloco[1:]], rotation=40, ha="right", fontsize=8)
    ax[2].set_ylabel("bits")
    ax[2].set_title("Totais do FIM do jogo: o objetivo vira consequência\n(quase decide o resultado, mas é vazamento)", fontsize=10)
    for a_ in ax:
        a_.grid(alpha=.25, axis="y")
    plt.tight_layout()
    _paineis(fig, ax, out, 'parte5')
    plt.savefig(os.path.join(out, "parte5_objetivos.png"), dpi=130)
    plt.close()


def _paineis(fig, ax, out, prefixo):
    """Salva cada painel como PNG separado (para relatórios em página retrato)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for i, a in enumerate(ax):
        bb = a.get_tightbbox(r).expanded(1.03, 1.03).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(out, f"{prefixo}_painel{i + 1}.png"), dpi=170, bbox_inches=bb)
