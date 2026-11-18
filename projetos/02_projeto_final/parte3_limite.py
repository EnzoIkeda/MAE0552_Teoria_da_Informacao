"""Parte 3 - Qual o menor erro possível ao prever a vitória com o que se sabe no min 15?

1. Limite de Fano:  Pe >= h^{-1}( H(V | X15) ), com H(V|X15) estimado por
   discretização (plug-in e Miller-Madow), para vários números de faixas.
   Isso é uma ESTIMATIVA do limite, não um piso garantido: discretizar aumenta
   H(V|X) e o plug-in diminui. O que se garante é um teto para o limite:
   como log-loss >= H(V|X), o limite de Fano verdadeiro é <= h^{-1}(log-loss).
2. Comparação com classificadores reais (logística e gradient boosting) em
   validação cruzada e com o erro de Bayes plug-in.
3. Regra da cadeia:  I(X1..Xk ; V) = sum_i I(Xi ; V | X1..X_{i-1}).
   Estimada como queda da log-loss (em bits) em validação cruzada:
   I(X;V) >= H(V) - logloss(X)   (limite inferior variacional), de modo que a
   soma dos incrementos telescopa exatamente.
4. Teste dedicado (exato, discreto) de "o que o 1º dragão acrescenta depois
   de conhecer o ouro":  I(dragão ; V | faixa de ouro) com teste de permutação.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import it_utils as it

FEATS_MIN15 = ["gd15", "xd15", "cd15", "kd15"]
NOMES = {"gd15": "dif. de ouro", "xd15": "dif. de XP", "cd15": "dif. de CS", "kd15": "dif. de abates",
         "fd": "1º dragão", "dd_final": "[VAZAMENTO] dragões no fim do jogo"}


def _logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _hgb(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def _cv_proba(model, X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return np.clip(cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1], 1e-6, 1 - 1e-6)


def _losses(y, p):
    """log-loss por amostra, em bits."""
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


# --------------------------------------------------------------------------- #
def fano_vs_classificadores(g, out, bins_list, B, seed):
    y = g.V.to_numpy()
    hv = it.entropy(y)
    rows = []
    # (a) só ouro, e (b) 4 variáveis do min 15, por nº de faixas
    for nome, feats, ks in (("ouro", ["gd15"], (5, 10, 20, 40)),
                            ("ouro+XP+CS+abates", FEATS_MIN15, bins_list)):
        for kk in ks:
            xb = it.encode(*[it.qbin(g[f], kk) for f in feats])
            h_plug = it.cond_entropy(y, xb)
            h_mm = it.cond_entropy(y, xb, mm=True)
            f_mm = it.fano_min_error(h_mm)
            # IC "básico" (2*estimativa - quantis): o bootstrap com reposição duplica linhas e puxa a
            # entropia plug-in para baixo, deslocando o IC percentil (a estimativa caía fora do próprio IC)
            media_b, q_lo, q_hi = it.bootstrap_ci(lambda i: it.fano_min_error(it.cond_entropy(y[i], xb[i], mm=True)),
                                                  len(y), B=B, seed=seed)
            rows.append(dict(conjunto=nome, faixas_por_var=kk, n_celulas=int(xb.max()) + 1,
                             H_V_dado_X_plugin=h_plug, H_V_dado_X_MM=h_mm,
                             fano_plugin=it.fano_min_error(h_plug), fano_MM=f_mm,
                             fano_MM_lo=2 * f_mm - q_hi, fano_MM_hi=2 * f_mm - q_lo, fano_MM_boot_media=media_b,
                             fano_MM_corrigido_vies=2 * f_mm - media_b,   # centro do IC básico
                             bayes_plugin_insample=it.bayes_error_plugin(xb, y),
                             bayes_plugin_cv=it.bayes_error_cv(xb, y, seed=seed)))
    tab = pd.DataFrame(rows).drop_duplicates(["conjunto", "faixas_por_var"])
    tab.to_csv(os.path.join(out, "parte3_fano_por_faixas.csv"), index=False)

    # classificadores reais
    clf = {}
    X = g[FEATS_MIN15].to_numpy(float)
    for nome, mdl in (("logística", _logit()), ("gradient boosting", _hgb(seed))):
        p = _cv_proba(mdl, X, y, seed)
        clf[nome] = dict(erro=float(np.mean((p > .5) != y)), logloss=float(_losses(y, p).mean()))
    p1 = _cv_proba(_logit(), g[["gd15"]].to_numpy(float), y, seed)
    clf["logística (só ouro)"] = dict(erro=float(np.mean((p1 > .5) != y)), logloss=float(_losses(y, p1).mean()))
    # A log-loss de qualquer modelo é um TETO para H(V | X) (entropia cruzada >= entropia condicional).
    # Como h^{-1} é crescente, h^{-1}(log-loss) é um teto para o limite de Fano verdadeiro: o piso
    # garantido de erro não pode passar disso. Já o Fano por discretização é só uma estimativa
    # (discretizar empurra H(V|X) para cima; o plug-in empurra para baixo).
    for v in clf.values():
        v["teto_do_limite_fano"] = it.fano_min_error(v["logloss"])
    base = min(y.mean(), 1 - y.mean())
    pd.DataFrame([dict(metodo=k, **v) for k, v in clf.items()] +
                 [dict(metodo="palpite ingênuo (lado mais frequente)", erro=base, logloss=hv)]
                 ).to_csv(os.path.join(out, "parte3_classificadores.csv"), index=False)
    return tab, clf, base, hv


# --------------------------------------------------------------------------- #
def cadeia(g, ordem, modelo, seed):
    y = g.V.to_numpy()
    mk = (lambda: _logit()) if modelo == "logit" else (lambda: _hgb(seed))
    # CE_0: preditor constante (priori) em CV
    p0 = _cv_proba(DummyClassifier(strategy="prior"), np.zeros((len(y), 1)), y, seed)
    loss_prev = _losses(y, p0)
    ce0 = loss_prev.mean()
    rows, acum = [], []
    for f in ordem:
        acum.append(f)
        loss_solo = _losses(y, _cv_proba(mk(), g[[f]].to_numpy(float), y, seed))
        loss_now = _losses(y, _cv_proba(mk(), g[acum].to_numpy(float), y, seed))
        d = loss_prev - loss_now
        rows.append(dict(variavel=NOMES[f], I_isolada=ce0 - loss_solo.mean(),
                         ganho_incremental=d.mean(), ep=d.std() / np.sqrt(len(d)),
                         I_acumulada=ce0 - loss_now.mean()))
        loss_prev = loss_now
    return pd.DataFrame(rows), ce0


def teste_dragao(g, bins, B, seed):
    """I(dragão ; V) e I(dragão ; V | faixa de ouro) - exatos (discretos)."""
    y = g.V.to_numpy()
    zg = it.qbin(g.gd15, bins)
    rng = np.random.default_rng(seed)
    rows = []
    for nome in ("fd", "dd_final"):
        x = g[nome].to_numpy()
        for ctx, z in (("marginal", None), (f"| ouro ({bins} faixas)", zg)):
            cmi = it.CondMI(x, z)
            strata = np.zeros(len(y), dtype=np.int64) if z is None else z
            obs, obs_mm = cmi(y), cmi(y, mm=True)
            nul = np.array([cmi(it.permute_within(strata, y, rng)) for _ in range(B)])
            rows.append(dict(variavel=NOMES[nome], condicao=ctx, I_plugin=obs, I_MM=obs_mm,
                             I_nulo_media=nul.mean(), I_corrigida=obs - nul.mean(),
                             p_valor=(1 + np.sum(nul >= obs)) / (B + 1)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def run(games, out, bins=(3, 4, 5, 6, 8), B=200, seed=0, modelo="hgb"):
    os.makedirs(out, exist_ok=True)
    g = games.dropna(subset=FEATS_MIN15).reset_index(drop=True)
    print(f"  {len(g):,} partidas que chegaram ao min 15 com dados completos")

    tab, clf, base, hv = fano_vs_classificadores(g, out, bins, max(B, 50) // 2, seed)

    ordem = FEATS_MIN15 + ["fd", "dd_final"]
    ch, ce0 = cadeia(g, ordem, modelo, seed)
    ch.to_csv(os.path.join(out, "parte3_cadeia.csv"), index=False)
    dr = teste_dragao(g, 10, B, seed)
    dr.to_csv(os.path.join(out, "parte3_dragao.csv"), index=False)

    # ---------------- figura A: erro mínimo vs classificadores ----------------
    t4 = tab[tab.conjunto != "ouro"]
    t1 = tab[tab.conjunto == "ouro"]
    k_ref = t4.sort_values("faixas_por_var").iloc[len(t4) // 2]   # faixa intermediária
    r1 = t1.iloc[len(t1) // 2]
    teto = min(clf["logística"]["teto_do_limite_fano"], clf["gradient boosting"]["teto_do_limite_fano"])
    itens = [
        ("baseline (chutar o lado mais frequente)", base, "gray"),
        (f"Fano estimado | só ouro ({int(r1.faixas_por_var)} faixas)", r1.fano_MM, "C0"),
        (f"Fano estimado | ouro+XP+CS+abates ({int(k_ref.faixas_por_var)} faixas/var)", k_ref.fano_MM, "C0"),
        ("teto do limite de Fano (log-loss, 4 vars)", teto, "C9"),
        ("erro de Bayes plug-in (4 vars, em CV)", k_ref.bayes_plugin_cv, "C2"),
        ("logística, só ouro (CV)", clf["logística (só ouro)"]["erro"], "C1"),
        ("logística, 4 vars (CV)", clf["logística"]["erro"], "C1"),
        ("gradient boosting, 4 vars (CV)", clf["gradient boosting"]["erro"], "C3"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    for i, (n, v, c) in enumerate(itens[::-1]):
        a.barh(i, v, color=c, alpha=.85)
        a.text(v + .003, i, f"{v:.3f}", va="center", fontsize=9)
    a.set_yticks(range(len(itens))); a.set_yticklabels([n for n, _, _ in itens[::-1]], fontsize=9)
    a.set_xlabel("probabilidade de erro"); a.set_xlim(0, base * 1.15)
    a.set_title("Min 15: limite teórico vs. classificadores reais"); a.grid(alpha=.3, axis="x")

    a = ax[1]
    for nome, c in (("ouro", "C0"), ("ouro+XP+CS+abates", "C3")):
        s = tab[tab.conjunto == nome]
        a.plot(s.faixas_por_var, s.fano_plugin, "o--", color=c, alpha=.6, label=f"Fano plug-in | {nome}")
        a.plot(s.faixas_por_var, s.fano_MM, "s-", color=c, label=f"Fano Miller-Madow | {nome}")
    a.axhline(clf["gradient boosting"]["erro"], color="k", ls=":", label="gradient boosting (CV)")
    a.axhline(teto, color="C9", ls="--", label="teto do limite de Fano (log-loss, 4 vars)")
    a.set_xlabel("faixas por variável (discretização)"); a.set_ylabel("erro mínimo (Fano)")
    a.set_title("Sensibilidade do limite de Fano à discretização"); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte3_erro_limite.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figura B: regra da cadeia ----------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    a = ax[0]
    base_ = 0.0
    for i, r in ch.iterrows():
        cor = "C3" if "VAZAMENTO" in r.variavel else "C0"
        a.bar(i, r.ganho_incremental, bottom=base_, color=cor, alpha=.85, yerr=1.96 * r.ep, capsize=3)
        base_ += r.ganho_incremental
    a.axhline(hv, color="k", ls="--", label=f"H(V) = {hv:.3f} bits (teto)")
    a.set_xticks(range(len(ch))); a.set_xticklabels([v.replace(" no fim do jogo", "\n no fim") for v in ch.variavel],
                                                    rotation=25, ha="right", fontsize=8)
    a.set_ylabel("bits sobre V"); a.set_title("Regra da cadeia: ganho incremental (cascata)")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
    a = ax[1]
    idx = np.arange(len(ch)); w = .38
    a.bar(idx - w / 2, ch.I_isolada, w, label="isolada: I(Xi ; V)")
    a.bar(idx + w / 2, ch.ganho_incremental, w, yerr=1.96 * ch.ep, capsize=3, label="incremental: I(Xi ; V | anteriores)")
    a.set_xticks(idx); a.set_xticklabels([v.replace(" no fim do jogo", "\n no fim") for v in ch.variavel],
                                         rotation=25, ha="right", fontsize=8)
    a.set_ylabel("bits"); a.set_title("Informação isolada vs. incremental (redundância)")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte3_cadeia.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n=== PARTE 3 ===")
    print(f"H(V) = {hv:.4f} bits | baseline de erro = {base:.4f}")
    print("\nFano vs discretização:")
    print(tab.round(4).to_string(index=False))
    print("\nClassificadores (5-fold CV):")
    for k, v in clf.items():
        print(f"  {k:24s} erro = {v['erro']:.4f}   logloss = {v['logloss']:.4f} bits   "
              f"h^-1(logloss) = {v['teto_do_limite_fano']:.4f}")
    print(f"Teto do limite de Fano (menor log-loss com 4 vars): {teto:.4f}. O Fano discretizado acima "
          f"dele é estimativa, não piso garantido.")
    print(f"\nRegra da cadeia (modelo = {modelo}, ordem = {', '.join(NOMES[f] for f in ordem)}):")
    print(ch.round(4).to_string(index=False))
    print("\nTeste dedicado do dragão (milibits):")
    d = dr.copy()
    for c in ("I_plugin", "I_MM", "I_nulo_media", "I_corrigida"):
        d[c] = d[c] * 1e3
    print(d.round(3).to_string(index=False))
    return tab, clf, ch, dr
