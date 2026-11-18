"""Parte 4 - O draft COMPLETO: quanto os 10 campeões, juntos, informam sobre a vitória?

Na Parte 2 medimos um campeão por vez. Aqui a variável é o draft inteiro (5 picks de cada lado, por posição),
com ~10^9 combinações possíveis por lado e ~10^5 partidas: contar células (plug-in) é inviável.
Usamos o limite inferior variacional (o mesmo da Parte 3):

    I(X; V) >= H(V) - log-loss(CV) de um classificador calibrado,

e uma "escada" de modelos aninhados. Pela regra da cadeia, o ganho de log-loss entre degraus consecutivos
estima I(novo ; V | anteriores):

    0. lado + 1ª escolha                          (referência)
    1. + campeões, efeito aditivo                 (força individual)
    2. + campeão x posição                        (a força pode estar numa só posição)
    3. + confrontos de lane (campeão A x campeão B na mesma posição)
    4. + sinergias (pares de campeões do mesmo time)
    5. + bans da fase 1 (ban1-3, ANTES de qualquer pick)
    6. + bans da fase 2 (ban4-5, DEPOIS dos primeiros picks: reativos)
    7. + interação ban x pick (ban reativo x campeões já escolhidos, do adversário e do próprio time)

Todas as variáveis entram com sinal (+1 azul, -1 vermelho), o que torna o modelo simétrico entre os lados.
Depois: (a) draft x força do time; (b) draft x ouro no min 15; (c) robustez (split temporal, placebo com V
embaralhado, gradient boosting categórico); (d) teste par a par de sinergias, confrontos e interações ban x pick contra o modelo aditivo.

Cuidado com o viés: os hiperparâmetros (regularização) são escolhidos num conjunto de AJUSTE (25% das partidas)
que NÃO entra na avaliação (75%, validação cruzada de 5 folds). Assim o limite inferior não é otimista.
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data import POS

NPOS = len(POS)
PAIRS_POS = list(itertools.combinations(range(NPOS), 2))
NOMES_MODELO = {
    "M0": "lado + 1ª escolha",
    "M1": "+ campeões (aditivo)",
    "M2": "+ campeão × posição",
    "M3": "+ confrontos de lane",
    "M4": "+ sinergias (pares do time)",
    "M5": "+ bans fase 1 (pré-pick)",
    "M6": "+ bans fase 2 (reativos)",
    "M7": "+ interação ban × pick",
}
# bloco -> papel (main = penalização padrão; int = interação, ganha um fator de escala ajustado)
PAPEL = {"fp": "main", "champ": "main", "champpos": "main", "lane": "int", "syn": "int", "bans1": "int",
         "bans2": "int", "bp_adv": "int", "bp_own": "int", "team": "main"}
ESCADA = {
    "M0": ["fp"],
    "M1": ["fp", "champ"],
    "M2": ["fp", "champ", "champpos"],
    "M3": ["fp", "champ", "champpos", "lane"],
    "M4": ["fp", "champ", "champpos", "lane", "syn"],
    "M5": ["fp", "champ", "champpos", "lane", "syn", "bans1"],
    "M6": ["fp", "champ", "champpos", "lane", "syn", "bans1", "bans2"],
    "M7": ["fp", "champ", "champpos", "lane", "syn", "bans1", "bans2", "bp_adv", "bp_own"],
}


# --------------------------------------------------------------------------- #
# Construção das variáveis (esparsas, com sinal +1 azul / -1 vermelho)
# --------------------------------------------------------------------------- #
def _csr(rows, cols, vals, shape):
    return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()  # duplicatas são somadas


def _keep_keys(keys, min_count):
    """Chaves (pares) com contagem >= min_count. Usa só a frequência, nunca V (sem vazamento)."""
    u, c = np.unique(keys, return_counts=True)
    return u[c >= min_count]


def indexar(d):
    cols = [f"{s}_{p}" for s in "br" for p in POS]
    champs = sorted(pd.unique(d[cols].to_numpy().ravel()))
    cmap = {c: i for i, c in enumerate(champs)}
    B = np.column_stack([d[f"b_{p}"].map(cmap).to_numpy() for p in POS]).astype(np.int64)
    R = np.column_stack([d[f"r_{p}"].map(cmap).to_numpy() for p in POS]).astype(np.int64)
    return champs, B, R


def construir_blocos(d, champs, B, R, min_lane=30, min_syn=30, min_team=10):
    n, nc = len(d), len(champs)
    ar = np.arange(n)
    blocos, meta = {}, {}

    blocos["fp"] = sp.csr_matrix(d.fp.to_numpy(float).reshape(-1, 1))

    rows = np.repeat(ar, NPOS)
    blocos["champ"] = _csr(np.r_[rows, rows], np.r_[B.ravel(), R.ravel()],
                           np.r_[np.ones(n * NPOS), -np.ones(n * NPOS)], (n, nc))
    pidx = np.tile(np.arange(NPOS), n)
    blocos["champpos"] = _csr(np.r_[rows, rows], np.r_[pidx * nc + B.ravel(), pidx * nc + R.ravel()],
                              np.r_[np.ones(n * NPOS), -np.ones(n * NPOS)], (n, NPOS * nc))

    # confrontos de lane: (posição p, campeão lo, campeão hi) com lo < hi; +1 se o azul tem lo, -1 se tem hi
    K, S, Rw = [], [], []
    for p in range(NPOS):
        a, b = B[:, p], R[:, p]
        ok = a != b
        K.append(((p * nc + np.minimum(a, b)) * nc + np.maximum(a, b))[ok])
        S.append(np.where(a < b, 1.0, -1.0)[ok])
        Rw.append(ar[ok])
    K, S, Rw = np.concatenate(K), np.concatenate(S), np.concatenate(Rw)
    keep = _keep_keys(K, min_lane)
    pos = np.searchsorted(keep, K)
    m = (pos < len(keep)) & (keep[np.minimum(pos, len(keep) - 1)] == K)
    blocos["lane"] = _csr(Rw[m], pos[m], S[m], (n, len(keep)))
    meta["lane_keys"] = keep

    # sinergias: pares de campeões do mesmo time (qualquer posição); +1 se azul, -1 se vermelho
    K, S, Rw = [], [], []
    for X, sg in ((B, 1.0), (R, -1.0)):
        for i, j in PAIRS_POS:
            lo, hi = np.minimum(X[:, i], X[:, j]), np.maximum(X[:, i], X[:, j])
            K.append(lo * nc + hi)
            S.append(np.full(n, sg))
            Rw.append(ar)
    K, S, Rw = np.concatenate(K), np.concatenate(S), np.concatenate(Rw)
    keep = _keep_keys(K, min_syn)
    pos = np.searchsorted(keep, K)
    m = (pos < len(keep)) & (keep[np.minimum(pos, len(keep) - 1)] == K)
    blocos["syn"] = _csr(Rw[m], pos[m], S[m], (n, len(keep)))
    meta["syn_keys"] = keep

    # bans por fase (+1 se o azul baniu, -1 se o vermelho baniu). Fase 1 = ban1-3 (antes dos picks);
    # fase 2 = ban4-5 (depois dos primeiros picks, portanto reativos).
    bl = [s.split("|") for s in d.b_bans]
    rl = [s.split("|") for s in d.r_bans]
    nomes = sorted({x for l in bl + rl for x in l if x})
    bmap = {c: i for i, c in enumerate(nomes)}
    cmap = {c: i for i, c in enumerate(champs)}
    for fase, sl in ((1, slice(0, 3)), (2, slice(3, 5))):
        rr, cc, vv = [], [], []
        for i, (lb, lr) in enumerate(zip(bl, rl)):
            for x in lb[sl]:
                if x:
                    rr.append(i); cc.append(bmap[x]); vv.append(1.0)
            for x in lr[sl]:
                if x:
                    rr.append(i); cc.append(bmap[x]); vv.append(-1.0)
        blocos[f"bans{fase}"] = _csr(rr, cc, vv, (n, len(nomes)))

    # interação ban (fase 2) x pick: "adv" = pick do adversário de quem baniu; "own" = pick do próprio time
    ban2_b = np.array([[cmap.get(x, -1) if x else -1 for x in l[3:5]] + [-1] * (2 - len(l[3:5])) for l in bl])
    ban2_r = np.array([[cmap.get(x, -1) if x else -1 for x in l[3:5]] + [-1] * (2 - len(l[3:5])) for l in rl])
    meta["ban2_b"], meta["ban2_r"] = ban2_b, ban2_r
    for nome, (pb, pr) in (("bp_adv", (R, B)), ("bp_own", (B, R))):
        K, S, Rw = [], [], []
        for ban, picks, sg in ((ban2_b, pb, 1.0), (ban2_r, pr, -1.0)):   # azul baniu / vermelho baniu
            for j in range(ban.shape[1]):
                for k in range(NPOS):
                    ok = ban[:, j] >= 0
                    K.append((ban[ok, j] * nc + picks[ok, k]))
                    S.append(np.full(ok.sum(), sg))
                    Rw.append(ar[ok])
        K, S, Rw = np.concatenate(K), np.concatenate(S), np.concatenate(Rw)
        keep = _keep_keys(K, 20)
        pos = np.searchsorted(keep, K)
        m = (pos < len(keep)) & (keep[np.minimum(pos, len(keep) - 1)] == K)
        blocos[nome] = _csr(Rw[m], pos[m], S[m], (n, len(keep)))

    # força do time (time-ano): +1 azul, -1 vermelho; só time-anos com >= min_team partidas
    tb, tr = d.b_team.to_numpy(), d.r_team.to_numpy()
    cont = pd.Series(np.r_[tb, tr]).value_counts()
    tys = sorted(cont[cont >= min_team].index)
    tmap = {t: i for i, t in enumerate(tys)}
    ib = np.array([tmap.get(t, -1) for t in tb])
    ir = np.array([tmap.get(t, -1) for t in tr])
    mb, mr = ib >= 0, ir >= 0
    blocos["team"] = _csr(np.r_[ar[mb], ar[mr]], np.r_[ib[mb], ir[mr]],
                          np.r_[np.ones(mb.sum()), -np.ones(mr.sum())], (n, len(tys)))
    return blocos, meta


def montar(blocos, nomes, escalas=None, linhas=None, extra=None):
    """Concatena os blocos escolhidos. 'escalas' multiplica cada bloco: escala s equivale a uma
    variância a priori C*s^2 para os coeficientes daquele bloco (regularização por bloco)."""
    escalas = escalas or {}
    mats = []
    for b in nomes:
        X = blocos[b]
        if linhas is not None:
            X = X[linhas]
        mats.append(X * escalas.get(b, 1.0))
    if extra is not None:
        mats.append(sp.csr_matrix(extra))
    return sp.hstack(mats, format="csr")


# --------------------------------------------------------------------------- #
# Ajuste, validação cruzada e perda
# --------------------------------------------------------------------------- #
def _lr(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=400, tol=1e-4)


def _bits(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def cv_oof(X, y, C, folds):
    """Probabilidades fora da amostra (e do 'prior' = taxa do treino) para cada partida."""
    oof, pri = np.empty(len(y)), np.empty(len(y))
    for tr, te in folds:
        mdl = _lr(C).fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        pri[te] = y[tr].mean()
    return oof, pri


def make_folds(y, k, seed):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def ajustar_hiper(blocos, nomes, y, idx_T, seed, init=None,
                  Cs=(0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), mults=(0.1, 0.3, 1.0, 3.0, 10.0)):
    """Escolhe C e a escala de CADA bloco por CV de 3 folds no conjunto de AJUSTE (busca coordenada a coordenada).

    Blocos diferentes pedem regularização diferente (a força do time é um sinal forte; pares de campeões são
    esparsos e ruidosos), por isso uma única penalização global não serve."""
    folds = make_folds(y[idx_T], 3, seed + 11)
    esc = {b: (init or {}).get(b, 1.0 if PAPEL[b] == "main" else 0.3) for b in nomes}
    memo = {}

    def perda(C, e):
        k = (C, tuple(sorted(e.items())))
        if k not in memo:
            oof, _ = cv_oof(montar(blocos, nomes, e, linhas=idx_T), y[idx_T], C, folds)
            memo[k] = _bits(y[idx_T], oof).mean()
        return memo[k]

    def melhor_C(e):
        r = {C: perda(C, e) for C in Cs}
        return min(r, key=r.get)

    C = melhor_C(esc)
    for _ in range(2):                                   # 2 varreduras sobre os blocos
        for b in nomes:
            if b == "fp":
                continue
            r = {m: perda(C, {**esc, b: esc[b] * m}) for m in mults}
            esc[b] = esc[b] * min(r, key=r.get)
        C = melhor_C(esc)
    return dict(C=C, escalas=esc, perda_ajuste=perda(C, esc))


# --------------------------------------------------------------------------- #
# Escada
# --------------------------------------------------------------------------- #
def _linha(nome, rotulo, loss, loss_prev, loss_prior, oof, y, hp, nfeat):
    d_inc = loss_prev - loss
    d_tot = loss_prior - loss
    return dict(modelo=nome, descricao=rotulo, logloss=loss.mean(),
                I_mbit=1e3 * d_tot.mean(), I_ep_mbit=1e3 * d_tot.std() / np.sqrt(len(d_tot)),
                ganho_mbit=1e3 * d_inc.mean(), ganho_ep_mbit=1e3 * d_inc.std() / np.sqrt(len(d_inc)),
                acerto=float(np.mean((oof > .5) == (y == 1))), C=hp["C"],
                escalas="; ".join(f"{k}={v:.3g}" for k, v in hp["escalas"].items() if k != "fp"), n_colunas=nfeat)


def rodar_escada(blocos, y, idx_E, hiper, folds, rotulos=NOMES_MODELO, verbose=True):
    yE = y[idx_E]
    oofs, losses, linhas = {}, {}, []
    prior_loss = None
    prev_loss = None
    for nome, bl in ESCADA.items():
        hp = hiper[nome]
        X = montar(blocos, bl, hp["escalas"], linhas=idx_E)
        oof, pri = cv_oof(X, yE, hp["C"], folds)
        loss = _bits(yE, oof)
        if prior_loss is None:
            prior_loss = _bits(yE, pri)
            prev_loss = prior_loss
        linhas.append(_linha(nome, rotulos[nome], loss, prev_loss, prior_loss, oof, yE, hp, X.shape[1]))
        oofs[nome], losses[nome] = oof, loss
        prev_loss = loss
        if verbose:
            r = linhas[-1]
            print(f"    {nome} {rotulos[nome]:<32} I = {r['I_mbit']:7.2f} mbit  (ganho {r['ganho_mbit']:+6.2f} "
                  f"± {1.96 * r['ganho_ep_mbit']:.2f})  C={hp['C']}")
    return pd.DataFrame(linhas), oofs, losses, prior_loss


# --------------------------------------------------------------------------- #
# Draft x time x ouro
# --------------------------------------------------------------------------- #
def cadeia_time_ouro(blocos, d, y, idx_E, hiper, folds, seed):
    """Informações condicionais entre draft (M4), força do time (time-ano), bans e ouro no min 15."""
    yE = y[idx_E]
    out = []

    def L(chave, nomes, extra=None, lin=idx_E, yy=yE, fo=folds):
        hp = hiper[chave]
        X = montar(blocos, nomes, hp["escalas"], linhas=lin, extra=extra)
        oof, pri = cv_oof(X, yy, hp["C"], fo)
        return _bits(yy, oof), _bits(yy, pri)

    def reg(nome, a, b):
        x = a - b
        out.append(dict(quantidade=nome, I_mbit=1e3 * x.mean(), ep_mbit=1e3 * x.std() / np.sqrt(len(x))))

    l_m0, _ = L("M0", ["fp"])
    l_d, _ = L("M4", ESCADA["M4"])
    l_t, _ = L("TEAM", ["fp", "team"])
    l_dt, _ = L("M4+TEAM", ESCADA["M4"] + ["team"])
    l_d5t, _ = L("M5+TEAM", ESCADA["M5"] + ["team"])
    l_d6t, _ = L("M6+TEAM", ESCADA["M6"] + ["team"])
    l_d7t, _ = L("M7+TEAM", ESCADA["M7"] + ["team"])
    reg("I(draft ; V | lado)", l_m0, l_d)
    reg("I(time ; V | lado)", l_m0, l_t)
    reg("I(draft ; V | time)", l_t, l_dt)
    reg("I(time ; V | draft)", l_d, l_dt)
    reg("I(bans fase 1 ; V | draft, time)", l_dt, l_d5t)
    reg("I(bans fase 2 ; V | draft, time, fase 1)", l_d5t, l_d6t)
    reg("I(ban × pick ; V | tudo acima, time)", l_d6t, l_d7t)

    # ouro no min 15: só partidas que chegaram ao min 15 (mesma amostra da Parte 3)
    gd = d.gd15.to_numpy()[idx_E]
    ok = np.where(~np.isnan(gd))[0]
    yS = yE[ok]
    foS = make_folds(yS, 5, seed + 5)
    gz = (gd[ok] - gd[ok].mean()) / gd[ok].std()
    linS = idx_E[ok]
    g_col = (gz * 5.0).reshape(-1, 1)           # escala grande = quase sem penalização
    kw = dict(lin=linS, yy=yS, fo=foS)
    lS_m0, _ = L("M0", ["fp"], **kw)
    lS_d, _ = L("M4", ESCADA["M4"], **kw)
    lS_g, _ = L("M0", ["fp"], extra=g_col, **kw)
    lS_dg, _ = L("M4", ESCADA["M4"], extra=g_col, **kw)
    reg("[min 15] I(draft ; V | lado)", lS_m0, lS_d)
    reg("[min 15] I(ouro ; V | lado)", lS_m0, lS_g)
    reg("[min 15] I(draft ; V | ouro)", lS_g, lS_dg)
    reg("[min 15] I(ouro ; V | draft)", lS_d, lS_dg)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Robustez
# --------------------------------------------------------------------------- #
def teste_temporal(blocos, d, y, hiper, primeiro_ano_teste=2019):
    """Origem móvel: para cada ano Y, treina com todos os anos anteriores e testa em Y.
    O draft informa sobre partidas de uma era que o modelo não viu?"""
    anos = d.year.to_numpy()
    rows, pooled = [], {}
    usados = [Y for Y in range(primeiro_ano_teste, int(anos.max()) + 1) if (anos == Y).sum() >= 500]
    for nome in ("M0", "M1", "M2", "M4"):
        hp = hiper[nome]
        X = montar(blocos, ESCADA[nome], hp["escalas"])
        todos = []
        for Y in usados:                                  # anos com >= 500 partidas
            tr, te = np.where(anos < Y)[0], np.where(anos == Y)[0]
            mdl = _lr(hp["C"]).fit(X[tr], y[tr])
            x = _bits(y[te], np.full(len(te), y[tr].mean())) - _bits(y[te], mdl.predict_proba(X[te])[:, 1])
            todos.append(x)
            rows.append(dict(teste=f"origem móvel, ano {Y}", modelo=nome, I_mbit=1e3 * x.mean(),
                             ep_mbit=1e3 * x.std() / np.sqrt(len(x)), n_teste=len(te)))
        x = np.concatenate(todos)
        pooled[nome] = dict(teste=f"origem móvel, {usados[0]}–{usados[-1]} (agregado)", modelo=nome,
                            I_mbit=1e3 * x.mean(), ep_mbit=1e3 * x.std() / np.sqrt(len(x)), n_teste=len(x))
    return rows, list(pooled.values())


def placebo(blocos, y, idx_E, hiper, folds, n_perm, seed):
    """Embaralha V (entre partidas) e repete a escada: um estimador sem viés positivo dá I <= 0."""
    rng = np.random.default_rng(seed + 77)
    rows = []
    for nome in ("M2", "M4"):
        hp = hiper[nome]
        X = montar(blocos, ESCADA[nome], hp["escalas"], linhas=idx_E)
        vals = []
        for _ in range(n_perm):
            yp = rng.permutation(y[idx_E])
            oof, pri = cv_oof(X, yp, hp["C"], folds)
            vals.append(1e3 * (_bits(yp, pri) - _bits(yp, oof)).mean())
        rows.append(dict(teste=f"placebo (V embaralhado, {n_perm} rep.)", modelo=nome,
                         I_mbit=float(np.mean(vals)), ep_mbit=float(np.std(vals, ddof=1)) if n_perm > 1 else np.nan,
                         n_teste=len(idx_E)))
    return rows


def gradient_boosting(B, R, y, idx_E, idx_T, folds, seed):
    """Modelo não linear flexível sobre os 10 campeões (categóricos), com aumento por troca de lados."""
    def feats(i, trocar):
        b, r = (R[i], B[i]) if trocar else (B[i], R[i])
        return np.column_stack([b, r, np.full(len(i), int(trocar))]).astype(np.int64)

    cat = np.array([True] * (2 * NPOS) + [False])

    def mk(n_iter):
        return HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.05, max_leaf_nodes=16,
                                              min_samples_leaf=100, l2_regularization=5.0,
                                              categorical_features=cat, early_stopping=False,
                                              random_state=seed)

    def ajusta(tr, n_iter):
        Xa = np.vstack([feats(tr, False), feats(tr, True)])
        ya = np.r_[y[tr], 1 - y[tr]]
        return mk(n_iter).fit(Xa, ya)

    # nº de iterações escolhido no conjunto de ajuste (treino 70% / validação 30%)
    rng = np.random.default_rng(seed + 3)
    perm = rng.permutation(idx_T)
    cut = int(0.7 * len(perm))
    trT, vaT = perm[:cut], perm[cut:]
    mdl = ajusta(trT, 400)
    ls = [_bits(y[vaT], p[:, 1]).mean() for p in mdl.staged_predict_proba(feats(vaT, False))]
    n_best = int(np.argmin(ls)) + 1

    yE = y[idx_E]
    oof, pri = np.empty(len(yE)), np.empty(len(yE))
    for tr, te in folds:
        m = ajusta(idx_E[tr], n_best)
        oof[te] = m.predict_proba(feats(idx_E[te], False))[:, 1]
        pri[te] = yE[tr].mean()
    x = _bits(yE, pri) - _bits(yE, oof)
    return dict(teste="gradient boosting categórico (10 campeões)", modelo=f"GBM ({n_best} árvores)",
                I_mbit=1e3 * x.mean(), ep_mbit=1e3 * x.std() / np.sqrt(len(x)), n_teste=len(yE))


# --------------------------------------------------------------------------- #
# Pares: sinergias (mesmo time) e confrontos (mesma posição)
# --------------------------------------------------------------------------- #
def _agrupa(keys, win, p):
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    return u, n, np.bincount(inv, win), np.bincount(inv, p), np.bincount(inv, p * (1 - p))


def testar_pares(champs, B, R, y, idx_E, p_oof, n_min=100):
    """Compara a taxa de vitória de cada par com o que o modelo ADITIVO (M2, fora da amostra) esperaria."""
    nc = len(champs)
    b, r, yy = B[idx_E], R[idx_E], y[idx_E]
    # sinergias: um registro por (partida, time, par)
    K, W, P = [], [], []
    for X, win, pw in ((b, yy, p_oof), (r, 1 - yy, 1 - p_oof)):
        for i, j in PAIRS_POS:
            K.append(np.minimum(X[:, i], X[:, j]) * nc + np.maximum(X[:, i], X[:, j]))
            W.append(win)
            P.append(pw)
    u, n, w, e, v = _agrupa(np.concatenate(K), np.concatenate(W).astype(float), np.concatenate(P))
    syn = pd.DataFrame(dict(tipo="sinergia", A=[champs[k // nc] for k in u], B=[champs[k % nc] for k in u],
                            posicao="", n=n, wins=w, esperado=e, var=v))
    # confrontos de lane: perspectiva do campeão de menor índice
    parts = []
    for p in range(NPOS):
        a, c = b[:, p], r[:, p]
        ok = a != c
        lo, hi = np.minimum(a, c)[ok], np.maximum(a, c)[ok]
        azul_tem_lo = (a < c)[ok]
        win = np.where(azul_tem_lo, yy[ok], 1 - yy[ok]).astype(float)
        pw = np.where(azul_tem_lo, p_oof[ok], 1 - p_oof[ok])
        u2, n2, w2, e2, v2 = _agrupa(lo * nc + hi, win, pw)
        parts.append(pd.DataFrame(dict(tipo="confronto", A=[champs[k // nc] for k in u2],
                                       B=[champs[k % nc] for k in u2], posicao=POS[p],
                                       n=n2, wins=w2, esperado=e2, var=v2)))
    return _finaliza(pd.concat([syn] + parts, ignore_index=True), n_min)


def _finaliza(df, n_min):
    df = df[df.n >= n_min].copy()
    df["win_obs"] = df.wins / df.n
    df["win_esp"] = df.esperado / df.n
    df["dif_pp"] = 100 * (df.win_obs - df.win_esp)
    df["z"] = (df.wins - df.esperado) / np.sqrt(df["var"])
    df["p"] = 2 * norm.sf(np.abs(df.z))
    df["q"] = np.nan
    for t, g in df.groupby("tipo"):                       # Benjamini-Hochberg por tipo
        m = len(g)
        o = g.p.sort_values().index
        q = (g.p.loc[o].to_numpy() * m / np.arange(1, m + 1))[::-1]
        q = np.minimum.accumulate(q)[::-1]
        df.loc[o, "q"] = np.minimum(q, 1.0)
    return df.drop(columns=["var"]).reset_index(drop=True)


def testar_ban_pick(champs, B, R, meta, y, idx_E, p_oof, n_min=100):
    """Ban reativo x pick: a taxa de vitória do time que baniu b (fase 2) quando o pick p está em campo,
    contra o esperado pelo modelo M6 (que já conhece picks e bans, mas não a interação). Perspectiva de quem baniu."""
    nc = len(champs)
    b2b, b2r = meta["ban2_b"][idx_E], meta["ban2_r"][idx_E]
    b, r, yy = B[idx_E], R[idx_E], y[idx_E]
    out = []
    for tipo, (pb, pr) in (("ban × pick do adversário", (r, b)), ("ban × pick do próprio time", (b, r))):
        K, W, P = [], [], []
        for ban, picks, win, pw in ((b2b, pb, yy, p_oof), (b2r, pr, 1 - yy, 1 - p_oof)):
            for j in range(ban.shape[1]):
                ok = ban[:, j] >= 0
                for k in range(NPOS):
                    K.append(ban[ok, j] * nc + picks[ok, k])
                    W.append(win[ok])
                    P.append(pw[ok])
        u, n, w, e, v = _agrupa(np.concatenate(K), np.concatenate(W).astype(float), np.concatenate(P))
        out.append(pd.DataFrame(dict(tipo=tipo, A=[champs[k // nc] for k in u], B=[champs[k % nc] for k in u],
                                     posicao="", n=n, wins=w, esperado=e, var=v)))
    return _finaliza(pd.concat(out, ignore_index=True), n_min)


def resumo_pares(df):
    rows = []
    for t, g in df.groupby("tipo"):
        rows.append(dict(tipo=t, pares_testados=len(g), dp_z=g.z.std(), frac_abs_z_gt2=(g.z.abs() > 2).mean(),
                         esperado_se_so_ruido=2 * norm.sf(2), n_q_lt_005=int((g.q < 0.05).sum()),
                         mediana_n=float(g.n.median()), max_abs_dif_pp=g.dif_pp.abs().max()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figura_escada(esc, tg, rob, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))
    # (a) ganhos incrementais
    a = ax[0]
    r = esc.iloc[1:]
    y = np.arange(len(r))[::-1]
    cores = ["C0", "C0", "C1", "C1", "C2", "C2", "C3"][: len(r)]
    a.barh(y, r.ganho_mbit, xerr=1.96 * r.ganho_ep_mbit, color=cores, capsize=3)
    a.set_yticks(y)
    a.set_yticklabels(r.descricao)
    a.axvline(0, color="k", lw=.8)
    for yi, v in zip(y, r.ganho_mbit):
        a.text(v, yi + 0.33, f" {v:+.2f}", fontsize=8, va="center")
    a.set_xlabel("ganho incremental I(novo ; V | anteriores) [mbit]")
    a.set_title(f"Escada do draft completo (acumulado: {esc.I_mbit.iloc[-1]:.1f} mbit)")
    # (b) draft x time x ouro
    b = ax[1]
    t = tg.copy()
    y2 = np.arange(len(t))[::-1]
    b.barh(y2, t.I_mbit.clip(lower=1e-3), xerr=1.96 * t.ep_mbit, color=["C0" if "draft" in q.split(";")[0] else "C3"
           for q in t.quantidade])
    b.set_xscale("log")
    b.set_yticks(y2)
    b.set_yticklabels(t.quantidade, fontsize=8)
    b.set_xlabel("mbit (escala log)")
    b.set_title("Draft x força do time x ouro (min 15)")
    # (c) robustez
    c = ax[2]
    rb = rob.copy()
    y3 = np.arange(len(rb))[::-1]
    c.barh(y3, rb.I_mbit, xerr=1.96 * rb.ep_mbit.fillna(0), color="C4")
    c.axvline(0, color="k", lw=.8)
    c.set_yticks(y3)
    c.set_yticklabels([f"{a_[:34]}\n[{m_}]" for a_, m_ in zip(rb.teste, rb.modelo)], fontsize=7)
    c.set_xlabel("I estimada [mbit]")
    c.set_title("Robustez: era, placebo e modelo não linear")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte4_escada.png"), dpi=140)
    plt.close(fig)


def figura_pares(df, out):
    tipos = [("sinergia", "Sinergias (mesmo time)"), ("confronto", "Confrontos de lane"),
             ("ban × pick do adversário", "Ban reativo × pick do adversário"),
             ("ban × pick do próprio time", "Ban reativo × pick do próprio time")]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
    xs = np.linspace(-5, 5, 400)
    for a, (t, nome) in zip(ax.ravel(), tipos):
        g = df[df.tipo == t]
        a.hist(g.z.clip(-8, 8), bins=60, range=(-8, 8), density=True, color="C0", alpha=.75,
               label=f"observado (dp = {g.z.std():.2f})")
        a.plot(xs, norm.pdf(xs), "k--", label="N(0,1): só ruído")
        a.set_title(f"{nome}: {len(g):,} pares com n ≥ 100".replace(",", "."), fontsize=10)
        a.set_xlabel("z = (vitórias − esperado pelo modelo sem a interação) / erro-padrão", fontsize=8)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parte4_pares.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run(drafts, out, seed=0, n_perm=3, frac_ajuste=0.25, verbose=True):
    os.makedirs(out, exist_ok=True)
    d = drafts.reset_index(drop=True)
    y = d.V.to_numpy().astype(np.int64)
    champs, B, R = indexar(d)
    print(f"  {len(d):,} partidas, {len(champs)} campeões")
    blocos, meta = construir_blocos(d, champs, B, R)
    print("  colunas por bloco: " + ", ".join(f"{k}={v.shape[1]:,}" for k, v in blocos.items()))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(d))
    nT = int(frac_ajuste * len(d))
    idx_T, idx_E = np.sort(perm[:nT]), np.sort(perm[nT:])
    print(f"  conjunto de ajuste (hiperparâmetros): {len(idx_T):,} | avaliação (CV 5 folds): {len(idx_E):,}")
    yE = y[idx_E]
    folds = make_folds(yE, 5, seed)

    print("  ajustando a regularização no conjunto de ajuste...")
    hiper, prev = {}, None
    for nome, bl in ESCADA.items():                       # cada degrau herda as escalas do anterior (ponto de partida)
        hiper[nome] = ajustar_hiper(blocos, bl, y, idx_T, seed, init=prev)
        prev = hiper[nome]["escalas"]
    hiper["TEAM"] = ajustar_hiper(blocos, ["fp", "team"], y, idx_T, seed)
    ini = {**hiper["M4"]["escalas"], "team": hiper["TEAM"]["escalas"]["team"]}
    hiper["M4+TEAM"] = ajustar_hiper(blocos, ESCADA["M4"] + ["team"], y, idx_T, seed, init=ini)
    prev = hiper["M4+TEAM"]["escalas"]
    for k in ("M5", "M6", "M7"):
        ini = {**hiper[k]["escalas"], "team": prev["team"]}
        hiper[k + "+TEAM"] = ajustar_hiper(blocos, ESCADA[k] + ["team"], y, idx_T, seed, init=ini)
        prev = hiper[k + "+TEAM"]["escalas"]
    for k, v in hiper.items():
        print(f"    {k:<8} C={v['C']}  escalas=" + ", ".join(f"{b}={e:.3g}" for b, e in v["escalas"].items() if b != "fp"))

    print("  escada de modelos (CV no conjunto de avaliação):")
    esc, oofs, losses, prior_loss = rodar_escada(blocos, y, idx_E, hiper, folds, verbose=verbose)
    esc.to_csv(os.path.join(out, "parte4_escada.csv"), index=False)
    base_acc = max(yE.mean(), 1 - yE.mean())
    print(f"  acerto de referência (lado mais frequente): {base_acc:.4f}")

    print("  draft x time x ouro...")
    tg = cadeia_time_ouro(blocos, d, y, idx_E, hiper, folds, seed)
    tg.to_csv(os.path.join(out, "parte4_time_ouro.csv"), index=False)
    print(tg.round(3).to_string(index=False))

    print("  robustez...")
    por_ano, agreg = teste_temporal(blocos, d, y, hiper)
    rob = agreg + placebo(blocos, y, idx_E, hiper, folds, n_perm, seed)
    rob.append(gradient_boosting(B, R, y, idx_E, idx_T, folds, seed))
    rob = pd.DataFrame(rob)
    pd.concat([pd.DataFrame(por_ano), rob], ignore_index=True).to_csv(os.path.join(out, "parte4_robustez.csv"), index=False)
    print(rob.round(3).to_string(index=False))

    print("  pares (sinergia / confronto) contra o modelo aditivo M2...")
    pares = testar_pares(champs, B, R, y, idx_E, oofs["M2"])
    bp = testar_ban_pick(champs, B, R, meta, y, idx_E, oofs["M6"])
    pares = pd.concat([pares, bp], ignore_index=True).sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)
    pares.to_csv(os.path.join(out, "parte4_pares.csv"), index=False)
    rp = resumo_pares(pares)
    rp.to_csv(os.path.join(out, "parte4_pares_resumo.csv"), index=False)
    print(rp.round(3).to_string(index=False))

    figura_escada(esc, tg[tg.quantidade.str.startswith("I(") | tg.quantidade.str.startswith("[min")], rob, out)
    figura_pares(pares, out)
    return esc, tg, rob, pares, rp
