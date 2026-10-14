"""
Topico 3: Compressao de Dados, AEP e o Criterio de Kelly
=========================================================

Simulacao seguindo o roteiro do Bloco 2.

Como sugerido no PDF, as quantidades centrais (entropia, informacao mutua,
divergencia KL) sao implementadas manualmente, em vez de chamar uma unica
funcao pronta de biblioteca. Bibliotecas (scipy) so aparecem como
conferencia opcional.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Funcoes basicas de teoria da informacao (reaproveitaveis nos outros
# topicos do curso)
# ---------------------------------------------------------------------


def entropy_bits(p):
    """H(p) em bits, para um vetor de probabilidades p (soma = 1)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # lim_{p->0+} p log p = 0, entao descartamos p = 0
    return -np.sum(p * np.log2(p))


def kl_divergence_bits(p, q):
    """D(p || q) em bits. p e q sao vetores de probabilidade do mesmo tamanho."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


def mutual_information_bits(joint):
    """I(X;Y) em bits a partir de uma tabela conjunta p(x,y) (matriz 2D, soma 1)."""
    joint = np.asarray(joint, dtype=float)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px @ py
    mask = joint > 0
    return np.sum(joint[mask] * np.log2(joint[mask] / outer[mask]))


# ---------------------------------------------------------------------
# Passo 1: mundo real da simulacao
# ---------------------------------------------------------------------
outcomes = ["casa", "empate", "visitante"]
m = len(outcomes)
p_true = np.array([0.5, 0.3, 0.2])  # probabilidades verdadeiras
odds = np.array([2.0, 3.0, 5.0])  # cotacoes "o para 1" da casa

assert np.isclose(p_true.sum(), 1.0)

H_p = entropy_bits(p_true)
print(f"Entropia de p (verdadeira): {H_p:.4f} bits")


# ---------------------------------------------------------------------
# Passo 2: gerar temporadas (n rodadas) - vetorizado para muitas
# temporadas de uma so vez, como sugerido no Bloco 3
# ---------------------------------------------------------------------
def simulate_seasons(p, n_rounds, n_seasons, rng):
    """Matriz (n_seasons, n_rounds) de indices de resultado (0..m-1)."""
    return rng.choice(len(p), size=(n_seasons, n_rounds), p=p)


# ---------------------------------------------------------------------
# Passo 3: tres estrategias de aposta
# ---------------------------------------------------------------------
def strategy_all_in_favorite(p, odds):
    b = np.zeros(len(p))
    b[np.argmax(p)] = 1.0
    return b


def strategy_uniform(p, odds):
    return np.full(len(p), 1.0 / len(p))


def strategy_kelly(p, odds):
    return p.copy()


strategies = {
    "tudo no favorito": strategy_all_in_favorite,
    "uniforme": strategy_uniform,
    "kelly": strategy_kelly,
}


# ---------------------------------------------------------------------
# Passos 4-5: evolucao do patrimonio EM LOG (evita estouro numerico),
# para varias temporadas simuladas em paralelo
# ---------------------------------------------------------------------
def log_wealth_trajectory(outcomes_matrix, b, odds):
    """
    outcomes_matrix: (n_seasons, n_rounds) indices de resultado
    b: fracoes apostadas (fixas ao longo da temporada)
    Retorna log2(patrimonio) acumulado apos cada rodada (patrim. inicial = 1).
    """
    per_round_gain = np.log2(b[outcomes_matrix] * odds[outcomes_matrix])
    return np.cumsum(per_round_gain, axis=1)


n_rounds = 200
n_seasons = 1000

outcomes_matrix = simulate_seasons(p_true, n_rounds, n_seasons, rng)

log_wealth_by_strategy = {}
for name, strat_fn in strategies.items():
    b = strat_fn(p_true, odds)
    log_wealth_by_strategy[name] = log_wealth_trajectory(outcomes_matrix, b, odds)


# ---------------------------------------------------------------------
# Passo 6: comparar as tres estrategias
# ---------------------------------------------------------------------
summary_rows = []
for name, log_wealth in log_wealth_by_strategy.items():
    final_log_wealth = log_wealth[:, -1]
    growth_rate = final_log_wealth / n_rounds  # (1/n) log2(patrimonio final)
    summary_rows.append(
        {
            "estrategia": name,
            "taxa_crescimento_media": growth_rate.mean(),
            "taxa_crescimento_desvio": growth_rate.std(),
            "frac_quase_zero (log2<-20)": np.mean(final_log_wealth < -20),
            "frac_ganho_extraordinario (log2>20)": np.mean(final_log_wealth > 20),
        }
    )

summary_df = pd.DataFrame(summary_rows)
print("\nPasso 6 - comparacao entre estrategias:")
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------
# Passo 7: comparar com o valor teorico W(b, p)
# ---------------------------------------------------------------------
def theoretical_growth_rate(b, p, odds):
    return np.sum(p * np.log2(b * odds))


print("\nPasso 7 - taxa de crescimento teorica W(b,p):")
for name, strat_fn in strategies.items():
    b = strat_fn(p_true, odds)
    W_theo = theoretical_growth_rate(b, p_true, odds)
    print(f"  {name}: W teorico = {W_theo:.4f} bits/aposta")


# ---------------------------------------------------------------------
# Passo 8: AEP - histograma de (1/n) log2(patrimonio) para n crescente
# (estrategia de Kelly, que e a que "concentra" ao redor de W*)
# ---------------------------------------------------------------------
b_kelly = strategy_kelly(p_true, odds)
W_star = theoretical_growth_rate(b_kelly, p_true, odds)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n in zip(axes, [10, 100, 1000], strict=True):
    om = simulate_seasons(p_true, n, n_seasons, rng)
    lw = log_wealth_trajectory(om, b_kelly, odds)
    growth = lw[:, -1] / n
    ax.hist(growth, bins=40, density=True)
    ax.axvline(W_star, color="red", linestyle="--", label="W* teorico")
    ax.set_title(f"n = {n}")
    ax.set_xlabel("(1/n) log2(patrimonio final)")
    ax.legend()
plt.tight_layout()
plt.show()
print("\nPasso 8 - histograma AEP exibido na tela")


# ---------------------------------------------------------------------
# Passos 9-10: informacao lateral (dica Y correlacionada com X)
# ---------------------------------------------------------------------
# p(y | x): a dica "acerta" o resultado real com alta probabilidade, mas
# e imperfeita (matriz de confusao).
confusion = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.15, 0.7, 0.15],
        [0.1, 0.2, 0.7],
    ]
)  # linha = resultado real x, coluna = dica y -> p(y|x)

joint_xy = p_true[:, None] * confusion  # p(x,y) = p(x) p(y|x)
assert np.isclose(joint_xy.sum(), 1.0)

I_XY = mutual_information_bits(joint_xy)

p_y = joint_xy.sum(axis=0)
p_x_given_y = joint_xy / p_y[None, :]  # colunas normalizadas: p(x|y)


def simulate_seasons_with_hint(joint_xy, n_rounds, n_seasons, rng):
    m_, k_ = joint_xy.shape
    flat = joint_xy.flatten()
    idx = rng.choice(len(flat), size=(n_seasons, n_rounds), p=flat)
    x_idx = idx // k_
    y_idx = idx % k_
    return x_idx, y_idx


x_idx, y_idx = simulate_seasons_with_hint(joint_xy, n_rounds, n_seasons, rng)


def log_wealth_with_hint(x_idx, y_idx, p_x_given_y, odds):
    """Aposta condicional b_i(y) = p(x=i | y) a cada rodada."""
    b_used = p_x_given_y[:, y_idx]  # (m, n_seasons, n_rounds)
    b_chosen = np.take_along_axis(b_used, x_idx[None, :, :], axis=0)[0]
    o_chosen = odds[x_idx]
    per_round_gain = np.log2(b_chosen * o_chosen)
    return np.cumsum(per_round_gain, axis=1)


log_wealth_hint = log_wealth_with_hint(x_idx, y_idx, p_x_given_y, odds)
growth_hint = log_wealth_hint[:, -1].mean() / n_rounds

# Kelly SEM dica, na mesma amostra de X, para comparar de forma pareada
log_wealth_no_hint = log_wealth_trajectory(x_idx, b_kelly, odds)
growth_no_hint = log_wealth_no_hint[:, -1].mean() / n_rounds

delta_W_empirico = growth_hint - growth_no_hint

print("\nPassos 9-10 - informacao lateral:")
print(f"  Taxa de crescimento SEM dica:  {growth_no_hint:.4f} bits/aposta")
print(f"  Taxa de crescimento COM dica:  {growth_hint:.4f} bits/aposta")
print(f"  Ganho empirico (delta W):      {delta_W_empirico:.4f} bits/aposta")
print(f"  I(X;Y) previsto pela teoria:   {I_XY:.4f} bits/aposta")


# ---------------------------------------------------------------------
# Passo 11: apostador com crencas erradas q != p
# ---------------------------------------------------------------------
q_wrong = np.array([0.7, 0.2, 0.1])  # superestima o favorito
assert np.isclose(q_wrong.sum(), 1.0)

log_wealth_q = log_wealth_trajectory(outcomes_matrix, q_wrong, odds)
growth_q = log_wealth_q[:, -1].mean() / n_rounds

D_pq = kl_divergence_bits(p_true, q_wrong)
perda_empirica = W_star - growth_q

print("\nPasso 11 - crencas erradas:")
print(f"  W* (kelly com p verdadeiro):   {W_star:.4f} bits/aposta")
print(f"  W(q,p) empirico (kelly com q): {growth_q:.4f} bits/aposta")
print(f"  Perda empirica (W* - W(q,p)):  {perda_empirica:.4f} bits/aposta")
print(f"  D(p||q) teorico:                {D_pq:.4f} bits")
