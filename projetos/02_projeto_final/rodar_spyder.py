# %% [markdown]
# Script em células para Spyder/VS Code/Jupyter. Rode célula por célula (Ctrl+Enter).
# Mantenha este arquivo na mesma pasta dos demais .py.

# %% 0. CONFIGURAÇÃO  (edite aqui)
import os, sys
DATA = r""                # .zip ou pasta com os CSVs (preencha)
SAIDA = "resultados"      # figuras e CSVs
ANOS = None               # ex.: [2023, 2024, 2025]; None = todos
PERM = 100                # permutações (partes 2 e 3); use 20 para testar rápido
BOOT = 300                # bootstrap (parte 1)
MEIAS = 50                # meias-amostras para o IC da parte 2 (0 desliga)
SEED = 0

AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, AQUI)
os.chdir(AQUI)

import data, parte1_curva, parte2_draft, parte2b_campeoes_extra, parte2c_picks_detalhe, parte3_limite, parte4_draft_completo
import parte5_objetivos, parte6_lanes, parte7_t1

# %% 1. CARREGAR DADOS (a 1ª vez demora; depois usa o cache)
raw = data.load_raw(DATA, cache_dir=os.path.join(AQUI, "cache"))
games = data.team_games(raw, ANOS)
picks = data.picks(raw, ANOS)
print(len(raw), "linhas |", len(games), "partidas |", len(picks), "picks")

# %% 2. PARTE 1 - curva da informação no tempo
res1 = parte1_curva.run(games, SAIDA, B=BOOT, seed=SEED)

# %% 3. PARTE 2 - draft / campeões
tab2, bias2, champ2 = parte2_draft.run(picks, SAIDA, B=PERM, seed=SEED, bias_B=max(10, PERM // 3), n_meias=MEIAS)

# %% 3b. PARTE 2b - aprofundando os campeões
g2b, het2b = parte2b_campeoes_extra.run(picks, SAIDA)

# %% 3c. PARTE 2c - detalhando os picks (top contribuições, estabilidade, posições)
g2c, pos2c, rel2c = parte2c_picks_detalhe.run(picks, SAIDA)

# %% 4. PARTE 3 - limite de previsão (Fano, classificadores, regra da cadeia)
tab3, clf3, cadeia3, drag3 = parte3_limite.run(games, SAIDA, B=PERM, seed=SEED)

# %% 5. PARTE 4 - draft completo (escada de modelos, draft x time x ouro, robustez, pares)
drafts = data.drafts(raw, ANOS)
esc4, tg4, rob4, pares4, rp4 = parte4_draft_completo.run(drafts, SAIDA, seed=SEED, n_perm=5)

# %% 6. PARTE 5 - objetivos (isolados vs depois do ouro, cadeia, vazamento)
res5 = parte5_objetivos.run(data.objetivos(raw, ANOS), SAIDA, B=PERM, seed=SEED)

# %% 7. PARTE 6 - lanes (isoladas, dado as outras 4, no tempo, razão de chances)
res6 = parte6_lanes.run(data.lanes(raw, ANOS), SAIDA, seed=SEED)

# %% 8. PARTE 7 - a T1 em bits (Elo, surpresa das séries do Mundial, volcano plot)
res7 = parte7_t1.run(raw, SAIDA)

# %% 9. RELATÓRIO (HTML + PDF a partir de resultados/)
import runpy
runpy.run_path(os.path.join(AQUI, "relatorio", "gerar_relatorio.py"), run_name="__main__")
