"""Projeto final de Teoria da Informação - League of Legends.

Uso:
    python main.py --data caminho/para/dados.zip            # roda as 7 partes
    python main.py --data pasta_com_csvs --partes 1 3       # só algumas partes
    python main.py --data dados.zip --anos 2023 2024 2025   # só alguns anos (a parte 7 usa sempre 2022-2025)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):      # nomes de time com símbolos (ex.: "∞") quebram o console do Windows
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
import data
import parte1_curva
import parte2_draft
import parte2b_campeoes_extra
import parte2c_picks_detalhe
import parte3_limite
import parte4_draft_completo
import parte5_objetivos
import parte6_lanes
import parte7_t1


# Se rodar sem argumentos (ex.: Spyder), usa este caminho. Edite aqui:
DATA_PADRAO = r""   # ex.: r"C:/caminho/para/dados.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PADRAO, help="pasta com os CSVs do Oracle's Elixir ou um .zip deles")
    ap.add_argument("--out", default="resultados", help="pasta de saída (figuras e CSVs)")
    ap.add_argument("--partes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7], choices=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--anos", nargs="+", type=int, default=None, help="filtra por ano (padrão: todos)")
    ap.add_argument("--perm", type=int, default=100, help="nº de permutações (partes 2, 3 e 5)")
    ap.add_argument("--boot", type=int, default=300, help="nº de reamostragens bootstrap (parte 1)")
    ap.add_argument("--meias", type=int, default=50,
                    help="nº de meias-amostras para o IC da informação corrigida (parte 2; 0 desliga)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recarregar", action="store_true", help="ignora o cache e relê os CSVs")
    a, _ = ap.parse_known_args()  # ignora argumentos extras do Spyder/Jupyter
    if not a.data:
        ap.error("informe --data ou preencha DATA_PADRAO no topo do main.py")

    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    print("Carregando dados...")
    raw = data.load_raw(a.data, cache_dir=os.path.join(here, "cache"), force=a.recarregar)
    faltam = [c for c in data.USECOLS if c not in raw.columns]
    if faltam:
        sys.exit(f"O cache não tem as colunas {faltam}: rode de novo com --recarregar")
    print(f"  {len(raw):,} linhas ({time.time() - t0:.0f}s)")

    if 1 in a.partes or 3 in a.partes:
        games = data.team_games(raw, a.anos)
        print(f"  {len(games):,} partidas (visão do time azul)")
    if 1 in a.partes:
        print("\n>>> PARTE 1: curva da informação no tempo")
        parte1_curva.run(games, a.out, B=a.boot, seed=a.seed)
    if 2 in a.partes:
        print("\n>>> PARTE 2: draft")
        parte2_draft.run(data.picks(raw, a.anos), a.out, B=a.perm, seed=a.seed,
                         bias_B=max(10, a.perm // 3), n_meias=a.meias)
        print("\n>>> PARTE 2b: aprofundando os campeões")
        parte2b_campeoes_extra.run(data.picks(raw, a.anos), a.out)
        print("\n>>> PARTE 2c: detalhando os picks")
        parte2c_picks_detalhe.run(data.picks(raw, a.anos), a.out)
    if 3 in a.partes:
        print("\n>>> PARTE 3: limite de previsão")
        parte3_limite.run(games, a.out, B=a.perm, seed=a.seed)
    if 4 in a.partes:
        print("\n>>> PARTE 4: draft completo (10 campeões por posição)")
        parte4_draft_completo.run(data.drafts(raw, a.anos), a.out, seed=a.seed, n_perm=max(3, a.perm // 20))
    if 5 in a.partes:
        print("\n>>> PARTE 5: objetivos (first blood, torres, dragão, arauto, barão...)")
        parte5_objetivos.run(data.objetivos(raw, a.anos), a.out, B=a.perm, seed=a.seed)
    if 6 in a.partes:
        print("\n>>> PARTE 6: qual lane é mais decisiva (ouro, XP, CS, KDA por lane)")
        parte6_lanes.run(data.lanes(raw, a.anos), a.out, seed=a.seed)
    if 7 in a.partes:
        print("\n>>> PARTE 7: a T1 em bits (favorito, azarão e surpresa das finais do Mundial)")
        parte7_t1.run(raw, a.out)
    print(f"\nPronto em {time.time() - t0:.0f}s. Resultados em: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
