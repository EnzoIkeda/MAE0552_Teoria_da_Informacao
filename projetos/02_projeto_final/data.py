"""Carregamento dos CSVs do Oracle's Elixir (pasta com CSVs ou .zip)."""
import glob
import os
import zipfile

import numpy as np
import pandas as pd

USECOLS = [
    "gameid", "datacompleteness", "league", "year", "patch", "side", "position",
    "teamname", "champion", "gamelength", "result",
    "firstdragon", "dragons", "opp_dragons",
    "golddiffat10", "golddiffat15", "golddiffat20", "golddiffat25",
    "xpdiffat15", "csdiffat15", "killsat15", "opp_killsat15",
    "firstPick", "ban1", "ban2", "ban3", "ban4", "ban5",   # parte 4 (draft completo)
    "firstblood", "firstherald", "firstbaron", "firsttower", "firstmidtower", "firsttothreetowers",   # parte 5
    "heralds", "opp_heralds", "void_grubs", "opp_void_grubs", "barons", "opp_barons",
    "kills", "deaths", "assists", "killsat15", "assistsat15", "deathsat15", "xpdiffat15", "csdiffat15",
    "golddiffat10", "golddiffat20", "golddiffat25",
    "towers", "opp_towers", "inhibitors", "opp_inhibitors", "turretplates", "opp_turretplates",
    "date", "playoffs", "playername",   # parte 7 (Elo cronológico e KDA por jogador)
]
MINUTES = (10, 15, 20, 25)


def _read(handle):
    # patch como texto: "13.10" não pode virar 13.1
    return pd.read_csv(handle, usecols=lambda c: c in USECOLS, dtype={"patch": str}, low_memory=False)


def load_raw(path, cache_dir=None, force=False):
    """Lê todos os CSVs (de uma pasta ou de um .zip) e guarda um cache em pickle."""
    path = str(path)
    if not os.path.exists(path) and os.path.exists(path + ".zip"):
        path += ".zip"  # Windows esconde a extensão .zip
    if not os.path.exists(path):
        raise FileNotFoundError(f"Caminho não encontrado: {path}")
    if os.path.isdir(path) and not glob.glob(os.path.join(path, "*.csv")):
        zips = glob.glob(os.path.join(path, "*.zip"))  # pasta que contém o zip
        if len(zips) == 1:
            path = zips[0]
    cache = os.path.join(cache_dir, "raw.pkl") if cache_dir else None
    if cache and os.path.exists(cache) and not force:
        return pd.read_pickle(cache)

    if not os.path.exists(path) and os.path.exists(path + ".zip"):
        path = path + ".zip"  # Windows pode esconder a extensão
    if not os.path.exists(path):
        raise FileNotFoundError(f"Caminho não encontrado: {path}")
    frames = []
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
                print(f"  lendo {name}")
                with zf.open(name) as fh:
                    frames.append(_read(fh))
    else:
        for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
            print(f"  lendo {os.path.basename(f)}")
            frames.append(_read(f))
    if not frames:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {path}")
    raw = pd.concat(frames, ignore_index=True)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        raw.to_pickle(cache)
    return raw


def team_games(raw, years=None):
    """Uma linha por partida, do ponto de vista do time AZUL (V = 1 se o azul venceu)."""
    t = raw[(raw.position == "team") & (raw.side == "Blue") & raw.result.notna()].copy()
    if years is not None:
        t = t[t.year.isin(years)]
    g = pd.DataFrame({
        "gameid": t.gameid.to_numpy(),
        "year": t.year.to_numpy(),
        "league": t.league.to_numpy(),
        # "complete" / "partial": ligas com dados parciais não têm as colunas "...at10/15/20/25"
        "completude": t.datacompleteness.fillna("NA").to_numpy(),
        "gamelength_min": (t.gamelength / 60).to_numpy(),
        "V": t.result.astype(int).to_numpy(),
    })
    for m in MINUTES:
        g[f"gd{m}"] = t[f"golddiffat{m}"].to_numpy()          # ouro azul - ouro vermelho
    g["xd15"] = t.xpdiffat15.to_numpy()
    g["cd15"] = t.csdiffat15.to_numpy()
    g["kd15"] = (t.killsat15 - t.opp_killsat15).to_numpy()
    # 1 = azul pegou o 1º dragão, -1 = vermelho, 0 = nenhum/desconhecido
    g["fd"] = t.firstdragon.map({1.0: 1, 0.0: -1}).fillna(0).astype(int).to_numpy()
    # ATENÇÃO: total no FIM do jogo (vazamento!)
    g["dd_final"] = (t.dragons - t.opp_dragons).fillna(0).to_numpy()
    return g.reset_index(drop=True)


def picks(raw, years=None):
    """Uma linha por (partida, time, campeão). V = 1 se o time do campeão venceu."""
    p = raw[(raw.position != "team") & raw.champion.notna() & raw.result.notna()].copy()
    if years is not None:
        p = p[p.year.isin(years)]
    p["patch"] = p.patch.fillna("NA")
    p["teamname"] = p.teamname.fillna("NA")
    out = pd.DataFrame({
        "gameid": p.gameid.to_numpy(),
        "side": p.side.to_numpy(),
        "year": p.year.to_numpy(),
        "league": p.league.to_numpy(),
        "patch": p.patch.to_numpy(),
        "team_year": (p.teamname + "|" + p.year.astype(str)).to_numpy(),
        "position": p.position.to_numpy(),
        "champion": p.champion.to_numpy(),
        "V": p.result.astype(int).to_numpy(),
    })
    return out.reset_index(drop=True)


POS = ("top", "jng", "mid", "bot", "sup")


def drafts(raw, years=None):
    """Uma linha por partida com o DRAFT COMPLETO (10 campeões por posição), do ponto de vista do azul.

    Colunas: gameid, year, league, patch, V (azul venceu), b_<pos> / r_<pos> (campeão do azul / vermelho),
    b_team, r_team (time-ano), fp (+1 azul tem a 1ª escolha, -1 vermelho, 0 desconhecido),
    b_bans, r_bans (ban1|...|ban5 em ordem; fases: ban1-3 antes dos picks, ban4-5 depois dos 3 primeiros picks), e as variáveis de jogo (gd15 etc.) de team_games().
    Só entram partidas com os 10 campeões conhecidos.
    """
    g = team_games(raw, years)
    pl = raw[(raw.position != "team") & raw.champion.notna() & raw.result.notna()]
    wide = pl.pivot_table(index="gameid", columns=["side", "position"], values="champion", aggfunc="first")
    cols = {}
    for side, tag in (("Blue", "b"), ("Red", "r")):
        for p in POS:
            cols[f"{tag}_{p}"] = wide[(side, p)]
    w = pd.DataFrame(cols).dropna().reset_index()

    t = raw[(raw.position == "team") & raw.result.notna()]
    tb = t[t.side == "Blue"].set_index("gameid")
    tr = t[t.side == "Red"].set_index("gameid")
    bancols = ["ban1", "ban2", "ban3", "ban4", "ban5"]

    def _bans(df):
        # ordem cronológica dos bans preservada (ban1..ban5); vazio = sem informação (ex.: 2014-2016 só têm 3 bans)
        return df[bancols].apply(lambda r: "|".join(x if isinstance(x, str) else "" for x in r), axis=1)

    ex = pd.DataFrame({
        "b_team": (tb.teamname.fillna("NA") + "|" + tb.year.astype(str)),
        "r_team": (tr.teamname.reindex(tb.index).fillna("NA") + "|" + tb.year.astype(str)),
        "fp": tb.firstPick.map({1.0: 1, 0.0: -1}).fillna(0).astype(int),
        "b_bans": _bans(tb),
        "r_bans": _bans(tr.reindex(tb.index)),
        "patch": tb.patch.fillna("NA"),
    }).reset_index()
    d = g.merge(w, on="gameid", how="inner").merge(ex, on="gameid", how="left")
    return d.reset_index(drop=True)


def objetivos(raw, years=None):
    """Uma linha por partida (visão do azul) com os OBJETIVOS (parte 5).

    Flags 'first*': +1 = azul fez primeiro, -1 = vermelho, NaN = sem dado (jogos 'partial').
    Sem timestamp na base: firstblood/firstdragon/firsttower acontecem quase sempre antes do min 15;
    firstherald depende da era (min 8 até 2023; min 14 a partir de 2024); firstmidtower, firsttothreetowers e firstbaron
    ocorrem muitas vezes DEPOIS do min 15.
    Totais '*_final' (dd, hd, bd, td, id, gr): azul - vermelho no FIM do jogo (consequência do resultado, não preditores).
    pl = placas de torre (azul - vermelho): as placas caem até ~min 14, então é quase uma variável do min 14 (2022+).
    """
    g = team_games(raw, years)
    t = raw[(raw.position == "team") & (raw.side == "Blue") & raw.result.notna()]
    if years is not None:
        t = t[t.year.isin(years)]
    t = t.reset_index(drop=True)
    sg = lambda c: t[c].map({1.0: 1, 0.0: -1}).to_numpy(float)
    o = g.copy()
    for nome, col in (("fb", "firstblood"), ("fd", "firstdragon"), ("fh", "firstherald"), ("ft", "firsttower"),
                      ("fmt", "firstmidtower"), ("f3t", "firsttothreetowers"), ("fbar", "firstbaron")):
        o[nome] = sg(col)
    o["pl"] = (t.turretplates - t.opp_turretplates).to_numpy(float)
    o["gr"] = (t.void_grubs - t.opp_void_grubs).to_numpy(float)
    o["dd"] = (t.dragons - t.opp_dragons).to_numpy(float)
    o["hd"] = (t.heralds - t.opp_heralds).to_numpy(float)
    o["bd"] = (t.barons - t.opp_barons).to_numpy(float)
    o["td"] = (t.towers - t.opp_towers).to_numpy(float)
    o["id"] = (t.inhibitors - t.opp_inhibitors).to_numpy(float)
    return o


def lanes(raw, years=None):
    """Uma linha por partida com, para cada lane (top, jng, mid, bot, sup), a diferença azul - vermelho de
    ouro/XP/CS (min 10 a 25), KDA aos 15 min e KDA final. Colunas: <variável>_<lane>, V, year."""
    p = raw[raw.position.isin(POS) & raw.result.notna()].copy()
    if years is not None:
        p = p[p.year.isin(years)]
    p["kda15"] = (p.killsat15 + p.assistsat15) / p.deathsat15.clip(lower=1)
    p["kdaF"] = (p.kills + p.assists) / p.deaths.clip(lower=1)
    p["net15"] = p.killsat15 + p.assistsat15 - p.deathsat15
    p["netF"] = p.kills + p.assists - p.deaths
    b = p[p.side == "Blue"].set_index(["gameid", "position"])
    r = p[p.side == "Red"].set_index(["gameid", "position"])
    j = b.join(r, lsuffix="_b", rsuffix="_r", how="inner")
    o = pd.DataFrame(index=j.index)
    o["V"] = j.result_b
    o["year"] = j.year_b
    for c in ("kda15", "kdaF", "net15", "netF"):
        o[c] = j[c + "_b"] - j[c + "_r"]
    for c in ("golddiffat10", "golddiffat15", "golddiffat20", "golddiffat25", "xpdiffat15", "csdiffat15"):
        o[c] = j[c + "_b"]  # nas linhas de jogador, a diferença já é "do jogador contra o oponente da mesma lane"
    w = o.reset_index().pivot(index="gameid", columns="position")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["V"] = w[[c for c in w.columns if c.startswith("V_")]].iloc[:, 0]
    w["year"] = w[[c for c in w.columns if c.startswith("year_")]].iloc[:, 0]
    return w.drop(columns=[c for c in w.columns if c.startswith("V_") or c.startswith("year_")])
