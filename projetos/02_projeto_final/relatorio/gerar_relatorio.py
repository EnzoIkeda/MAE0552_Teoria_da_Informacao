"""Gera o relatório final (HTML -> PDF pelo Chrome sem interface) a partir de ../resultados.

Uso:  python gerar_relatorio.py            (gera relatorio.html e, se achar o Chrome, relatorio.pdf)
As tabelas vêm dos CSVs de resultados/; o texto está neste arquivo. Rode o main.py antes.
"""
import os
import shutil
import subprocess
import sys
from math import log2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(AQUI, "..", "resultados"))
FIG = os.path.join(AQUI, "fig")
os.makedirs(FIG, exist_ok=True)


# --------------------------------------------------------------------------- #
# Formatação (pt-BR) e blocos de HTML
# --------------------------------------------------------------------------- #
def nf(x, d=2):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def mb(x, d=2):            # bits -> milibits
    return nf(1000 * x, d)


def pc(x, d=1):
    return nf(100 * x, d) + "%"


def sg(x, d=1):
    return ("+" if x > 0 else "−" if x < 0 else "") + nf(abs(x), d)


def L(nome):
    return pd.read_csv(os.path.join(RES, nome))


def rel(caminho):           # caminho relativo ao HTML
    return os.path.relpath(caminho, AQUI).replace("\\", "/")


def box(tipo, titulo, corpo):
    return f'<div class="box {tipo}"><div class="bt">{titulo}</div>{corpo}</div>'


def tab(cab, linhas, leg, cls=""):
    th = "".join(f"<th>{c}</th>" for c in cab)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>" for l in linhas)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table><p class="cap">{leg}</p>'


def fig(caminho, leg, largura="100%"):
    return (f'<figure><img src="{rel(caminho)}" style="width:{largura}"/>'
            f'<figcaption>{leg}</figcaption></figure>')


def res_png(nome):
    return os.path.join(RES, nome)


def ul(itens):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in itens) + "</ul>"


def h2(x):
    p = min(max(x, 1e-12), 1 - 1e-12)
    return -(p * log2(p) + (1 - p) * log2(1 - p))


# --------------------------------------------------------------------------- #
# Números
# --------------------------------------------------------------------------- #
c1 = L("parte1_curva.csv")
P1 = c1[c1.amostra.str.startswith("comum")].set_index("minuto")
P1c = c1[c1.amostra == "completa por t"].set_index("minuto")
EXC = L("parte1_exclusoes_por_liga.csv")
EXC_TOT = EXC[["partidas", "sem_dados_min10", "sem_min25_curto", "sem_min25_outro"]].sum()
EXC_LIGAS = (EXC[EXC.sem_dados_min10 > 0].groupby("league").sem_dados_min10.sum()
             .sort_values(ascending=False))

T2 = L("parte2_informacao_campeao.csv").set_index("contexto")
D2 = L("parte2_diferenca_vs_marginal.csv").set_index("contexto")
VN = L("parte2_vies_vs_N.csv")
B2 = L("parte2b_campeoes.csv").set_index("champion")
HET = L("parte2b_heterogeneidade_patch.csv").set_index("champion")
C2 = L("parte2c_contribuicoes.csv").set_index("champion")
POS2 = L("parte2c_posicoes.csv")
R2 = L("parte2c_resumo.csv").set_index(L("parte2c_resumo.csv").columns[0]).valor

FANO = L("parte3_fano_por_faixas.csv")
CLF = L("parte3_classificadores.csv").set_index("metodo")
CAD = L("parte3_cadeia.csv")
DRA = L("parte3_dragao.csv")

ESC = L("parte4_escada.csv").set_index("modelo")
TO = L("parte4_time_ouro.csv").set_index("quantidade")
ROB = L("parte4_robustez.csv")
PRS = L("parte4_pares_resumo.csv").set_index("tipo")
PARES = L("parte4_pares.csv")

M5 = L("parte5_marginais.csv").set_index("objetivo")
DO5 = L("parte5_dado_ouro.csv")
CH5 = L("parte5_cadeia.csv")
PV5 = L("parte5_cadeia_placas_vermes.csv")
PF5 = L("parte5_posfato.csv")

I6 = L("parte6_isoladas.csv").set_index("lane")
S6 = L("parte6_sem_a_lane.csv")
TP6 = L("parte6_tempo.csv")
CO6 = L("parte6_coeficientes.csv").set_index("lane")

SP7 = L("parte7_surpresa_series.csv")
CA7 = L("parte7_campanhas.csv").set_index("ano")
RK7 = L("parte7_elo_pre_mundial.csv")
VR7 = L("parte7_volcano_resumo.csv")

HV = P1.loc[25, "H_V"]
cl = lambda k: CLF.loc[k]
teto4 = min(cl("logística").teto_do_limite_fano, cl("gradient boosting").teto_do_limite_fano)
f20 = FANO[(FANO.conjunto == "ouro") & (FANO.faixas_por_var == 20)].iloc[0]
f5 = FANO[(FANO.conjunto != "ouro") & (FANO.faixas_por_var == 5)].iloc[0]
f3 = FANO[(FANO.conjunto != "ouro") & (FANO.faixas_por_var == 3)].iloc[0]
f8 = FANO[(FANO.conjunto != "ouro") & (FANO.faixas_por_var == 8)].iloc[0]
cad = CAD.set_index("variavel")
t2 = lambda c, k="I_corrigida_perm": T2.loc[c, k]
esc = lambda m, k: ESC.loc[m, k]
to = lambda q: TO.loc[q]
ag = ROB[ROB.teste.str.contains("agregado")].set_index("modelo")
pl = ROB[ROB.teste.str.startswith("placebo")].set_index("modelo")
gbm = ROB[ROB.teste.str.startswith("gradient")].iloc[0]
picks_mbit = esc("M4", "I_mbit")
bans_mbit = esc("M7", "I_mbit")
ouro15_mbit = to("[min 15] I(ouro ; V | lado)").I_mbit
custo_m1 = 173 / (2 * 0.8 * 77349 * np.log(2)) * 1e3
# parte 2b
sd_obs = B2.wr.std(ddof=1)
se_rms = np.sqrt((B2.se ** 2).mean())
tau_b = np.sqrt(max(B2.wr.var(ddof=1) - (B2.se ** 2).mean(), 0))
tau_a = np.sqrt(max(B2.adj.var(ddof=1) - (B2.se ** 2).mean(), 0))
z2_b, z2_a = int((B2.z_raw.abs() > 2).sum()), int((B2.z_adj.abs() > 2).sum())
het15 = HET[HET.n_patches >= 15]
# parte 5
d5 = DO5[DO5.era == "2014-2026"].set_index("objetivo")
ref5 = d5[d5.index.str.startswith("[ref]")].I_marg_mbit.iloc[0]
c5o = CH5[CH5.ordem == "ouro primeiro"].set_index("bloco")
c5b = CH5[CH5.ordem != "ouro primeiro"].set_index("bloco")
pl5 = PV5[PV5.bloco.str.contains("placas")].iloc[0]
pl5_n = PV5[PV5.bloco.str.contains("placas")].amostra.iloc[0]
gr5 = PV5[PV5.bloco.str.contains("vermes")].iloc[0]
gr5_n = PV5[PV5.bloco.str.contains("vermes")].amostra.iloc[0]
era5 = lambda era, obj: DO5[(DO5.era == era) & (DO5.objetivo == obj)].I_dado_ouro_corr_mbit.iloc[0]
pf5 = PF5.set_index("bloco")
# parte 6
s6 = lambda mod: S6[S6.modelo == mod].set_index("lane")
tp6 = TP6.pivot(index="lane", columns="minuto", values="I")
# parte 7
S7 = SP7[SP7.momento == "antes da série"].reset_index(drop=True)
S7m = SP7[SP7.momento == "antes do Mundial"].set_index(["ano", "fase"])
s7 = lambda ano, fase: S7[(S7.ano == ano) & (S7.fase == fase)].iloc[0]
rk_t1 = RK7[RK7.time == "T1"].set_index("ano")
vr = lambda ano, time: VR7[(VR7.ano == ano) & (VR7.time == time)].iloc[0]


# --------------------------------------------------------------------------- #
# Figuras conceituais (A.1, A.2, S.1)
# --------------------------------------------------------------------------- #
AZ = "#1f3a5f"


def figuras_conceituais():
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig_, a = plt.subplots(figsize=(7.2, 3.6))
    a.plot(p, h, color=AZ, lw=2)
    for x, lab, off in ((0.5, "moeda justa\n1,000 bit", (-28, -34)), (0.75, "75/25\n0,811 bit", (9, 2)),
                        (0.88, "88/12\n0,529 bit", (9, 2)), (0.95, "95/5\n0,286 bit", (-58, -16))):
        a.scatter([x], [h2(x)], color="#2b6cb0", zorder=3)
        a.annotate(lab, (x, h2(x)), xytext=off, textcoords="offset points", fontsize=8, color="#2b6cb0")
    a.scatter([0.53], [h2(0.53)], color="#c53030", zorder=4)
    a.annotate("partida de LoL (azul vence 53%)\nH(V) = 0,997 bit", (0.53, h2(0.53)), xytext=(-150, 14),
               textcoords="offset points", fontsize=8, color="#c53030", arrowprops=dict(arrowstyle="-", color="#c53030"))
    a.set_xlabel("probabilidade de um dos lados vencer, p")
    a.set_ylabel("incerteza H (bits)")
    a.set_ylim(0, 1.15)
    a.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "A1_entropia.png"), dpi=170)
    plt.close()

    fig_, a = plt.subplots(figsize=(7.2, 2.6))
    for i, m in enumerate((10, 15, 20, 25)):
        I, Hc = P1.loc[m, "I_plugin"], P1.loc[m, "H_V_dado_X"]
        y = 3 - i
        a.barh(y, I, color="#2b6cb0")
        a.barh(y, Hc, left=I, color="#cbd5e0")
        a.text(I / 2, y, nf(I, 2), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        a.text(I + Hc / 2, y, f"{nf(Hc, 2)} (falta resolver)", ha="center", va="center", fontsize=8)
    a.set_yticks([3, 2, 1, 0])
    a.set_yticklabels(["min 10", "min 15", "min 20", "min 25"])
    a.set_xlim(0, 1.0)
    a.set_xlabel("bits (a barra inteira é H(V) ≈ 1 bit)")
    a.set_title("O ouro resolve uma parte crescente do 1 bit de incerteza", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "A2_ouro_barra.png"), dpi=170)
    plt.close()

    itens = [
        ("torres no fim do jogo (consequência) · P5", PF5[PF5.bloco.str.contains("torres")].I_isolada.iloc[0] * 1e3, "#c53030"),
        ("ouro no min 25 · P1", P1.loc[25, "I_plugin"] * 1e3, AZ),
        ("1º barão, isolado (nasce no min 20+) · P5", M5.loc["1º barão", "I_corrigida_mbit"], "#dd6b20"),
        ("ouro no min 15 · P1", P1.loc[15, "I_plugin"] * 1e3, AZ),
        ("ouro no min 10 · P1", P1.loc[10, "I_plugin"] * 1e3, AZ),
        ("1ª torre, isolada · P5", M5.loc["1ª torre", "I_corrigida_mbit"], "#3182ce"),
        ("força do time (time-ano) · P4", to("I(time ; V | lado)").I_mbit, "#38a169"),
        ("1º arauto, isolado · P5", M5.loc["1º arauto", "I_corrigida_mbit"], "#3182ce"),
        ("1º dragão, isolado · P5", M5.loc["1º dragão", "I_corrigida_mbit"], "#3182ce"),
        ("first blood, isolado · P5", M5.loc["first blood", "I_corrigida_mbit"], "#3182ce"),
        ("picks + bans (draft com bans) · P4", bans_mbit, "#38a169"),
        ("picks dos 10 campeões · P4", picks_mbit, "#38a169"),
        ("1 campeão · P2", t2("nenhum (marginal)") * 1e3, "#38a169"),
    ]
    fig_, a = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(itens))[::-1]
    a.barh(y, [v for _, v, _ in itens], color=[c for _, _, c in itens])
    for yi, (_, v, _) in zip(y, itens):
        a.text(v * 1.08, yi, nf(v, 1 if v >= 10 else 2), va="center", fontsize=8)
    a.set_yticks(y)
    a.set_yticklabels([n for n, _, _ in itens], fontsize=8)
    a.set_xscale("log")
    a.set_xlim(0.5, 3000)
    a.set_xlabel("milibits (escala logarítmica; 1000 mbit = 1 bit)")
    a.set_title("Informação sobre a vitória: valores isolados", fontsize=10)
    a.grid(alpha=.25, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "S1_regua.png"), dpi=170)
    plt.close()


# --------------------------------------------------------------------------- #
# Capítulos
# --------------------------------------------------------------------------- #
def capa():
    c7 = CA7.loc[2024]
    resumo = ul([
        f"<b>Ouro:</b> a diferença de ouro informa {nf(P1.loc[10,'I_plugin'],2)} / {nf(P1.loc[15,'I_plugin'],2)} / "
        f"{nf(P1.loc[20,'I_plugin'],2)} / {nf(P1.loc[25,'I_plugin'],2)} bit nos minutos 10, 15, 20 e 25. Mesmo aos 25 "
        f"minutos ainda restam {nf(P1.loc[25,'H_V_dado_X'],2)} bit de incerteza.",
        f"<b>Draft:</b> um campeão sozinho informa ≈ {mb(t2('nenhum (marginal)'),2)} milibit; os 10 picks juntos, "
        f"≈ {nf(picks_mbit,1)} milibits (com os bans, {nf(bans_mbit,1)}), quase tudo pela soma da força média de cada "
        f"campeão. Esses valores são limites inferiores: o modelo não usa a força de cada campeão em cada patch.",
        "<b>Viés:</b> estimadores ingênuos superestimam a informação em até duas ordens de grandeza; a correção por "
        "permutação é indispensável, e o embaralhamento precisa respeitar o vencedor e o perdedor de cada partida.",
        f"<b>Previsão no minuto 15:</b> erro mínimo prático ≈ 25% (contra 46,8% de um palpite ingênuo). O piso teórico de Fano fica abaixo de {pc(teto4,1)}. XP, CS e abates acrescentam "
        f"&lt; 0,01 bit depois do ouro.",
        f"<b>Objetivos:</b> isolados parecem informativos, mas depois de conhecer o ouro sobra ≈ 1 milibit ou menos, "
        f"exceto o 1º dragão (≈ {nf(d5.loc['1º dragão','I_dado_ouro_corr_mbit'],0)} milibits).",
        f"<b>Lanes:</b> sozinhas, meio ≈ atirador ≈ selva; dado as outras quatro, o atirador lidera "
        f"({nf(s6('ouro').loc['bot','I_dado_outras'],3)} bit), seguido de meio e top.",
        f"<b>A T1 em bits:</b> com a força de cada time medida antes de cada série, a campanha campeã de 2024 "
        f"acumulou {nf(c7.surpresa_total_bits,1)} bits de surpresa (uma chance de 1 em "
        f"{nf(1/c7.p_do_que_aconteceu,0)}), a maior entre 2022 e 2025; a final perdida em 2022, como favorita, "
        f"valeu {nf(s7(2022,'final').surpresa_bits,1)} bits.",
        "<b>Regra de leitura:</b> bits medem quanto prever melhora, não a importância causal de algo para o jogo.",
    ])
    return f"""
<section class="capa">
<div class="faixa"></div>
<p class="sobre">Teoria da Informação · MAE0552 · Projeto final</p>
<h1 class="titulo">Quais informações do jogo realmente dizem quem vai vencer?</h1>
<p class="sub">League of Legends medido em bits</p>
<p class="sub2">Relatório completo (Partes 1 a 7), com os conceitos de Teoria da Informação explicados passo a passo,
do zero, e um estudo de caso: a T1 no Mundial</p>
<div class="box resumo"><div class="bt">Resumo em uma página</div>
<p>Tratamos a vitória de uma partida profissional como uma variável aleatória binária <b>V</b> e medimos, em <b>bits</b>,
quanto cada informação do jogo reduz a incerteza sobre ela. São 103.132 partidas, 1.031.320 picks e 173 campeões
(Oracle's Elixir, 2014 a 2026).</p>{resumo}</div>
<p class="rodape-capa">Dados: Oracle's Elixir, 2014–2026 · Novembro de 2026<br/>
Partes 1 a 4: relatório principal; Partes 5 e 6: análises complementares; Parte 7: estudo de caso da T1. Todas as
tabelas e figuras foram geradas pelo código em <code>lol_info/</code> (ver Reprodutibilidade).</p>
</section>"""


def sumario():
    itens = [("uso", "Como usar este documento"), ("A", "A. Conceitos de Teoria da Informação, passo a passo"),
             ("B", "B. Dados e definições"), ("p1", "Parte 1 — A curva da informação no tempo"),
             ("p2", "Parte 2 — O draft, um campeão por vez"), ("p3", "Parte 3 — O limite de previsão no minuto 15"),
             ("p4", "Parte 4 — O draft completo"), ("p5", "Parte 5 — Os objetivos do jogo"),
             ("p6", "Parte 6 — Qual lane é a mais decisiva?"), ("p7", "Parte 7 — A T1 em bits: favorito, azarão e surpresa"),
             ("sintese", "Síntese e conclusões"), ("lim", "Limitações gerais"), ("repro", "Reprodutibilidade")]
    li = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in itens)
    return f'<section class="quebra"><h1>Sumário</h1><ol class="sum">{li}</ol></section>'


def uso():
    caixas = "".join(box(t, n, f"<p>{d}</p>") for t, n, d in (
        ("simples", "Em palavras simples", "Traduz a ideia para linguagem do dia a dia, sem fórmulas."),
        ("exemplo", "Exemplo numérico", "Mostra uma conta feita à mão com números pequenos, para você ver de onde vêm os valores."),
        ("ler", "Como ler", "Explica como ler cada coluna de uma tabela ou cada painel de uma figura."),
        ("atencao", "Atenção", "Avisa sobre armadilhas e erros comuns de interpretação."),
        ("vaz", "Cuidado: vazamento", "Marca variáveis que só existem depois do resultado (informação do futuro)."),
        ("aprend", "O que aprendemos", "Fecha a seção com a conclusão principal.")))
    return f"""<section class="quebra" id="uso"><h1>Como usar este documento</h1>
<p>O texto foi escrito para quem está começando em Teoria da Informação. Cada parte segue o mesmo roteiro: primeiro a
pergunta e uma explicação intuitiva, depois o que foi feito, as tabelas e figuras com um guia de leitura e, por fim, o
que se aprendeu. Para facilitar, algumas caixas coloridas se repetem:</p>{caixas}
<p><b>Sugestão de leitura.</b> Se os conceitos ainda não estão claros, leia primeiro o capítulo A. Se já os domina, vá
direto à Parte 1. A Parte 7 usa os conceitos para contar uma história (a T1 no Mundial), e a Síntese, no fim, reúne
tudo em uma só régua.</p></section>"""


def cap_A():
    tA1 = tab(["Probabilidade de um lado vencer", "Incerteza H (bits)", "Intuição"], [
        ["50% / 50%", "1,000", "moeda justa"], ["53% / 47%", "0,997", "uma partida de LoL (azul vence ~53%)"],
        ["60% / 40%", "0,971", "levemente favorável"], ["75% / 25%", "0,811", "favorito claro"],
        ["88% / 12%", "0,529", f"equivale à incerteza que sobra aos 25 min ({nf(P1.loc[25,'H_V_dado_X'],3)})"],
        ["95% / 5%", "0,286", "quase certo"], ["100% / 0%", "0,000", "sem incerteza"]],
        "Tabela A.1 — Entropia de um resultado binário para alguns valores de p.")
    tA2 = tab(["Grupo (X)", "Partidas", "Azul vence", "P(azul vence)", "Incerteza do grupo"], [
        ["Azul à frente no ouro", "500", "350", "70%", "0,881 bit"], ["Azul atrás no ouro", "500", "150", "30%", "0,881 bit"],
        ["Todas (sem saber X)", "1.000", "500", "50%", "1,000 bit = H(V)"]],
        "Tabela A.2 — Exemplo didático (números inventados).")
    glos = tab(["Termo", "Significado"], [
        ["Lado azul / vermelho", "Os dois times; o lado azul vence ~53% das partidas profissionais."],
        ["Ouro (gold)", "Recurso que os jogadores acumulam para comprar itens. “Diferença de ouro” = ouro do azul menos ouro do vermelho."],
        ["XP", "Experiência: define o nível dos campeões."],
        ["CS (creep score)", "Quantidade de tropas e monstros abatidos (farm)."],
        ["Abates, mortes, assistências", "Kills, deaths, assists. KDA = (abates + assistências) / max(mortes, 1)."],
        ["First blood", "O primeiro abate da partida."],
        ["Dragão", "Monstro neutro de objetivo; o 1º dragão é o primeiro a ser derrubado."],
        ["Arauto (Rift Herald)", "Monstro neutro; nasce no min 8 até 2023 e no min 14 desde 2024."],
        ["Barão", "Monstro neutro mais forte; nasce no min 20."],
        ["Torre, placas, inibidor", "Estruturas defensivas. As placas das torres externas só caem até ≈ min 14."],
        ["Vermes do vazio (grubs)", "Monstros neutros introduzidos em 2024."],
        ["Lanes", "Top, selva (jng), meio (mid), atirador (bot) e suporte (sup)."],
        ["Draft, pick, ban", "Fase de escolha: cada time bane campeões e escolhe (pick) cinco."],
        ["Patch", "Versão do jogo; mudanças de balanceamento alteram a força dos campeões."],
        ["Mundial (Worlds), MSI, seletiva", "Torneios internacionais; a seletiva regional decide as últimas vagas de cada liga para o Mundial."],
        ["Série md5", "Melhor de cinco: vence quem ganhar três jogos."]],
        "Tabela A.3 — Termos do jogo usados no texto.")
    cola = tab(["Medida", "Pergunta que responde", "Exemplo neste estudo", "Como ler"], [
        ["H(V)", "Quão incerto é o resultado antes de saber qualquer coisa?", "0,998 bit", "Teto: nenhuma variável informa mais que isso."],
        ["I(ouro em t; V)", "Saber o ouro no minuto t ajuda a prever quem vence?",
         f"{nf(P1.loc[10,'I_plugin'],2)} bit (min 10) → {nf(P1.loc[25,'I_plugin'],2)} bit (min 25)",
         "O ouro elimina 13% da incerteza aos 10 min e 47% aos 25."],
        ["H(V | X)", "Quanta incerteza sobra depois de saber X?", f"{nf(P1.loc[25,'H_V_dado_X'],3)} bit aos 25 min", "H(V | X) = H(V) − I(X; V)."],
        ["I(campeão; V | contexto Z)", "O campeão ainda diz algo depois que o contexto é conhecido?",
         f"{mb(t2('patch'),1)} mbit (patch); {mb(t2('time-ano + lado (extra)'),1)} mbit (time-ano)",
         "Mede o que o campeão acrescenta dentro de cada contexto."],
        ["Isolada vs. incremental", "A variável informa sozinha? E depois das outras?",
         f"XP: {nf(cad.loc['dif. de XP','I_isolada'],3)} isolado; {nf(cad.loc['dif. de XP','ganho_incremental'],3)} incremental",
         "Isolada muito maior = redundância."],
        ["Plug-in vs. corrigida", "Quanto do valor é sinal e quanto é ruído?",
         f"{mb(t2('patch + liga + lado','I_plugin'),1)} → {mb(t2('patch + liga + lado'),1)} mbit", "Só a estimativa corrigida é confiável."],
        ["Limite de Fano", "Qual o menor erro que qualquer previsor teria?",
         f"≈ 20% (min 15), no máximo {pc(teto4,1)}", "Piso teórico, só estimado; o erro real é ≈ 25%."],
        ["Surpresa −log₂ P", "Quão inesperado foi um resultado que aconteceu?",
         f"final de 2022: {nf(s7(2022,'final').surpresa_bits,2)} bits", "1 bit = acontecer algo de chance 50%; 3 bits = chance 12,5%."],
        ["IC 95% e p-valor", "O número é estável ou pode ser acaso?",
         f"[{nf(P1.loc[10,'I_lo'],4)}; {nf(P1.loc[10,'I_hi'],4)}] bit; p &lt; 0,01", "A magnitude vem dos bits, não do p."]],
        "Tabela A.4 — Guia de leitura das medidas. Bits medem quanto a previsão melhora, não a importância de algo para o jogo.")
    return f"""<section class="quebra" id="A"><h1>A. Conceitos de Teoria da Informação, passo a passo</h1>
<h2>A.1 A pergunta e a variável V</h2>
<p>Uma partida profissional de League of Legends (LoL) termina com um vencedor. Chamamos de <b>V</b> a variável que registra
isso: V = 1 se o time azul venceu e V = 0 se o vermelho venceu. Antes de saber qualquer coisa, o resultado é quase uma moeda:
o lado azul vence cerca de 53% das vezes.</p>
<p>A pergunta do projeto é: se eu descubro uma informação <b>X</b> sobre o jogo (a diferença de ouro no minuto 15, o campeão
escolhido, quem pegou o 1º dragão), <b>quanto da incerteza sobre V ela elimina?</b> A Teoria da Informação dá uma régua única,
a <i>informação mútua</i> medida em bits, que serve para comparar fontes de natureza muito diferente (números contínuos,
categorias, contagens).</p>
<h2>A.2 O bit e a entropia: medindo a incerteza</h2>
<p>Imagine que alguém lançou uma moeda justa e esconde o resultado. Para descobrir, você precisa de <b>uma pergunta de sim ou
não</b> (“deu cara?”). Essa é a unidade: <b>1 bit é a incerteza de uma moeda justa</b>. Se a moeda for viciada e cair de um
lado 95% das vezes, você já “quase sabe” o resultado e a incerteza cai para 0,29 bit. Se o resultado é certo, a incerteza é zero.</p>
<p>Essa incerteza se chama <b>entropia</b>. Para um resultado binário em que um dos lados ocorre com probabilidade p:</p>
<p class="eq">H(V) = −[ p · log<sub>2</sub>(p) + (1 − p) · log<sub>2</sub>(1 − p) ]</p>
<p>Não é preciso decorar a fórmula. O que importa é o formato (Figura A.1): a incerteza é máxima, 1 bit, quando p = 50%, e
diminui conforme o resultado fica mais previsível.</p>{tA1}
{fig(os.path.join(FIG,'A1_entropia.png'), 'Figura A.1 — Curva da entropia binária H(p). A partida de LoL (ponto vermelho) fica quase no topo: o resultado é quase uma moeda justa.', '78%')}
{box('simples','Em palavras simples','<p>“Bit” aqui não é o bit do computador. É uma medida de <b>surpresa</b>: um resultado que tinha probabilidade P traz −log<sub>2</sub> P bits de surpresa quando acontece (chance de 50% vale 1 bit; de 25%, 2 bits; de 12,5%, 3 bits). A entropia é a surpresa <i>média</i>: uma partida equilibrada vale ≈ 1 bit; uma partida cujo vencedor já é quase certo vale quase 0. A Parte 7 usa a surpresa de resultados específicos.</p>')}
<h2>A.3 Entropia condicional: a incerteza que sobra</h2>
<p>Suponha agora que você descobre X (por exemplo, o time azul está à frente no ouro aos 15 minutos). A <b>entropia
condicional</b> H(V | X) é a incerteza que sobra, em média, depois de saber X: calculamos a incerteza dentro de cada grupo de X
e fazemos a média ponderada pelo tamanho do grupo.</p>
{box('exemplo','Exemplo numérico','<p>Um exemplo inventado, só para ver a conta. Em 1.000 partidas, o azul vence 500 (então H(V) = 1 bit). Separamos as partidas pelo ouro no minuto 15:</p>' + tA2 + '<p>A incerteza que sobra é a média das incertezas dos dois grupos (500/1000 de cada): H(V | X) = 0,5 · 0,881 + 0,5 · 0,881 = <b>0,881 bit</b>.</p>')}
<h2>A.4 Informação mútua: quanto X reduz a incerteza</h2>
<p>A <b>informação mútua</b> entre X e V é a diferença entre a incerteza antes e depois de saber X:</p>
<p class="eq">I(X; V) = H(V) − H(V | X)</p>
<p>No exemplo: I = 1,000 − 0,881 = <b>0,119 bit</b>. Saber quem está à frente no ouro elimina cerca de 12% da incerteza sobre o
vencedor (coincidência feliz: é quase o valor real do minuto 10, {nf(P1.loc[10,'I_plugin'],3)} bit).</p>
{box('simples','Em palavras simples','<p>Se I(X; V) = 0, X não ajuda em nada. Se I(X; V) = 1 bit, X resolve o resultado por completo. A informação mútua nunca passa de H(V) (≈ 1 bit): esse é o teto. Ela detecta qualquer tipo de associação (não só a linear) e vale igual olhando de X para V ou de V para X.</p>')}
<p>Como as informações de LoL são pequenas, usamos o <b>milibit</b>: 1 mbit = 0,001 bit. Assim, o campeão escolhido informa
≈ {mb(t2('nenhum (marginal)'),2)} mbit; o ouro no minuto 15 informa ≈ {nf(P1.loc[15,'I_plugin']*1000,0)} mbit. A Figura A.2
mostra a ideia com os números reais da Parte 1.</p>
{fig(os.path.join(FIG,'A2_ouro_barra.png'), f'Figura A.2 — Fração do 1 bit de incerteza resolvida pelo ouro em cada minuto (amostra comum, N = {nf(P1.loc[25,"N"],0)}; Parte 1).', '78%')}
{box('atencao','Atenção','<p><b>Informação não é causalidade.</b> I(X; V) alta significa que X ajuda a <i>prever</i> V. Não significa que X <i>causa</i> V. Uma variável pode ser decisiva para o jogo e informar pouco porque outra já a contém, ou parecer decisiva porque já contém o resultado (vazamento).</p>')}
<h2>A.5 Informação condicional, redundância e complementaridade</h2>
<p>Muitas variáveis de LoL contam a mesma história. Ouro, XP e abates sobem juntos quando um time está ganhando. Então a
pergunta útil não é só “quanto X informa?”, mas <b>“quanto X acrescenta depois que já conheço Z?”</b>:</p>
<p class="eq">I(X; V | Z) = H(V | Z) − H(V | X, Z)</p>
<p>Pense em duas testemunhas de um acidente: se a segunda repete o que a primeira disse, acrescenta pouco. Comparamos:</p>
{ul(['<b>Isolada</b>: I(X; V), o quanto X informa sozinha.',
     '<b>Incremental</b>: I(X; V | anteriores), o quanto X acrescenta depois das outras.',
     f'<b>Isolada ≫ incremental → redundância.</b> Exemplo real (Parte 3): o XP informa {nf(cad.loc["dif. de XP","I_isolada"],3)} bit sozinho, mas só {nf(cad.loc["dif. de XP","ganho_incremental"],3)} bit depois do ouro.',
     f'<b>Incremental &gt; isolada → complementaridade.</b> Exemplo real (Parte 4): o draft informa {nf(to("[min 15] I(draft ; V | lado)").I_mbit,1)} mbit sozinho, mas {nf(to("[min 15] I(draft ; V | ouro)").I_mbit,1)} mbit depois do ouro. Parte disso é automática: quando X e Z são independentes, I(X; V | Z) − I(X; V) = I(X; Z | V) ≥ 0, isto é, conhecer Z sempre torna X mais informativo. O resto é complementaridade de fato (Parte 4).'])}
<h2>A.6 Regra da cadeia: somando contribuições</h2>
<p>A informação conjunta de várias variáveis se decompõe em uma soma de contribuições incrementais:</p>
<p class="eq">I(X<sub>1</sub>, …, X<sub>k</sub>; V) = Σ<sub>i</sub> I(X<sub>i</sub>; V | X<sub>1</sub>, …, X<sub>i−1</sub>)</p>
<p>É como dividir a conta de um restaurante: o total é fixo, mas a parcela de cada pessoa depende da ordem em que os pedidos
são contados. A <b>ordem altera os incrementos individuais, não a soma total</b>. Exemplo (Parte 3, minuto 15): ouro
{nf(cad.loc['dif. de ouro','ganho_incremental'],4)} + XP {nf(cad.loc['dif. de XP','ganho_incremental'],4)} + CS
{nf(cad.loc['dif. de CS','ganho_incremental'],4)} + abates {nf(cad.loc['dif. de abates','ganho_incremental'],4)} =
{nf(cad.loc['dif. de abates','I_acumulada'],4)} bit.</p>
<h2>A.7 O menor erro possível: Bayes e Fano</h2>
<p>Informação também diz o quanto se pode acertar. Chamamos de <b>erro</b> a fração de partidas em que o palpite está errado.
Três referências aparecem no texto:</p>
{ul(['<b>Palpite ingênuo</b>: apostar sempre no lado mais frequente (azul). Erra 46,8% das vezes.',
     '<b>Erro de Bayes</b>: o erro do melhor palpite possível que olha X. Para cada valor de X, aposta-se no lado mais provável. É o piso real de qualquer classificador que usa só X.',
     '<b>Limite de Fano</b>: um piso teórico calculado a partir de H(V | X):'])}
<p class="eq">P<sub>e</sub> ≥ h<sup>−1</sup>( H(V | X) )</p>
<p>onde h é a entropia binária (a curva da Figura A.1) e h<sup>−1</sup> é a sua inversa para valores menores que 50%. Em
palavras: “se depois de saber X ainda sobram H bits de incerteza, nenhum previsor consegue errar menos do que isso”.</p>
{box('exemplo','Exemplo numérico','<p>No exemplo da Tabela A.2, H(V | X) = 0,881 bit. A inversa dá 30%: nenhum palpite que use só “quem está à frente” erra menos de 30%. E o palpite “aposte em quem está à frente” erra exatamente 30% (150 + 150 = 300 erros em 1.000 partidas). Aqui Fano é justo porque os dois grupos são igualmente incertos.</p>')}
<p>Na vida real os grupos têm incertezas diferentes (uma vantagem de 5 mil de ouro é quase decisiva; uma de 200 é quase nada).
Por causa da concavidade de h, o erro verdadeiro fica acima do valor de Fano: Fano é <b>válido, mas folgado</b>. Há ainda um
cuidado prático: o H(V | X) verdadeiro não é observado, só estimado. Discretizar X aumenta a estimativa e o plug-in a
diminui, então o “limite” calculado é uma estimativa. O que se garante é um <b>teto</b> para ele: a log-loss de qualquer
modelo (A.9) é maior ou igual a H(V | X), então o limite verdadeiro não passa de h<sup>−1</sup>(log-loss). No minuto 15 isso
dá no máximo {pc(teto4,1)}, enquanto os classificadores reais erram ≈ 25%.</p>
<h2>A.8 O problema de estimar: o viés do plug-in</h2>
<p>Até aqui calculamos com probabilidades “verdadeiras”. Na prática só temos uma amostra, e estimamos as probabilidades pelas
frequências observadas (estimador <b>plug-in</b>). Ele funciona bem com poucos grupos e muitos dados, mas <b>superestima a
informação quando há muitos grupos com poucos dados</b>.</p>
<p>Intuição: se você sortear 20 partidas ao acaso e as separar em 10 grupos, quase sempre algum grupo vai parecer “favorecer”
o azul só por coincidência. Somando essas coincidências, o plug-in acusa informação mesmo quando ela não existe. Caso real
(Parte 2): ao estudar o campeão dentro de {nf(T2.loc['patch + liga + lado','n_estratos'],0)} contextos, o plug-in deu
{mb(t2('patch + liga + lado','I_plugin'),1)} mbit, dos quais {mb(t2('patch + liga + lado','I_nulo_media'),1)} eram puro
ruído; o sinal estimado é ≈ {mb(t2('patch + liga + lado'),1)} mbit.</p>
<p>Usamos estas ferramentas para controlar isso:</p>
{ul(['<b>Miller-Madow</b>: uma correção analítica de primeira ordem. Soma (m − 1)/(2N) nats a cada entropia, em que m é o número de células não vazias e N o número de observações. Ajuda, mas falha quando os estratos são muito esparsos.',
     '<b>Correção por permutação</b> (a principal). (1) Embaralhe os valores de V dentro de cada estrato de Z, o que preserva P(V | Z) e destrói qualquer relação com X. Quando o contexto não separa os lados, embaralha-se o resultado da <i>partida inteira</i> (azul ↔ vermelho), para manter um vencedor e um perdedor por partida. (2) Calcule a informação nesses dados: tudo o que aparecer é ruído. (3) Repita 100 vezes e tire a média: é o <i>ruído esperado</i>. (4) <b>Informação corrigida = observada − ruído esperado.</b>',
     '<b>p-valor</b>: fração das permutações cuja informação alcançou ou superou a observada. Com 100 permutações, o menor p possível é 1/101 ≈ 0,0099; escrevemos “p &lt; 0,01”.',
     '<b>Intervalos de confiança</b>: o <i>bootstrap</i> reamostra as partidas com reposição (300 vezes); o intervalo de 95% vai do percentil 2,5% ao 97,5%. Para a informação corrigida da Parte 2 usamos <i>meias-amostras</i> (metade das partidas, sorteada 50 vezes, sem reposição), porque duplicar linhas atrapalha a correção por permutação.'])}
<h2>A.9 Limite inferior por modelos: log-loss e validação cruzada</h2>
<p>Variáveis contínuas (ouro, XP) ou com muitas categorias (os 10 campeões de um draft) não cabem bem na contagem por células.
Nesses casos treinamos um <b>modelo</b> que prevê P(V | X) e medimos o quanto ele reduz a incerteza.</p>
<p>A medida é a <b>log-loss</b> (em bits): a média de −log<sub>2</sub> da probabilidade que o modelo deu ao que de fato
aconteceu, ou seja, a surpresa média do modelo. Um modelo que sempre diz “53%” tem log-loss igual a H(V) ≈ 1 bit; um modelo
melhor tem log-loss menor. Para qualquer modelo avaliado em dados que ele não viu, vale</p>
<p class="eq">I(X; V) ≥ H(V) − log-loss</p>
<p>Por isso é um <b>limite inferior</b>: se o modelo for fraco, subestimamos a informação; em média, nunca a superestimamos
(só o ruído de amostra pode empurrar para cima, e por isso há intervalos de confiança). Atenção: isso vale para o total de
cada modelo. O <i>incremento</i> entre dois modelos (a queda da log-loss ao acrescentar uma variável, usada na regra da
cadeia) é a diferença de dois limites inferiores e não tem direção garantida: se o modelo maior paga mais custo de estimar
parâmetros, o incremento sai subestimado e pode até ficar negativo.</p>
{ul(['<b>Validação cruzada (5 folds)</b>: dividimos as partidas em 5 blocos, treinamos com 4 e testamos no quinto, rodando os 5 arranjos. Assim o modelo não “decora” as respostas.',
     '<b>Regressão logística</b>: modelo simples, que soma pesos × variáveis e converte em probabilidade.',
     '<b>Gradient boosting</b>: modelo de muitas árvores de decisão, capaz de capturar relações não lineares.',
     '<b>Placebo</b>: com V embaralhado, a informação estimada tem que ser ≈ 0 (ou negativa). Foi o que aconteceu na Parte 4.'])}
<h2>A.10 Armadilhas: vazamento, seleção, significância e causalidade</h2>
{ul([f'<b>Vazamento temporal.</b> Só vale como preditor o que se sabe <i>antes</i> do momento da previsão. A diferença de torres derrubadas no fim do jogo carrega {nf(PF5[PF5.bloco.str.contains("torres")].I_isolada.iloc[0],2)} bit dos 0,997 de H(V). Não é mérito preditivo: para vencer é preciso derrubar as torres até o nexus.',
     '<b>Viés de seleção.</b> A diferença de ouro no minuto 25 só existe se o jogo durou pelo menos 25 minutos. Por isso a Parte 1 compara os minutos numa amostra comum. E ligas inteiras sem dados de tempo ficam de fora das Partes 1, 3, 5 e 6 (capítulo B).',
     f'<b>Significativo não é grande.</b> Com 1 milhão de picks, até {mb(t2("nenhum (marginal)"),2)} mbit tem p &lt; 0,01. O p-valor diz que o efeito não é acaso; a magnitude, em bits, diz se ele importa.',
     '<b>Correlação não é causa.</b> Um campeão pode ser decisivo em certos confrontos e se anular na média; a informação mede o que ajuda a prever, não o que causa a vitória.'])}
<h2>A.11 Glossário de League of Legends</h2>{glos}
<h2>A.12 Cola de medidas</h2>{cola}
</section>"""


def cap_B():
    lig = ", ".join(f"{k} ({nf(v,0)})" for k, v in EXC_LIGAS.head(4).items())
    tB = tab(["Parte", "Amostra", "N", "Observação"], [
        ["1", "jogos que chegaram ao min 25", nf(P1.loc[25, "N"], 0), f"amostra comum; H(V) = {nf(HV,4)} bit"],
        ["2", "picks (partida, time, campeão)", "1.031.320", "173 campeões; V = vitória do time do pick"],
        ["3", "jogos que chegaram ao min 15", nf(P1c.loc[15, "N"], 0), "H(V) = 0,9970 bit; erro ingênuo 0,468"],
        ["4", "partidas com os 10 campeões conhecidos", "103.132", "25% para ajuste de regularização; 77.349 na validação cruzada"],
        ["5", "min 15 e todos os first*", "89.501", "H(V) = 0,997 bit"],
        ["6", "5 lanes completas até o min 15", "92.747", "tempo: amostra comum do min 25"],
        ["7", "jogos de 2014 a 2025 das principais ligas, MSI e Mundial", "—", "Elo antes de cada série; séries da T1 no Mundial de 2022 a 2025"]],
        "Tabela B.1 — Amostras usadas em cada parte.")
    return f"""<section class="quebra" id="B"><h1>B. Dados e definições</h1>
<p>Os dados vêm do <b>Oracle's Elixir</b>: 13 arquivos CSV (2014 a 2026) com 1.237.584 linhas. Cada partida tem 10 linhas de
jogadores e 2 de times. Foram usadas <b>103.132 partidas</b> (206.264 pares partida-lado) e <b>1.031.320 picks</b> de
<b>173 campeões</b> distintos.</p>
{ul(['<b>V</b> = 1 se o time azul venceu (Partes 1, 3, 4, 5 e 6; uma linha por partida). H(V) ≈ 0,998 bit, pois o azul vence ~53%. Na Parte 2 a unidade é o <i>pick</i> (partida, time, campeão) e V é a vitória do time que fez o pick. Na Parte 7 a unidade é a <i>série</i> (melhor de cinco).',
     '<b>Variáveis de jogo</b>: diferença de ouro (azul menos vermelho) nos minutos 10, 15, 20 e 25; diferenças de XP, CS e abates no min 15; first blood, 1º dragão, 1º arauto, 1ª torre e outros objetivos (Parte 5); ouro, XP, CS e KDA por lane (Parte 6).',
     '<b>Discretização</b>: variáveis contínuas são divididas em faixas de mesma frequência (quantis), por exemplo 20 faixas com 5% das partidas cada. Há análise de sensibilidade ao número de faixas.',
     f'<b>Ligas sem dados de tempo</b>: {nf(EXC_TOT.sem_dados_min10,0)} partidas ({pc(EXC_TOT.sem_dados_min10/EXC_TOT.partidas,0)}) não têm a diferença de ouro nem no minuto 10. Não é duração (nenhum jogo profissional acaba antes do minuto 10): são ligas marcadas como dados parciais, principalmente {lig}. As Partes 1, 3, 5 e 6 valem para as ligas com dados completos.',
     f'<b>Viés de seleção temporal</b>: entre as partidas com dados, {nf(EXC_TOT.sem_min25_curto,0)} acabaram antes do minuto 25. Por isso a Parte 1 usa uma <b>amostra comum</b> (jogos que chegaram ao minuto 25), que compara os minutos de forma justa, e mostra também a amostra completa por minuto.'])}
{tB}
{box('ler','Como ler','<p>H(V) varia na 3ª casa (0,998; 0,9976; 0,9970) porque a proporção de vitórias do lado azul muda um pouco conforme a amostra.</p>')}
</section>"""


def parte1():
    linhas = [[str(m), f"{nf(P1.loc[m,'I_plugin'],4)} [{nf(P1.loc[m,'I_lo'],4)}; {nf(P1.loc[m,'I_hi'],4)}]",
               nf(P1.loc[m, "I_logit_cv"], 4), nf(P1.loc[m, "H_V_dado_X"], 4), pc(P1.loc[m, "frac_resolvida"]),
               nf(P1.loc[m, "erro_min_fano"], 3), nf(P1.loc[m, "erro_bayes_plugin"], 3), nf(P1c.loc[m, "I_plugin"], 4)]
              for m in (10, 15, 20, 25)]
    t = tab(["Min", "I(ouro; V) plug-in [IC 95%]", "Limite inf. (logística, CV)", "H(V | ouro) [bits]", "Incerteza resolvida",
             "Erro mín. (Fano, estimado)", "Erro de Bayes (plug-in)", "I (amostra completa em t)"], linhas,
            f"Tabela 1.1 — Amostra comum (N = {nf(P1.loc[25,'N'],0)}; H(V) = {nf(HV,4)} bit), 20 faixas de quantis para o ouro.")
    return f"""<section class="parte" id="p1"><h1>Parte 1 — A curva da informação no tempo</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> quanto a diferença de ouro no minuto t informa sobre quem vence, e em que minuto a partida já está “decidida”?</p><p>Medimos o quanto o ouro nos livra da incerteza em quatro momentos (minutos 10, 15, 20 e 25). A surpresa é quanto ainda sobra de incerteza em jogos profissionais, que são muito equilibrados.</p>')}
<h2>1.1 O que foi feito</h2>
{ul(['Variável X: diferença de ouro (azul − vermelho) no minuto t, dividida em 20 faixas de mesma frequência.',
     f'Amostra comum: {nf(P1.loc[25,"N"],0)} jogos que chegaram ao minuto 25 (H(V) = {nf(HV,4)} bit), para comparar os minutos de forma justa.',
     'Estimador principal: informação mútua plug-in, com IC 95% por bootstrap (300 reamostragens). Conferência independente: limite inferior variacional (regressão logística, validação cruzada de 5 folds).',
     'Também calculamos H(V | ouro), a estimativa do limite de Fano e o erro de Bayes plug-in (ver capítulo A).'])}
<h2>1.2 Resultados</h2>{t}
{box('ler','Como ler',ul(['<b>I(ouro; V)</b>: bits de incerteza que o ouro elimina. <b>H(V | ouro)</b>: o que sobra (os dois somam H(V)).','<b>Incerteza resolvida</b> = I / H(V).','<b>Erro mín. (Fano)</b> é a estimativa do piso teórico; <b>Erro de Bayes</b> é o erro do melhor palpite que só olha o ouro.','<b>Exemplo:</b> no minuto 15 o ouro elimina 22% da incerteza, e o melhor palpite baseado nele ainda erra ~27% das partidas.']))}
{fig(res_png('parte1_curva.png'), 'Figura 1.1 — Esquerda: informação mútua entre a diferença de ouro e a vitória (barras de erro: bootstrap 95%; pontilhado: amostra completa por minuto). Centro: incerteza restante H(V | ouro) e fração resolvida. Direita: P(azul vence) em função da diferença de ouro, por minuto.')}
<h2>1.3 Leitura dos resultados</h2>
{ul([f'<b>A informação cresce de forma acelerada.</b> De {nf(P1.loc[10,"I_plugin"],3)} bit (min 10) para {nf(P1.loc[25,"I_plugin"],3)} bit (min 25). A incerteza cai de 0,998 para {nf(P1.loc[25,"H_V_dado_X"],3)} bit: 47% foi resolvida aos 25 minutos. Só o ouro não “decide” o jogo profissional: 25% da incerteza se resolve por volta do minuto 16 (interpolação linear entre os 4 pontos) e 50% não é atingido até o minuto 25.',
     '<b>Confirmação independente.</b> A estimativa por discretização é confirmada por um limite inferior (regressão logística em validação cruzada), com diferença máxima de ~0,001 bit. O IC de 95% é estreito (≈ ±0,004 bit) e o resultado é estável ao número de faixas.',
     f'<b>Seleção.</b> Na amostra completa a informação nos minutos iniciais é maior ({nf(P1c.loc[10,"I_plugin"],3)} bit no min 10, contra {nf(P1.loc[10,"I_plugin"],3)}). Jogos que acabam cedo (stomps) somem das colunas tardias, e a amostra comum, formada só por jogos longos e mais equilibrados, subestima a informação “real” dos primeiros minutos. A comparação entre minutos, porém, só é justa na amostra comum.',
     '<b>O mesmo saldo vale menos tarde.</b> O painel da direita da Figura 1.1 mostra que, com +3 mil de ouro (faixa de 2,5 a 3,5 mil), P(vitória) é 0,89 no minuto 10 e 0,78 no minuto 25. No fim do jogo as diferenças absolutas são naturalmente maiores, então 3 mil já é um saldo comum.'])}
{box('aprend','O que aprendemos',f'<p>O ouro resolve 13%, 22%, 32% e 47% da incerteza nos minutos 10, 15, 20 e 25. Em jogos profissionais a partida raramente está decidida só pelo ouro, mesmo aos 25 minutos: ainda sobra {nf(P1.loc[25,"H_V_dado_X"],2)} bit, comparável à incerteza de uma moeda que cai de um lado ~88% das vezes.</p>')}
</section>"""


def parte2():
    ctxs = ["nenhum (marginal)", "lado", "patch", "liga", "patch + liga + lado", "time-ano + lado (extra)"]
    linhas = [[c, nf(T2.loc[c, "n_estratos"], 0), mb(T2.loc[c, "I_plugin"]), mb(T2.loc[c, "I_MM"]),
               mb(T2.loc[c, "I_nulo_media"]), f"<b>{mb(T2.loc[c,'I_corrigida_perm'])}</b>",
               f"[{mb(T2.loc[c,'IC95_lo'])}; {mb(T2.loc[c,'IC95_hi'])}]",
               "&lt; 0,01" if T2.loc[c, "p_valor"] < 0.0100001 else nf(T2.loc[c, "p_valor"], 2)] for c in ctxs]
    t21 = tab(["Contexto Z", "Estratos", "Plug-in [mbit]", "Miller-Madow [mbit]", "Ruído esperado (nulo) [mbit]",
               "Corrigida (perm.) [mbit]", "IC 95% (corrigida)", "p-valor"], linhas,
              "Tabela 2.1 — I(campeão; V | Z) em milibits (1 mbit = 0,001 bit), 173 campeões, 100 permutações. IC 95% por 50 meias-amostras (metade das partidas, sem reposição). O menor p-valor possível com 100 permutações é 0,0099.")
    dif = lambda c: (D2.loc[c, "dif_vs_marginal"], D2.loc[c, "IC95_lo"], D2.loc[c, "IC95_hi"])
    dp, dl, dt = dif("patch"), dif("liga"), dif("time-ano + lado (extra)")
    top = C2.head(10)
    lin22 = [[c, nf(r.n, 0), pc(r.pickrate_pct / 100), pc(r.wr), nf(r.contrib_mbit, 4), pc(r.pct_do_total / 100),
              sg(r.adj_pp), nf(abs(r.z_adj), 1), f"{nf(r.wr_ate2023,1)}% → {nf(r.wr_2024mais,1)}%"] for c, r in top.iterrows()]
    t22 = tab(["Campeão", "Picks", "Pick rate", "Win rate", "Contrib. [mbit]", "% de I", "WR − esperada (time-ano) [p.p.]",
               "|z| ajust.", "WR até 2023 → 2024+"], lin22,
              "Tabela 2.2 — Os 10 campeões que mais contribuem para I(campeão; V) (plug-in). Pick rate = % dos 206.264 pares partida-lado em que o campeão foi escolhido.", "pequena")
    pos = []
    for c in top.index:
        q = POS2[POS2.campeao == c]
        pos.append([c, "; ".join(f"{p} {nf(w,1)}% ({nf(n,0)})" for p, w, n in zip(q.posicao, q.winrate_pct, q.n))])
    t23 = tab(["Campeão", "Win rate por posição (nº de picks; só posições com ≥ 300 picks)"], pos,
              "Tabela 2.3 — Win rate por posição dos 10 campeões do topo (sup = suporte, jng = selva, mid = meio, top = topo).", "pequena")
    b = lambda c: ((B2.loc[c, "wr"] - .5) * 100, B2.loc[c, "adj"] * 100)
    tf, mp, dr, xe = b("Twisted Fate"), b("Malphite"), b("Darius"), b("Xerath")
    vn = VN[VN.contexto == "patch + liga + lado"].set_index("frac").I_corrigida_perm * 1e3
    het_top = ", ".join(het15.head(4).index)
    karma = C2.loc["Karma"]
    return f"""<section class="parte" id="p2"><h1>Parte 2 — O draft, um campeão por vez</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> um campeão parece forte porque é bom ou porque aparece em contextos favoráveis (patch, liga, time forte)?</p><p>Comparamos I(campeão; V), que olha o campeão sozinho, com I(campeão; V | contexto), que mede o que o campeão diz <i>dentro</i> de cada contexto.</p>')}
<h2>2.1 Quanto o campeão informa sobre a vitória</h2>{t21}
{box('ler','Como ler',ul(['“Contexto Z” é o que se mantém fixo ao medir a informação do campeão. “Estratos” é o número de grupos que o contexto cria.','“Plug-in” é o cálculo ingênuo. “Ruído esperado” é o que o cálculo ingênuo daria <i>mesmo que o campeão não tivesse relação alguma com a vitória</i> (obtido embaralhando V).',f'“Corrigida” = plug-in − ruído, a melhor estimativa do sinal real. <b>Exemplo:</b> em patch + liga + lado, o plug-in dá {mb(t2("patch + liga + lado","I_plugin"),1)} mbit, mas {mb(t2("patch + liga + lado","I_nulo_media"),1)} são ruído.']))}
{fig(res_png('parte2_informacao.png'), 'Figura 2.1 — Esquerda: informação por contexto (escala symlog), com o IC 95% da corrigida; quase todo o valor do plug-in nos contextos finos é ruído. Direita: estimativas em função do tamanho da amostra (subamostras de partidas). O plug-in cai com N; a corrigida fica perto do valor final, mas em patch + liga + lado ainda cresce com N (ver o último ponto da leitura).')}
{ul([f'<b>O sinal é minúsculo, mas real.</b> I(campeão; V) ≈ {mb(t2("nenhum (marginal)"),2)} mbit (corrigida; IC [{mb(t2("nenhum (marginal)","IC95_lo"))}; {mb(t2("nenhum (marginal)","IC95_hi"))}]), cerca de {nf(t2("nenhum (marginal)")/HV*100,3)}% de H(V), e estatisticamente distinta de zero.',
     f'<b>O viés domina.</b> Com o contexto patch + liga + lado ({nf(T2.loc["patch + liga + lado","n_estratos"],0)} estratos × 173 campeões), o plug-in dá {mb(t2("patch + liga + lado","I_plugin"),0)} mbit, dos quais {mb(t2("patch + liga + lado","I_nulo_media"),0)} mbit ({pc(t2("patch + liga + lado","I_nulo_media")/t2("patch + liga + lado","I_plugin"),1)}) são puro ruído; a estimativa corrigida é {mb(t2("patch + liga + lado"),2)} mbit. Miller-Madow corrige só parte do viés porque é uma aproximação de primeira ordem que falha com estratos esparsos.',
     f'<b>Patch e liga não são confundidores.</b> Em cada partida há exatamente um vencedor e um perdedor no mesmo patch e liga, então P(V | patch, liga) = 50% por construção e I(patch; V) = 0. Pela regra da cadeia, I(X; V | Z) = I(X; V) + I(Z; V | X) − I(Z; V); com I(Z; V) = 0, condicionar só pode <i>aumentar</i> a informação: dentro de um patch o campeão é mais informativo ({mb(t2("patch"),2)} mbit, {sg(dp[0]*1e3,2)} mbit sobre o marginal, IC da diferença [{mb(dp[1])}; {mb(dp[2])}]), porque a força dos campeões muda com o tempo. Na liga, a diferença ({sg(dl[0]*1e3,2)} mbit, IC [{mb(dl[1])}; {mb(dl[2])}]) é menor.',
     f'<b>A força do time confunde campeões específicos, não o agregado.</b> Condicionar em time-ano + lado dá {mb(t2("time-ano + lado (extra)"),2)} mbit (IC [{mb(t2("time-ano + lado (extra)","IC95_lo"))}; {mb(t2("time-ano + lado (extra)","IC95_hi"))}]), sem redução em relação ao marginal (diferença {sg(dt[0]*1e3,2)} mbit, IC [{mb(dt[1])}; {mb(dt[2])}]). Se a força do time explicasse a informação do campeão, ela cairia. O efeito aparece campeão a campeão (seção 2.3), e a Parte 4 confirma que, no draft inteiro, condicionar no time muda pouco.',
     f'<b>Contextos muito finos escondem o sinal.</b> Em patch + liga + lado e em time-ano + lado a maioria das células (estrato × campeão) tem 1 ou 2 picks. Célula de 1 pick não carrega sinal nenhum para o plug-in, então a corrigida tende a ficar <i>abaixo</i> da verdade quanto menos dados por célula. Em patch + liga + lado isso aparece com clareza: com metade das partidas a estimativa é {mb(t2("patch + liga + lado","I_corrigida_media_meias"),1)} mbit, com todas é {mb(t2("patch + liga + lado"),1)} mbit, e ela ainda cresce com N (Figura 2.1, direita); o valor deve ser lido como limite inferior. Em time-ano + lado o intervalo é largo demais ({mb(t2("time-ano + lado (extra)","IC95_lo"))} a {mb(t2("time-ano + lado (extra)","IC95_hi"))} mbit) para dizer a direção.'])}
<h2>2.2 Win rate bruta versus ajustada pelo contexto</h2>
{fig(res_png('parte2_campeoes.png'), 'Figura 2.2 — Campeões com ≥ 500 picks. Eixo x: win rate bruta menos 50%; eixo y: win rate menos a esperada dado o contexto (leave-one-out por partida-lado). Barras: IC 95% binomial. Pontos sobre a diagonal indicam que o contexto não explica a diferença.')}
<p>Com o contexto patch + liga + lado, os pontos ficam sobre a diagonal (r = 0,99): o contexto não altera a win rate dos
campeões, como previsto acima. Com time-ano + lado (r = 0,92), alguns campeões se movem de forma apreciável.</p>
<h2>2.3 Aprofundamento: ruído, força real e mudanças ao ajustar pelo time</h2>
{box('simples','Três ferramentas desta seção',ul(['<b>z (desvio / erro-padrão):</b> mede a que distância a win rate de um campeão está do esperado, em unidades do ruído de amostra. |z| &gt; 2 acontece por acaso em ~5% dos casos.','<b>Empirical Bayes (encolhimento):</b> parte da dispersão observada entre campeões é ruído; “encolhemos” as win rates extremas em direção à média para estimar a força real.','<b>Q de Cochran:</b> teste de que a win rate de um campeão é constante entre patches.']))}
{ul([f'<b>A dispersão é maior que o acaso.</b> Entre os {len(B2)} campeões com ≥ 500 picks, o desvio-padrão da win rate é {nf(sd_obs*100,2)} p.p., contra {nf(se_rms*100,2)} p.p. de ruído amostral típico. {z2_b} campeões têm |z| &gt; 2 (esperados por acaso: ~{nf(0.0455*len(B2),0)}). O desvio-padrão da <i>força real</i> entre campeões é ~{nf(tau_b*100,1)} p.p. (bruta) e ~{nf(tau_a*100,1)} p.p. (ajustada por time-ano).',
     f'<b>Mudanças ao ajustar pelo time</b> (p.p. sobre 50%): Twisted Fate cai de {sg(tf[0])} para {sg(tf[1])} (escolhido por times fortes); Malphite sobe de {sg(mp[0])} para {sg(mp[1])} e Darius de {sg(dr[0])} para {sg(dr[1])} (escolhidos por times fracos, então a win rate bruta os subestima); Xerath vai de {sg(xe[0])} para {sg(xe[1])}. No conjunto, porém, o ajuste muda pouco: a força real cai de ~{nf(tau_b*100,1)} para ~{nf(tau_a*100,1)} p.p. e os campeões com |z| &gt; 2 vão de {z2_b} para {z2_a}.',
     '<b>Cuidado com o ajuste.</b> Time-ano é calculado com os mesmos resultados e inclui o efeito dos campeões que o time escolhe (um time que joga campeões fortes ganha mais). Ajustar por ele pode tirar do campeão parte do próprio efeito, então a win rate ajustada tende a subestimar a força real.',
     f'<b>A força varia entre patches.</b> O Q de Cochran rejeita win rate constante por patch para {int((HET.p<0.01).sum())} de {len(HET)} campeões (p &lt; 0,01; por acaso seriam ~{nf(0.01*len(HET),0)}). Entre os campeões com muitos patches destacam-se {het_top}.'])}
{fig(res_png('parte2b_campeoes_extra.png'), 'Figura 2.3 — Esquerda: distribuição do z bruto e ajustado, versus N(0,1), que seria o caso de “só ruído”. Centro: win rate final após ajuste por time-ano e encolhimento empirical Bayes. Direita: win rate por patch dos três campeões com maior heterogeneidade (≥ 15 patches).')}
<h2>2.4 Então os campeões não decidem a partida?</h2>
<p>A conclusão correta é mais estreita do que “campeões não importam”: <b>saber qual campeão foi escolhido, sozinho, diz muito
pouco sobre quem vai vencer</b>. Quatro razões:</p>
{ul(['<b>Quase todos os campeões ficam perto de 50%.</b> Escolher Rakan (53,6% de vitória) muda a aposta de 50% para 53,6%. Em bits, um campeão com 55% de win rate valeria só ~0,007 bit por pick, e a maioria dos 173 campeões está bem mais perto de 50%.',
     f'<b>Uma escala diferente.</b> Para quem monta estratégia, 55% é um campeão forte; para prever um jogo isolado, é quase uma moeda. Os campeões diferem em força real (~{nf(tau_a*100,1)} a {nf(tau_b*100,1)} p.p.), mas essa diferença é pequena frente à incerteza de cada partida.',
     f'<b>O ouro resume o que aconteceu.</b> Aos 15 min, o ouro sozinho carrega ~0,26 bit, da ordem de 300 vezes a informação do campeão. Ele reflete a habilidade dos jogadores, as decisões e também o efeito dos campeões.',
     '<b>Hipótese (não testada):</b> como os times escolhem e banem, campeões muito fortes tendem a ser banidos ou disputados pelos dois lados, o que empurra as win rates para perto de 50%.'])}
{box('atencao','Atenção — o que esta parte não mediu',ul(['O draft completo: medimos um campeão por vez; combinações, sinergias e confrontos ficam para a Parte 4.','Efeito causal: pouca informação não significa pouco efeito no jogo.','O efeito do campeão que aparece depois via ouro já está “contado” no ouro.']))}
<h2>2.5 Detalhando os picks: quem mais contribui e por quê</h2>
<p>A informação mútua marginal se decompõe <b>exatamente</b> em uma parcela por campeão:</p>
<p class="eq">I(campeão; V) = Σ<sub>c</sub> p(c) · D( P(V | c) ‖ P(V) )</p>
<p>Cada parcela é o produto de (i) <i>quão frequente</i> é o pick, p(c), por (ii) <i>quão longe de 50%</i> está a win rate do
campeão, medida em bits pela divergência D (para V ≈ 50/50, D = 1 − h(win rate)).</p>
{box('exemplo','Exemplo numérico',f'<p><b>Karma.</b> p(c) = {nf(karma.n,0)} / 1.031.320 = {pc(karma.n/1031320,2)} dos picks. A win rate de {pc(karma.wr)} dá D = {nf(karma.kl_bits,4)} bit por pick. A parcela é ≈ {nf(karma.contrib_mbit,4)} mbit, ou {pc(karma.pct_do_total/100)} do total. A soma das parcelas dos 173 campeões é {nf(R2["total"],4)} mbit, igual ao I plug-in marginal da Tabela 2.1.</p>')}
{t22}
{box('ler','Como ler','<p>“Contribuição” é a parcela do campeão na informação total; “% de I” a divide pelo total. “WR − esperada” compara a win rate com o esperado dado o time-ano de quem o escolheu. “|z| ajustado” divide essa diferença pelo erro-padrão: acima de ~3, é improvável ser acaso.</p>')}
{fig(res_png('parte2c_picks_detalhe.png'), 'Figura 2.4 — Esquerda: as 15 maiores contribuições (rótulo = win rate). Centro e direita: a win rate de cada campeão (≥ 250 picks por metade) em duas metades da amostra, aleatórias (centro) e temporais (direita); pontos na diagonal indicam força estável.')}
{ul([f'<b>Frequência × distância a 50%.</b> Rakan está em 2º lugar porque é escolhido em 8,3% dos pares partida-lado, com +3,6 p.p.; Twisted Fate chega ao 3º com só 2,0% de pick rate porque sua win rate é a mais distante de 50% (56,2%). Campeões fracos contam tanto quanto fortes: 4 dos 10 primeiros (Karma, Lulu, Viktor, K\'Sante) têm win rate abaixo de 50%.',
     f'<b>A informação é difusa.</b> Os 10 primeiros respondem por {pc(R2["top10_share"])} do total e os 20 primeiros por {pc(R2["top20_share"])}.',
     f'<b>Sinal, não ruído.</b> O ruído esperado de cada campeão é ≈ 1/(2 N ln 2) = {nf(R2["piso"],4)} mbit, e as parcelas do top 10 ficam entre 0,020 e 0,066 mbit, de ~29 a ~95 vezes o piso. Atenção: estas parcelas são plug-in; a correção rigorosa vale para o total.',
     f'<b>Estabilidade na amostra.</b> Dividindo os jogos ao acaso em duas metades, a win rate dos campeões correlaciona r = {nf(R2["r_rand"],2)} entre as metades ({nf(R2["n_rand"],0)} campeões; mesmo lado de 50% em {pc(R2["same_sign"],0)} deles), e todos os 10 primeiros mantêm sinal e magnitude.',
     f'<b>Estabilidade no tempo é menor.</b> Comparando até 2023 com 2024 em diante, r cai para {nf(R2["r_temp"],2)} ({nf(R2["n_temp"],0)} campeões): a força muda com a era do jogo e os patches. Os fracos (Karma, Lulu, Viktor) se aproximam de 50% em 2024+ (Viktor: 46,1% → 48,4%).',
     '<b>Efeito do contexto.</b> Descontando a força do time, Twisted Fate cai de +6,2 para +3,2 p.p. e Bard de +4,3 para +3,5; Rakan e Viktor encolhem ~0,7 p.p.; Camille e K\'Sante não mudam.',
     '<b>Posição importa.</b> Muitos campeões são jogados em mais de uma posição, e a força pode estar numa só (Tabela 2.3): Maokai vence 54,9% na selva e 50,7% no top; Camille é forte só no top; Karma é fraco em todas.'])}
{t23}
{box('aprend','O que aprendemos','<p>Os campeões que mais “informam” são os frequentes e consistentemente acima ou abaixo de 50%. As diferenças são reais e estáveis na amostra, parcialmente explicadas pela força do time em alguns casos (Twisted Fate, Bard) e pela posição em outros (Maokai, Camille). Mesmo assim, entre os 10 primeiros nenhum passa de ~6 p.p. de distância a 50%, o que explica por que a informação do campeão isolado é pequena.</p>')}
</section>"""


def parte3():
    lin = [["Palpite ingênuo (lado mais frequente)", "—", "—", nf(0.468, 3)],
           ["Limite de Fano estimado, só ouro (20 faixas)", nf(f20.H_V_dado_X_MM, 4), nf(f20.fano_MM, 3), "—"],
           ["Limite de Fano estimado, ouro+XP+CS+abates (5 faixas/var)", nf(f5.H_V_dado_X_MM, 4), nf(f5.fano_MM, 3), "—"],
           ["<b>Teto do limite de Fano</b>, só ouro: h⁻¹(log-loss da logística)", nf(cl("logística (só ouro)").logloss, 4) + " (log-loss)", f"≤ {nf(cl('logística (só ouro)').teto_do_limite_fano,3)}", "—"],
           ["<b>Teto do limite de Fano</b>, 4 variáveis: h⁻¹(log-loss da logística)", nf(cl("logística").logloss, 4) + " (log-loss)", f"≤ {nf(teto4,3)}", "—"],
           ["Erro de Bayes plug-in em CV, ouro (20 faixas)", "—", "—", nf(f20.bayes_plugin_cv, 3)],
           ["Erro de Bayes plug-in em CV, 4 variáveis (5 faixas)", "—", "—", nf(f5.bayes_plugin_cv, 3)],
           ["Logística, só ouro (CV)", nf(cl("logística (só ouro)").logloss, 4) + " (log-loss)", "—", nf(cl("logística (só ouro)").erro, 3)],
           ["Logística, 4 variáveis (CV)", nf(cl("logística").logloss, 4) + " (log-loss)", "—", nf(cl("logística").erro, 3)],
           ["Gradient boosting, 4 variáveis (CV)", nf(cl("gradient boosting").logloss, 4) + " (log-loss)", "—", nf(cl("gradient boosting").erro, 3)]]
    t31 = tab(["Método", "H(V | X) ou log-loss [bits]", "Limite de Fano", "Erro observado"], lin,
              "Tabela 3.1 — Limite teórico versus erro de classificadores. “4 variáveis” = diferenças de ouro, XP, CS e abates no minuto 15. Fano estimado com H(V | X) corrigida por Miller-Madow. A log-loss de um modelo é um teto para H(V | X), então h⁻¹(log-loss) é um teto para o limite verdadeiro.")
    lc = [[("<b>" + v + "</b>") if "VAZ" in v else v, nf(r.I_isolada, 4), nf(r.ganho_incremental, 4), "± " + nf(1.96 * r.ep, 4),
           nf(r.I_acumulada, 4)] for v, r in cad.iterrows()]
    t32 = tab(["Variável (ordem da cadeia)", "Isolada I(Xi; V)", "Incremental I(Xi; V | anteriores)", "IC 95% (incremental)",
               "Acumulada"], lc, "Tabela 3.2 — Informação em bits, estimada pela queda da log-loss (gradient boosting, 5-fold CV). A linha “VAZAMENTO” usa o total de dragões no fim do jogo, que não existiria no minuto 15.")
    ld = [[r.variavel, r.condicao, mb(r.I_plugin), mb(r.I_MM), mb(r.I_nulo_media), mb(r.I_corrigida)] for _, r in DRA.iterrows()]
    t33 = tab(["Variável", "Condição", "Plug-in [mbit]", "Miller-Madow [mbit]", "Nulo [mbit]", "Corrigida [mbit]"], ld,
              "Tabela 3.3 — Informação exata (discreta) com teste de permutação (100 permutações, p &lt; 0,01 em todas as linhas).")
    fd = DRA.set_index(["variavel", "condicao"])
    fd_m, fd_o = fd.iloc[0].I_corrigida * 1e3, fd.iloc[1].I_corrigida * 1e3
    return f"""<section class="parte" id="p3"><h1>Parte 3 — O limite de previsão no minuto 15</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> com tudo o que se sabe até o minuto 15, qual é o menor erro possível ao prever o vencedor? E quanto cada variável acrescenta depois das anteriores?</p><p>Comparamos o piso teórico (Fano) com classificadores reais e usamos a regra da cadeia para ver quais variáveis são redundantes.</p>')}
<p>Amostra: {nf(P1c.loc[15,'N'],0)} partidas que chegaram ao minuto 15 com dados completos. H(V) = 0,9970 bit; o erro de quem
sempre aposta no lado mais frequente é 0,468.</p>
<h2>3.1 Fano versus classificadores reais</h2>{t31}
{fig(res_png('parte3_erro_limite.png'), 'Figura 3.1 — Esquerda: limite de Fano (estimado e teto) e erros observados. Direita: sensibilidade da estimativa de Fano ao número de faixas da discretização; a linha tracejada clara é o teto dado pela log-loss.')}
{ul([f'O melhor erro obtido é ≈ {pc(cl("logística").erro)} (logística com 4 variáveis), versus 46,8% do palpite ingênuo: em 3 de cada 4 partidas o palpite acerta. Um modelo não linear (gradient boosting) não melhora a logística, e acrescentar XP, CS e abates ao ouro reduz o erro em apenas ~0,6 p.p.',
     f'<b>O limite de Fano fica em torno de 20%, e não passa de {pc(teto4)}.</b> A estimativa por discretização dá {pc(f5.fano_MM)}, mas ela não é um piso garantido: o H(V | X) verdadeiro não é observado, e discretizar aumenta a estimativa. O que se garante é um teto: a log-loss da logística ({nf(cl("logística").logloss,4)} bit) é ≥ H(V | X), então o limite verdadeiro é no máximo h⁻¹({nf(cl("logística").logloss,4)}) = {pc(teto4)}, cerca de 5 p.p. abaixo do erro alcançado.',
     f'<b>Fano é folgado para V binário</b>: usa a entropia média H(V | X) e, por concavidade de h, o erro verdadeiro E[min(p, 1−p)] fica acima de h⁻¹(H(V | X)). O erro de Bayes plug-in em validação cruzada (≈ {pc(f5.bayes_plugin_cv)}) concorda com os classificadores, sugerindo que ~25% é próximo do mínimo prático com essas variáveis.',
     f'<b>Sensibilidade.</b> Com 4 variáveis, a estimativa de Fano cai de {nf(f3.fano_MM,3)} (3 faixas) para {nf(f8.fano_MM,3)} (8 faixas) e não converge: com mais faixas as células ficam esparsas e o Miller-Madow deixa de corrigir o viés. O intervalo bootstrap da Tabela de sensibilidade (no CSV) usa o bootstrap “básico”, que corrige esse viés e fica acima da estimativa ingênua quando há muitas faixas.'])}
<h2>3.2 Regra da cadeia: o que cada variável acrescenta</h2>{t32}
{box('ler','Como ler',f'<p>“Isolada” é quanto a variável informa sozinha; “Incremental” é quanto acrescenta depois das anteriores da lista. Se isolada ≫ incremental, a variável é redundante. “Acumulada” soma os incrementais. <b>Exemplo:</b> o XP informa {nf(cad.loc["dif. de XP","I_isolada"],3)} bit sozinho, mas só {nf(cad.loc["dif. de XP","ganho_incremental"],3)} bit depois do ouro.</p>')}
{fig(res_png('parte3_cadeia.png'), 'Figura 3.2 — Esquerda: cascata da regra da cadeia, que soma os ganhos incrementais até o teto H(V). Direita: informação isolada versus incremental; a diferença mede a redundância entre variáveis.')}
{ul([f'<b>O ouro sozinho dá {nf(cad.loc["dif. de ouro","I_isolada"],3)} bit.</b> Depois dele, XP acrescenta só {nf(cad.loc["dif. de XP","ganho_incremental"],4)} bit, CS {nf(cad.loc["dif. de CS","ganho_incremental"],4)} e abates {nf(cad.loc["dif. de abates","ganho_incremental"],4)}: isoladamente cada um é informativo, mas quase tudo é <b>redundante</b> com o ouro.',
     f'<b>O primeiro dragão é a variável mais útil depois do ouro:</b> acrescenta {nf(cad.loc["1º dragão","ganho_incremental"],4)} bit, embora isolado carregue {nf(cad.loc["1º dragão","I_isolada"],3)} bit. É informação de natureza diferente (controle de objetivo).',
     f'<b>Vazamento temporal.</b> O total de dragões no fim do jogo “acrescenta” {nf(cad.iloc[-1].ganho_incremental,3)} bit, ~17 vezes mais que o primeiro dragão. É ilusório: trata-se de informação do futuro. O primeiro dragão também não tem horário na base, embora ocorra quase sempre antes do minuto 15.'])}
<h2>3.3 Teste dedicado: o 1º dragão depois de conhecer o ouro</h2>{t33}
{box('aprend','O que aprendemos',f'<p>Condicionar no ouro reduz a informação do primeiro dragão de ~{nf(fd_m,1)} para ~{nf(fd_o,1)} mbit: cerca de dois terços do seu valor preditivo já estavam no ouro. No minuto 15 o erro mínimo prático é ~25%, com o ouro respondendo por quase tudo; XP, CS e abates são redundantes com ele, e o primeiro dragão é o único complemento relevante. Incluir variáveis do fim do jogo infla a informação sem nenhum valor preditivo.</p>')}
</section>"""


def parte4():
    lin = []
    for m, r in ESC.iterrows():
        g = "—" if m == "M0" else f"{sg(r.ganho_mbit,2)} ± {nf(1.96*r.ganho_ep_mbit,2)}"
        lin.append([m, r.descricao, nf(r.n_colunas, 0), f"{nf(r.I_mbit,2)} ± {nf(1.96*r.I_ep_mbit,2)}", g, pc(r.acerto)])
    t41 = tab(["Modelo", "Acrescenta", "Colunas", "I acumulada [mbit]", "Ganho incremental [mbit]", "Acerto"], lin,
              "Tabela 4.1 — Escada do draft completo (1 mbit = 0,001 bit). Acerto de referência (sempre apostar no azul): 53,3%.")
    q = ["I(draft ; V | lado)", "I(time ; V | lado)", "I(draft ; V | time)", "I(time ; V | draft)", "I(bans fase 1 ; V | draft, time)",
         "I(bans fase 2 ; V | draft, time, fase 1)", "I(ban × pick ; V | tudo acima, time)", "[min 15] I(draft ; V | lado)",
         "[min 15] I(ouro ; V | lado)", "[min 15] I(draft ; V | ouro)", "[min 15] I(ouro ; V | draft)"]
    t42 = tab(["Quantidade", "Valor [mbit]", "IC 95%"], [[x, nf(TO.loc[x, "I_mbit"], 2), "± " + nf(1.96 * TO.loc[x, "ep_mbit"], 2)] for x in q],
              "Tabela 4.2 — Informações condicionais (draft = M4, só picks; time = time-ano; ouro = diferença no min 15, partidas que chegaram ao min 15).")
    rb = [[f"origem móvel, 2019–2026 (agregado)", m, nf(ag.loc[m, "I_mbit"], 2), "± " + nf(1.96 * ag.loc[m, "ep_mbit"], 2)] for m in ("M0", "M1", "M2", "M4")]
    rb += [[f"placebo (V embaralhado, 5 rep.)", m, nf(pl.loc[m, "I_mbit"], 2), "± " + nf(1.96 * pl.loc[m, "ep_mbit"], 2)] for m in ("M2", "M4")]
    rb += [["gradient boosting categórico (10 campeões)", gbm.modelo, nf(gbm.I_mbit, 2), "± " + nf(1.96 * gbm.ep_mbit, 2)]]
    t43 = tab(["Teste", "Modelo", "I [mbit]", "IC 95% / dp"], rb,
              "Tabela 4.3 — Origem móvel: para cada ano Y (2019 a 2026) treina-se com todos os anos anteriores e testa-se em Y. No placebo o IC é ±1,96 dp das repetições.")
    pr = [[t, nf(PRS.loc[t, "pares_testados"], 0), nf(PRS.loc[t, "dp_z"], 2), pc(PRS.loc[t, "frac_abs_z_gt2"]), pc(PRS.loc[t, "esperado_se_so_ruido"]),
           nf(PRS.loc[t, "n_q_lt_005"], 0)] for t in ("ban × pick do adversário", "ban × pick do próprio time", "confronto", "sinergia")]
    t44 = tab(["Tipo", "Pares testados", "dp de z", "|z| &gt; 2 observado", "|z| &gt; 2 esperado", "q &lt; 0,05"], pr,
              "Tabela 4.4 — Resumo dos testes par a par. Os testes se sobrepõem (as mesmas partidas entram em vários pares), então z não são independentes; o dp de z é a medida global.")
    tops = []
    for t in ("sinergia", "confronto", "ban × pick do adversário"):
        s = PARES[PARES.tipo == t].reindex(PARES[PARES.tipo == t].z.abs().sort_values(ascending=False).index).head(4)
        for _, r in s.iterrows():
            nome = f"{r.A} + {r.B}" if t == "sinergia" else (f"{r.A} × {r.B} ({r.posicao})" if t == "confronto" else f"ban {r.A} × pick adv. {r.B}")
            tops.append([t, nome, nf(r.n, 0), pc(r.win_obs), pc(r.win_esp), nf(r.z, 1), nf(r.q, 2)])
    t45 = tab(["Tipo", "Par", "n", "Vitória obs.", "Vitória esp.", "z", "q"], tops,
              "Tabela 4.5 — Os 4 maiores |z| de cada tipo (ban × pick: só do adversário). Nenhum sobrevive à correção para múltiplos testes.", "pequena")
    ganho_int = esc("M4", "I_mbit") - esc("M1", "I_mbit")
    return f"""<section class="parte" id="p4"><h1>Parte 4 — O draft completo</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> quanto o draft <i>completo</i> informa sobre a vitória, e esse total é maior que a soma das partes?</p><p>A Parte 2 mediu um campeão por vez. Aqui a variável é o draft inteiro: os 10 campeões por posição e, depois, os bans.</p>')}
<h2>4.1 Método: uma escada de modelos</h2>
<p>Há ~10<sup>9</sup> drafts possíveis por lado e só 103.132 partidas. Contar células (plug-in) é inviável: quase todo draft
aparece uma única vez. Usamos então o limite inferior variacional (A.9), I(X; V) ≥ H(V) − log-loss em validação cruzada.</p>
<p>Montamos uma <b>escada de modelos aninhados</b> (regressão logística com penalização L2, sobre uma matriz esparsa). Cada
degrau acrescenta um tipo de informação, e a queda da log-loss entre degraus consecutivos estima I(novo; V | anteriores).
Cada campeão entra com sinal +1 para o azul e −1 para o vermelho, o que dá simetria entre os lados.</p>
{ul(['<b>M0</b>: lado + quem tem a 1ª escolha. <b>M1</b>: os campeões, com efeito aditivo (cada campeão soma sua força média).','<b>M2</b>: campeão × posição. <b>M3</b>: confrontos de lane (A × B na mesma posição). <b>M4</b>: sinergias (pares do mesmo time).','<b>M5</b>: bans da fase 1 (ban1 a ban3, antes de qualquer pick). <b>M6</b>: bans da fase 2 (ban4 e ban5, depois dos primeiros picks; portanto reativos). <b>M7</b>: interação ban × pick.','A ordem dos bans vem das colunas ban1 a ban5 (cobertura ~98% desde 2017; em 2014–2016 só há 3 bans por time).'])}
{box('ler','Como ler',ul(['<b>Controle do viés de seleção de modelo.</b> A regularização é escolhida em um conjunto de ajuste de 25% das partidas, que <i>não</i> entra na avaliação; as estimativas vêm de validação cruzada de 5 folds nos 77.349 jogos restantes.','IC 95% = ±1,96 erro-padrão da diferença de log-loss por partida. O total de cada degrau é um limite inferior: em média só erra para baixo, e o placebo confirma que não há viés positivo. O <i>ganho</i> entre degraus é a diferença de dois limites e não tem direção garantida.']))}
<h2>4.2 A escada: quanto cada tipo de informação do draft acrescenta</h2>{t41}
{fig(res_png('parte4_escada.png'), 'Figura 4.1 — Esquerda: ganho incremental de cada degrau. Centro: draft × força do time × ouro (escala log). Direita: robustez (origem móvel por ano, placebo com V embaralhado, gradient boosting).')}
{ul([f'<b>Os picks informam ~{nf(picks_mbit,1)} mbit</b> (≈ {nf(picks_mbit/1000/HV*100,2)}% de H(V)); com os bans, a escada chega a {nf(bans_mbit,1)} mbit (≈ {nf(bans_mbit/1000/HV*100,1)}%). Quase tudo ({nf(esc("M1","I_mbit"),1)} mbit, {pc(esc("M1","I_mbit")/picks_mbit,0)} dos picks) vem do efeito <b>aditivo</b> dos campeões. O M1 é coerente com a Parte 2: 10 × {mb(t2("nenhum (marginal)"),2)} ≈ {nf(10*t2("nenhum (marginal)")*1000,1)} mbit, menos o custo de estimar 173 coeficientes em validação cruzada (da ordem de {nf(custo_m1,1)} mbit, k/(2n) sem regularização), dá ~{nf(esc("M1","I_mbit"),1)} mbit.',
     f'Posição ({sg(esc("M2","ganho_mbit"),2)}), confrontos de lane ({sg(esc("M3","ganho_mbit"),2)}) e sinergias ({sg(esc("M4","ganho_mbit"),2)} ± {nf(1.96*esc("M4","ganho_ep_mbit"),2)}) acrescentam menos de 1 mbit cada. As interações existem na amostra, mas são pequenas e, como mostra a robustez, não generalizam entre eras.',
     f'<b>Bans: a informação está nos bans pré-pick, não nos reativos.</b> Os bans da fase 1 acrescentam {sg(esc("M5","ganho_mbit"),2)} mbit; os da fase 2, {sg(esc("M6","ganho_mbit"),2)}; a interação ban × pick, só {sg(esc("M7","ganho_mbit"),2)}. Depois de conhecer draft e time, sobram {nf(to("I(bans fase 1 ; V | draft, time)").I_mbit,2)} ± {nf(1.96*to("I(bans fase 1 ; V | draft, time)").ep_mbit,2)} mbit (fase 1) e {nf(to("I(bans fase 2 ; V | draft, time, fase 1)").I_mbit,2)} ± {nf(1.96*to("I(bans fase 2 ; V | draft, time, fase 1)").ep_mbit,2)} (fase 2); a interação ban × pick fica em {nf(to("I(ban × pick ; V | tudo acima, time)").I_mbit,2)} ± {nf(1.96*to("I(ban × pick ; V | tudo acima, time)").ep_mbit,2)} mbit, negativa com IC que exclui o zero: acrescentá-la piora a previsão fora da amostra (sobreajuste), o que é compatível com nenhuma informação real. Leitura: os bans pré-pick refletem hábitos do time e do meta; o ban reativo muda o draft, mas o efeito já aparece nos picks. É uma interpretação, não um teste causal.'])}
<h2>4.3 Draft, força do time e ouro</h2>{t42}
{ul([f'<b>Condicionar no time quase não muda o draft</b> ({nf(to("I(draft ; V | lado)").I_mbit,1)} → {nf(to("I(draft ; V | time)").I_mbit,1)} mbit, ~{pc(1-to("I(draft ; V | time)").I_mbit/to("I(draft ; V | lado)").I_mbit,0)} a menos). A força do time (≈ {nf(to("I(time ; V | lado)").I_mbit,0)} mbit) é ~9 vezes maior que o draft e não depende dele. Em nível de campeão individual o time confunde alguns casos (Twisted Fate, Parte 2); no draft inteiro o efeito agregado é pequeno. Esta é a medida mais confiável de confusão pelo time, porque não sofre com células esparsas.',
     f'<b>Ouro × draft.</b> O ouro no min 15 informa {nf(ouro15_mbit,0)} mbit; o draft sozinho, {nf(to("[min 15] I(draft ; V | lado)").I_mbit,1)}. Dado o ouro, o draft informa mais ({nf(to("[min 15] I(draft ; V | ouro)").I_mbit,1)} mbit): draft e ouro são <b>complementares</b>, não redundantes como XP e CS. Parte disso é automática: quando duas variáveis são independentes, conhecer uma sempre torna a outra mais informativa (A.5). Com efeitos aditivos e independentes, os {nf(to("[min 15] I(draft ; V | lado)").I_mbit,1)} mbit iriam para ~12,6 mbit só por isso; o observado vai além. Uma leitura possível para o excesso (hipótese, não testada): a mesma vantagem de ouro converte em vitória de modo diferente conforme a composição (escalonamento, capacidade de fechar o jogo).'])}
<h2>4.4 Robustez</h2>{t43}
{ul([f'<b>Placebo:</b> com V embaralhado, a informação estimada é negativa (M2 {nf(pl.loc["M2","I_mbit"],1)}; M4 {nf(pl.loc["M4","I_mbit"],1)} mbit): o estimador não tem viés positivo, ao contrário do plug-in da Parte 2.',
     f'<b>Outra era:</b> treinando só no passado, o draft ainda informa ~{nf(ag.loc["M1","I_mbit"],1)} mbit (aditivo), mas posição, confrontos e sinergias <b>não melhoram</b> ({nf(ag.loc["M2","I_mbit"],1)} e {nf(ag.loc["M4","I_mbit"],1)} mbit). A força dos campeões muda com o patch (Parte 2, r = {nf(R2["r_temp"],2)} entre eras), e as interações mudam mais ainda.',
     f'<b>Modelo não linear:</b> o gradient boosting categórico sobre os 10 campeões obtém {nf(gbm.I_mbit,1)} mbit, abaixo do aditivo ({nf(esc("M1","I_mbit"),1)}). Com {gbm.modelo.split("(")[1].rstrip(")")} escolhidas por parada antecipada e 10 variáveis categóricas de ~170 níveis, ele é um modelo fraco aqui: o resultado não prova que não há estrutura além da aditiva, só que este modelo não a encontra.'])}
<h2>4.5 Sinergias e confrontos, par a par</h2>
{box('simples','Por que precisamos de FDR','<p>Se testarmos milhares de pares (ex.: Kai\'Sa + Orianna), alguns vão parecer fortes só por acaso. Para cada par com n ≥ 100, comparamos as vitórias observadas com as esperadas pelo modelo aditivo por posição (M2) fora da amostra: z = (vitórias − esperadas) / erro-padrão. Sem interações reais, os z seguiriam uma normal N(0,1). Aplicamos a correção de <b>Benjamini-Hochberg</b>, que controla a fração de “descobertas falsas” (FDR); o <b>q</b> é o p-valor ajustado.</p>')}
{fig(res_png('parte4_pares.png'), 'Figura 4.2 — Distribuição de z para sinergias, confrontos de lane e ban reativo × pick (adversário e próprio time), contra N(0,1).', '86%')}
{t44}{t45}
{ul([f'<b>Sinergias:</b> dp de z = {nf(PRS.loc["sinergia","dp_z"],2)} e {pc(PRS.loc["sinergia","frac_abs_z_gt2"])} com |z| &gt; 2, o que é o esperado só por acaso; nenhum par sobrevive ao FDR.',
     f'<b>Confrontos de lane:</b> dp de z = {nf(PRS.loc["confronto","dp_z"],2)} e {pc(PRS.loc["confronto","frac_abs_z_gt2"])} com |z| &gt; 2 (esperado 4,6%): há um pequeno excesso de variabilidade, compatível com alguns counters reais, mas nenhum identificável individualmente.',
     '<b>Ban reativo × pick:</b> dp de z = 0,99 nos dois casos, com ~4,5% de |z| &gt; 2 em ~4.400 pares cada: indistinguível de ruído. Contrapicks existem no jogo, mas não deixam rastro detectável no resultado além do que os picks já dizem.'])}
<h2>4.6 Respondendo à pergunta: o todo é maior que a soma das partes?</h2>
{box('aprend','O que aprendemos',f'<p><b>Só um pouco, e a comparação certa é dentro da escada.</b> O modelo aditivo (M1) dá {nf(esc("M1","I_mbit"),1)} mbit; com posição, confrontos e sinergias (M4), os picks chegam a {nf(picks_mbit,1)} mbit, ou seja, {sg(ganho_int,1)} mbit ({pc(ganho_int/esc("M1","I_mbit"),0)}) de informação não aditiva na validação cruzada, que não se sustenta em outra era e em que nenhum par isolado é distinguível do acaso. Comparar {nf(picks_mbit,1)} mbit com 10 × {mb(t2("nenhum (marginal)"),2)} mbit da Parte 2 misturaria estimadores (4.2). Os bans acrescentam ~4 mbit antes de se conhecer o time, quase tudo dos bans pré-pick, e somem ao condicionar no time; a parte reativa (ban × pick) não informa. Medir um campeão por vez não perdia quase nada de informação <b>sobre a força média de cada campeão entre as eras</b>. Os picks continuam sendo ~0,9% de H(V) e ~30 vezes menos informativos que o ouro aos 15 minutos; picks e bans juntos, ~1,3% e ~20 vezes menos.</p>')}
<h3>Limitações específicas da Parte 4</h3>
{ul(['<b>Limite inferior.</b> Os valores são limites inferiores variacionais: um modelo melhor pode achar mais informação. Não é possível derivar um limite de Fano para o draft, pois Fano exige H(V | X) e aqui só temos um teto para ele.',
     f'<b>Força por patch.</b> O modelo usa a força média de cada campeão em 2014–2026, sem termos campeão × patch. Na Parte 2, um campeão isolado informa ~{nf(t2("patch")/t2("nenhum (marginal)"),0)} vezes mais dentro do patch, então {nf(picks_mbit,1)} mbit provavelmente subestimam bastante a informação do draft. Um degrau com campeão × ano (ou × patch) mediria essa parcela.',
     '<b>Efeitos de era.</b> A informação por ano é instável (de −2 a +6 mbit nos testes de origem móvel); o resultado agregado é uma média sobre eras.',
     '<b>Força do time é proxy.</b> Time-ano é estimado com as partidas do próprio time-ano (inclusive futuras): descreve a força, não a prevê antes da partida. A Parte 7 usa uma força medida só com o passado.',
     '<b>Bans.</b> A divisão em fases supõe que ban1–3 e ban4–5 são cronológicos; a ordem intercalada entre os dois lados não está na base.',
     '<b>Testes de pares</b> são dependentes entre si; pares com n &lt; 100 não foram avaliados.'])}
</section>"""


def parte5():
    objs = ["first blood", "1º dragão", "1º arauto", "1ª torre", "1ª torre do meio", "1ª a derrubar 3 torres", "1º barão"]
    t51 = tab(["Grupo", "Variáveis", "Por quê"], [
        ["Iniciais (valem no min 15)", "first blood, 1º dragão, 1º arauto, 1ª torre", "ocorrem quase sempre antes do min 15. O arauto nasce no min 8 até 2023 e no min 14 desde 2024."],
        ["Tardias (muitas vezes depois do min 15)", "1ª torre do meio, 1ª a derrubar 3 torres, 1º barão", "o barão nasce no min 20; as demais costumam ocorrer depois do min 15."],
        ["Pós-fato (consequência)", "totais de dragões, arautos, barões, torres e inibidores no fim do jogo", "só existem no fim: carregam o resultado."]],
        "Tabela 5.1 — Classificação dos objetivos pelo momento em que existem.")
    t52 = tab(["Objetivo", "Grupo", "N", "P(vence | fez primeiro)", "I(obj; V) [mbit]"],
              [[o, M5.loc[o, "grupo"], nf(M5.loc[o, "N"], 0), pc(M5.loc[o, "P_vence_se_faz_primeiro"]), nf(M5.loc[o, "I_corrigida_mbit"], 1)] for o in objs],
              f"Tabela 5.2 — Informação isolada, corrigida por permutação (p &lt; 0,01 em todas). Para comparação, o ouro no min 15 informa ≈ {nf(ref5,0)} mbit na amostra comum.")
    t53 = tab(["Objetivo", "Isolada [mbit]", "Dado o ouro [mbit]", "Fração que resta", "p"],
              [[("<b>" + o + "</b>") if o == "1º dragão" else o, nf(d5.loc[o, "I_marg_mbit"], 1), nf(d5.loc[o, "I_dado_ouro_corr_mbit"], 2),
                pc(d5.loc[o, "fracao_que_resta"]), "&lt; 0,01" if d5.loc[o, "p_valor"] < 0.0100001 else nf(d5.loc[o, "p_valor"], 2)] for o in objs],
              "Tabela 5.3 — I(objetivo; V) e I(objetivo; V | faixa de ouro no min 15, 10 faixas), corrigidas por permutação (100 permutações; p mínimo possível 0,0099).")
    t54 = tab(["Bloco (ordem: ouro primeiro)", "Isolada [bit]", "Incremental [bit]", "IC 95%", "Acumulada"],
              [[b, nf(r.I_isolada, 4), nf(r.ganho_incremental, 4), "± " + nf(r.ic95, 4), nf(r.I_acumulada, 4)] for b, r in c5o.iterrows()],
              "Tabela 5.4 — Ganho incremental I(Xi; V | anteriores), estimado pela queda da log-loss (gradient boosting, 5-fold CV). As três últimas linhas (“tardia”) podem incluir eventos posteriores ao min 15.")
    t55 = tab(["Bloco (sobre o estado aos 15 min)", "Isolada [bit]", "Incremental [bit]", "Acumulada"],
              [[b, "—" if pd.isna(r.I_isolada) else nf(r.I_isolada, 4), nf(r.ganho_incremental, 4), nf(r.I_acumulada, 4)] for b, r in pf5.iterrows()],
              "Tabela 5.5 — Totais do fim do jogo acrescentados ao estado do min 15 (ouro, XP, CS, abates, objetivos iniciais).")
    soma_ini = c5b.loc["+ 1ª torre", "I_acumulada"]
    ouro_dep = c5b.loc["+ dif. de ouro", "ganho_incremental"]
    return f"""<section class="parte" id="p5"><h1>Parte 5 — Os objetivos do jogo</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> first blood, torres, dragões, arauto e barão são importantes para prever o vencedor?</p><p>Atenção ao <b>quando</b>. A base não traz o minuto de cada objetivo, só quem o fez <i>primeiro</i> (variáveis first*) e os totais do fim do jogo. Por isso separamos as variáveis conforme o momento em que existem.</p>')}
<h2>5.1 Quando cada objetivo existe</h2>{t51}
<p>Amostra comum: 89.501 partidas que chegaram ao min 15 e têm todos os first*; H(V) = 0,997 bit.</p>
<h2>5.2 Isolados, os objetivos parecem muito informativos</h2>{t52}
<p>Lendo só esta tabela, o 1º barão ({nf(M5.loc['1º barão','I_corrigida_mbit'],0)} mbit) pareceria mais informativo que o próprio
ouro ({nf(ref5,0)} mbit). Mas o barão só nasce no min 20: ele já contém o futuro. O teste justo é perguntar o que cada objetivo
acrescenta <i>depois de conhecer o ouro</i>.</p>
<h2>5.3 Depois do ouro, quase tudo some</h2>{t53}
{fig(res_png('parte5_painel1.png'), 'Figura 5.1 — Objetivo isolado vs. depois de conhecer o ouro (azul: iniciais; laranja: tardias).', '70%')}
{ul([f'<b>First blood e 1º arauto não acrescentam nada mensurável depois do ouro</b>, e a 1ª torre acrescenta ≈ {nf(d5.loc["1ª torre","I_dado_ouro_corr_mbit"],1)} mbit. A informação deles já estava no ouro: um objetivo inicial é, em boa parte, o ouro chegando antes.',
     f'<b>O 1º dragão é a exceção entre os iniciais:</b> {pc(d5.loc["1º dragão","fracao_que_resta"],0)} da sua informação sobrevive ao ouro (≈ {nf(d5.loc["1º dragão","I_dado_ouro_corr_mbit"],0)} mbit), o mesmo resultado da Parte 3.',
     f'<b>Por era:</b> de 2014 a 2023 o arauto acrescenta {nf(era5("2014-2023","1º arauto"),1)} mbit depois do ouro; desde 2024, {nf(era5("2024+","1º arauto"),1)} mbit, e a 1ª torre passa de {nf(era5("2014-2023","1ª torre"),2)} para {nf(era5("2024+","1ª torre"),1)} mbit. É compatível com o arauto nascer no min 14 (parte do efeito acontece depois do min 15) e com a era das placas de torre, mas sem horário as duas explicações não se separam.',
     f'<b>As tardias “sobrevivem” ao ouro porque contêm o futuro:</b> o barão (min 20+) mostra {nf(d5.loc["1º barão","I_dado_ouro_corr_mbit"],0)} mbit dado o ouro aos 15 min. Não valem como preditoras do min 15.'])}
<h2>5.4 Regra da cadeia no minuto 15</h2>{t54}
{fig(res_png('parte5_painel2.png'), 'Figura 5.2 — Regra da cadeia no min 15, ouro primeiro (vermelho: pode ocorrer depois do min 15).', '66%')}
<p><b>Invertendo a ordem</b>, first blood, 1º dragão, 1º arauto e 1ª torre juntos dão {nf(soma_ini,3)} bit, e o ouro ainda
acrescenta {sg(ouro_dep,3)} bit depois deles. Os objetivos iniciais são em grande parte um resumo parcial e ruidoso do ouro, e o
contrário não vale.</p>
<h3>Placas de torre e vermes do vazio</h3>
<p>As placas das torres externas só caem até o min ≈ 14 (no máximo 15 por time) e funcionam quase como uma variável do min 15.
Na amostra de {pl5_n.replace(',', '.')}, isoladas elas informam {nf(pl5.I_isolada,3)} bit, mas
depois do ouro, XP, CS e abates acrescentam {sg(pl5.ganho_incremental,4)} ± {nf(pl5.ic95,4)} bit: nada. O ouro já contém a
informação das placas (cada placa dá ouro a quem a derruba). <b>Atenção:</b> em 2026 a coluna de placas do Oracle's Elixir muda
de escala (média de ~25 por time e máximo de 45, impossível para placas até o min 14); com 2026 misturado na amostra, as placas
pareciam acrescentar 0,082 bit depois do ouro, um artefato da base. Os vermes do vazio (amostra {gr5_n.replace(',', '.').replace(' (', ', ').rstrip(')')}) também não
acrescentam nada (≈ 0 bit, IC ± {nf(gr5.ic95,4)}).</p>
<h2>5.5 Os totais do fim do jogo são consequência, não preditor</h2>{t55}
{fig(res_png('parte5_painel3.png'), 'Figura 5.3 — Totais do fim do jogo: o objetivo vira consequência.', '62%')}
{box('vaz','Cuidado: vazamento',f'<p>A diferença de torres derrubadas, sozinha, carrega {nf(PF5[PF5.bloco.str.contains("torres")].I_isolada.iloc[0],2)} bit dos 0,997 de H(V). Não é mérito preditivo: para vencer é preciso derrubar as torres até o nexus, então o total de torres é parte do próprio resultado.</p>')}
{box('aprend','O que aprendemos',f'<p>Depois do ouro, first blood, 1ª torre e 1º arauto acrescentam ≈ 1 mbit ou menos; o 1º dragão (≈ {nf(d5.loc["1º dragão","I_dado_ouro_corr_mbit"],0)} mbit) é a única exceção entre os iniciais. As variáveis que parecem muito informativas (barão, torres do meio) ou contêm o futuro ou são consequência do ouro.</p>')}
</section>"""


def parte6():
    lanes = ["top", "jng", "mid", "bot", "sup"]
    nome = {"top": "top", "jng": "selva (jng)", "mid": "meio (mid)", "bot": "atirador (bot)", "sup": "suporte (sup)"}
    t61 = tab(["Lane", "Ouro min 15", "XP", "CS", "KDA 15", "KDA final [consequência]"],
              [[nome[l], nf(I6.loc[l, "ouro"], 4), nf(I6.loc[l, "XP"], 4), nf(I6.loc[l, "CS"], 4), nf(I6.loc[l, "KDA 15"], 4),
                nf(I6.loc[l, "[consequência] KDA final"], 4)] for l in lanes],
              "Tabela 6.1 — Informação isolada I(variável da lane; V), em bits (limite inferior variacional, regressão logística, 5-fold CV). A estimativa por discretização do ouro (20 faixas, corrigida por permutação) concorda em até 0,005 bit.")
    so, sk = s6("ouro"), s6("ouro+XP+CS+KDA")
    ordem = so.I_dado_outras.sort_values(ascending=False).index
    t62 = tab(["Lane (ordem de importância)", "Ouro, sozinha", "Ouro, dado as outras 4", "Parcela que resta", "Ouro+XP+CS+KDA, dado as outras 4"],
              [[nome[l], nf(so.loc[l, "I_isolada"], 4), f"{nf(so.loc[l,'I_dado_outras'],4)} ± {nf(so.loc[l,'ic95'],4)}",
                pc(so.loc[l, "I_dado_outras"] / so.loc[l, "I_isolada"]), f"{nf(sk.loc[l,'I_dado_outras'],4)} ± {nf(sk.loc[l,'ic95'],4)}"] for l in ordem],
              f"Tabela 6.2 — I(lane; V | outras 4 lanes) = perda de informação ao retirar a lane do modelo completo (as cinco lanes juntas informam {nf(so.I_completo.iloc[0],3)} bit só com ouro e {nf(sk.I_completo.iloc[0],3)} bit com ouro, XP, CS e KDA). IC 95% do pareamento por partida.")
    t63 = tab(["Lane", "min 10", "min 15", "min 20", "min 25"],
              [[nome.get(l, l) if l in nome else f"<b>{l}</b>"] + [nf(tp6.loc[l, m], 4) for m in (10, 15, 20, 25)]
               for l in ["bot", "jng", "mid", "sup", "top", "5 lanes juntas"]],
              "Tabela 6.3 — I(ouro da lane no min t; V), amostra comum de jogos que chegaram ao min 25, em bits.")
    t64 = tab(["Lane", "Desvio-padrão do ouro (min 15)", "Razão de chances por +1 dp de ouro"],
              [[nome[l], nf(CO6.loc[l, "dp_ouro"], 0), nf(CO6.loc[l, "odds_ratio_por_dp"], 2)]
               for l in CO6.odds_ratio_por_dp.sort_values(ascending=False).index],
              "Tabela 6.4 — Regressão logística com as cinco lanes juntas (ouro, min 15).")
    soma_cond = so.I_dado_outras.sum()
    orb = CO6.loc["bot", "odds_ratio_por_dp"]
    return f"""<section class="parte" id="p6"><h1>Parte 6 — Qual lane é a mais decisiva?</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> cada partida tem cinco duelos de lane (top, selva, meio, atirador e suporte). Qual deles, quando está à frente, melhor indica a vitória?</p><p>Para cada lane usamos a diferença azul menos vermelho de ouro, XP e CS e o KDA do jogador. Medimos a informação de cada lane sozinha e depois <i>o que ela acrescenta quando já conhecemos as outras quatro</i>.</p>')}
<p>Amostra: 92.747 partidas com as cinco lanes completas até o min 15. KDA = (abates + assistências) / max(mortes, 1).</p>
{box('vaz','Cuidado com o KDA','<p>O KDA <i>final</i> é calculado no fim da partida e carrega o resultado (quem vence abate mais e morre menos): isoladamente ele informa 0,60 a 0,67 bit por lane, mais que o ouro. É vazamento e só aparece como referência. As análises usam o KDA aos 15 min.</p>')}
<h2>6.1 Cada lane sozinha</h2>{t61}
{fig(res_png('parte6_painel1.png'), 'Figura 6.1 — Informação isolada de cada lane (min 15). As barras cinzas, o KDA final, são consequência do resultado.', '62%')}
{ul([f'Pelo ouro, <b>meio ({nf(I6.loc["mid","ouro"],3)}), atirador ({nf(I6.loc["bot","ouro"],3)}) e selva ({nf(I6.loc["jng","ouro"],3)})</b> ficam praticamente empatados; suporte ({nf(I6.loc["sup","ouro"],3)}) e top ({nf(I6.loc["top","ouro"],3)}) vêm depois.',
     f'Pelo KDA aos 15 min, a <b>selva é a mais informativa ({nf(I6.loc["jng","KDA 15"],3)})</b>: uma leitura possível (hipótese) é que o KDA do caçador resume a participação do time nas lutas, enquanto o do top é o menos informativo ({nf(I6.loc["top","KDA 15"],3)}).',
     f'O <b>CS do suporte não informa nada</b> ({nf(I6.loc["sup","CS"],4)} bit): provavelmente porque o suporte pouco farma (hipótese).'])}
<h2>6.2 Depois de conhecer as outras quatro lanes</h2>
{box('simples','Por que condicionar nas outras lanes',f'<p>Dizer que a selva informa {nf(so.loc["jng","I_isolada"],3)} bit sozinha não significa que ela acrescenta isso. Se a selva está à frente, é provável que o top e o meio também estejam. Retiramos a lane do modelo completo e medimos o que se perde: é a informação <i>própria</i> da lane, I(lane; V | outras 4). Como em A.9, é a diferença de dois limites inferiores, com IC pelo pareamento por partida.</p>')}
{t62}
{fig(res_png('parte6_painel2.png'), 'Figura 6.2 — Isolada vs. dado as outras quatro lanes (ouro, min 15).', '62%')}
{ul([f'<b>O ranking muda quando se controla pelas outras lanes.</b> Selva e meio parecem os mais informativos sozinhos porque seu ouro acompanha o do time inteiro. Depois de conhecer as outras quatro, o atirador lidera ({nf(so.loc["bot","I_dado_outras"],3)} bit), seguido de meio ({nf(so.loc["mid","I_dado_outras"],3)}) e top ({nf(so.loc["top","I_dado_outras"],3)}); a selva cai para {nf(so.loc["jng","I_dado_outras"],3)} e o suporte para {nf(so.loc["sup","I_dado_outras"],3)}.',
     f'<b>Metade da informação é compartilhada.</b> A soma das parcelas condicionais (≈ {nf(soma_cond,2)} bit) é metade do total ({nf(so.I_completo.iloc[0],3)}). As lanes vencem e perdem juntas: é o efeito bola de neve do time.',
     f'Com XP, CS e KDA no modelo, atirador, meio e top ficam estatisticamente próximos ({nf(sk.loc["bot","I_dado_outras"],3)}; {nf(sk.loc["mid","I_dado_outras"],3)}; {nf(sk.loc["top","I_dado_outras"],3)}); selva ({nf(sk.loc["jng","I_dado_outras"],3)}) e suporte ({nf(sk.loc["sup","I_dado_outras"],3)}) ficam claramente abaixo.',
     f'O <b>top retém {pc(so.loc["top","I_dado_outras"]/so.loc["top","I_isolada"],0)}</b> da sua informação quando se conhecem as outras lanes (leitura possível: é a lane mais isolada do mapa), enquanto o suporte retém só {pc(so.loc["sup","I_dado_outras"]/so.loc["sup","I_isolada"],0)} e a selva {pc(so.loc["jng","I_dado_outras"]/so.loc["jng","I_isolada"],0)}.'])}
<h2>6.3 Quando cada lane passa a importar</h2>{t63}
{fig(res_png('parte6_painel3.png'), 'Figura 6.3 — Informação do ouro de cada lane no tempo.', '62%')}
{ul([f'Aos 10 min, selva, meio e atirador têm o mesmo peso (≈ {nf(tp6.loc["mid",10],3)} bit); top e suporte informam menos.',
     f'Aos 25 min, o <b>top continua o menos informativo</b> ({nf(tp6.loc["top",25],3)}), enquanto o suporte sobe a ponto de empatar com as demais ({nf(tp6.loc["sup",25],3)}): uma leitura possível (hipótese) é que, no fim do jogo, o ouro do suporte passa a refletir o do time.'])}
<h2>6.4 Em termos de chances</h2>
{box('simples','Em palavras simples',f'<p><b>Razão de chances (odds ratio).</b> “Chance” (odds) é p / (1 − p): 50% equivale a chance 1; 75% a chance 3. Uma razão de chances de {nf(orb,2)} significa que a chance de vitória é multiplicada por {nf(orb,2)}. <b>Ilustração:</b> partindo de 50% e mantendo as outras lanes iguais, +1 desvio-padrão no atirador leva a probabilidade a {pc(orb/(1+orb),0)}; no suporte, a {pc(CO6.loc["sup","odds_ratio_por_dp"]/(1+CO6.loc["sup","odds_ratio_por_dp"]),0)}.</p>')}
{t64}
{box('aprend','O que aprendemos','<p>Em termos de informação, o <b>atirador é a lane mais decisiva</b> (maior informação própria, maior razão de chances), seguido de perto por meio e top; selva e suporte informam menos por conta própria, embora o KDA da selva seja o mais informativo isoladamente. As diferenças entre as três primeiras são pequenas (≈ 0,01 bit).</p>')}
<h3>Limitações específicas da Parte 6</h3>
{ul(['<b>Informação não é causalidade.</b> “Decisiva” significa que estar à frente na lane indica a vitória, não que a lane cause a vitória. O ouro do atirador, por exemplo, também é consequência de ter sido protegido pela selva e pelo suporte.','<b>Lanes correlacionadas.</b> A informação condicional depende de quais outras variáveis estão no modelo; metade da informação é compartilhada.','<b>Modelo logístico</b> (linear no ouro): um modelo melhor pode achar mais informação no total. As parcelas condicionais são diferenças entre dois modelos e não têm essa garantia; a concordância com a estimativa discretizada sugere que o erro é pequeno.','<b>Posições da base.</b> O pareamento usa o rótulo de posição do Oracle\'s Elixir; trocas de lane ficam com o rótulo da base.','<b>KDA com mortes mínimas de 1.</b> A razão é assimétrica e muito variável; o resultado com abates + assistências − mortes (também testado) é parecido.'])}
</section>"""


def parte7():
    linhas = []
    for _, r in S7.iterrows():
        pm = S7m.loc[(r.ano, r.fase)] if (r.ano, r.fase) in S7m.index else None
        linhas.append([str(r.ano), r.fase, r.rival, r.placar, "—" if pm is None else pc(pm.p_serie_T1, 0), pc(r.p_serie_T1, 0),
                       "venceu" if r.T1_venceu else "<b>perdeu</b>", nf(r.surpresa_bits, 2), nf(r.entropia_bits, 2)])
    t71 = tab(["Ano", "Fase", "Rival", "Placar (T1)", "P(T1 vence) antes do Mundial", "P(T1 vence) antes da série", "T1",
               "Surpresa [bits]", "Surpresa média esperada H(P) [bits]"], linhas,
              "Tabela 7.1 — Séries eliminatórias (melhor de cinco) da T1. P vem do Elo contínuo com jogos anteriores ao corte. Surpresa = −log₂ P(resultado observado), usando P antes da série.", "pequena")
    lc = [[str(a), nf(r.series, 0), nf(r.surpresa_total_bits, 2), nf(r.entropia_total_bits, 2), f"1 em {nf(1/r.p_do_que_aconteceu,0)}",
           "campeã" if r.campea else "vice"] for a, r in CA7.iterrows()]
    t72 = tab(["Ano", "Séries no Mundial", "Surpresa da campanha [bits]", "Surpresa média esperada [bits]",
               "Chance do que aconteceu", "Resultado"], lc,
              "Tabela 7.2 — Campanhas da T1 no Mundial (play-in e mata-mata; a seletiva de 2024 fica fora). Surpresas de séries independentes se somam, porque probabilidades se multiplicam.")
    rk = [[str(a), f"{nf(r.posicao,0)}º de {nf(r.participantes,0)}", nf(r.elo_pre_mundial, 0),
           ", ".join(RK7[(RK7.ano == a) & (RK7.posicao <= 3)].time)] for a, r in rk_t1.iterrows()]
    t73 = tab(["Ano", "Posição da T1 no Elo (entre os participantes)", "Elo da T1", "Três primeiros"], rk,
              "Tabela 7.3 — Força antes do Mundial (Elo contínuo no dia anterior ao primeiro jogo entre ligas diferentes).")
    vl = [[str(r.ano), r.time, nf(r.d_medio, 2), nf(r.acima_bonferroni, 0), nf(r.abaixo_de_zero, 0)] for _, r in VR7.iterrows()]
    t74 = tab(["Ano", "Time", "Efeito médio dos 5 titulares (dp)", "Acima da linha de Bonferroni", "Abaixo da média da posição"], vl,
              "Tabela 7.4 — Resumo do volcano plot: KDA de cada titular na temporada regular da LCK contra os demais jogadores da mesma posição.")
    f22, s24, f24, f25, sj23 = s7(2022, "final"), s7(2024, "semi"), s7(2024, "final"), s7(2025, "final"), s7(2023, "semi")
    m22 = S7m.loc[(2022, "final")]
    m24s = S7m.loc[(2024, "semi")]
    c22, c23, c24, c25 = CA7.loc[2022], CA7.loc[2023], CA7.loc[2024], CA7.loc[2025]
    v22, v24, g24 = vr(2022, "T1"), vr(2024, "T1"), vr(2024, "Gen.G")
    return f"""<section class="parte" id="p7"><h1>Parte 7 — A T1 em bits: favorito, azarão e surpresa</h1>
{box('simples','Em poucas palavras','<p><b>Pergunta:</b> as campanhas da T1 no Mundial foram surpreendentes? Quanto, em bits?</p><p>Entre 2022 e 2025 o mesmo time viveu os dois papéis: em 2022 chegou à final como favorita e perdeu para a DRX, quarta classificada da Coreia, que veio desde a fase de entrada; em 2024 entrou como quarta da Coreia, depois de vencer a KT por 3–2 na seletiva, eliminou a favorita Gen.G na semifinal (3–1) e virou a final contra a Bilibili Gaming (3–2), em Londres. Em 2025 entrou de novo como quarta da Coreia, pelo play-in, e venceu a KT na final (3–2). As Partes 1 a 6 mediram o quanto cada informação reduz a incerteza <i>em média</i>; aqui medimos a surpresa de resultados <i>específicos</i>.</p>')}
<h2>7.1 Medida: a surpresa de um resultado</h2>
<p>Se um resultado tinha probabilidade P de acontecer, sua <b>surpresa</b> (autoinformação) é −log<sub>2</sub> P bits. Uma série
que o time tinha 50% de chance de vencer vale 1 bit quando ele vence; uma com 25% vale 2 bits; uma com 12,5%, 3 bits. A
entropia H(P) da série é a surpresa <i>média</i> que se esperava antes do jogo: quando a surpresa observada fica bem acima
de H(P), houve zebra.</p>
{box('exemplo','Exemplo numérico',f'<p>Antes da final de 2022, a T1 tinha {pc(f22.p_serie_T1,0)} de chance de vencer a série. Ela perdeu, um resultado de chance {pc(1-f22.p_serie_T1,0)}: surpresa = −log<sub>2</sub>({nf(1-f22.p_serie_T1,2)}) = <b>{nf(f22.surpresa_bits,2)} bits</b>. A surpresa média esperada era H({nf(f22.p_serie_T1,2)}) = {nf(f22.entropia_bits,2)} bit. Surpresas de eventos independentes se somam: vencer duas séries de 50% vale 1 + 1 = 2 bits, que é −log<sub>2</sub>(0,5 × 0,5).</p>')}
<h2>7.2 Como a força foi medida (sem olhar o futuro)</h2>
{ul(['<b>Elo cronológico.</b> Cada time tem uma nota; depois de cada jogo, o vencedor ganha pontos proporcionais à surpresa do resultado (K = 20). A chance de vencer um jogo é 1 / (1 + 10<sup>(Elo_rival − Elo_time)/400</sup>). Só entram jogos <b>anteriores</b> ao corte, ao contrário do time-ano das Partes 2 e 4, que usa o ano inteiro.',
     '<b>Contínuo desde 2014, com as ligas conectadas.</b> Na virada do ano, cada time regride 25% em direção à média da sua liga, e um time novo entra com essa média. Assim a força das ligas, aprendida nos jogos de MSI e Mundial, se acumula entre os anos. Um Elo que recomeça em 1500 todo ano coloca times que dominam ligas fracas acima dos coreanos e chineses (em 2022, a DetonatioN FocusMe, do Japão, apareceria em 2º lugar); o resultado com esse Elo anual está no CSV, como comparação.',
     '<b>Chance da série.</b> Com p = chance de vencer um jogo e jogos independentes, a chance de vencer uma melhor de cinco é P = p³ · [1 + 3(1 − p) + 6(1 − p)²].',
     '<b>Dois cortes:</b> antes do Mundial (dia anterior ao primeiro jogo entre ligas diferentes) e antes de cada série (inclui os jogos do próprio Mundial até ali). No Oracle\'s Elixir a liga “WLDs” também traz as seletivas regionais, por isso o início do Mundial é definido pelo primeiro jogo entre ligas diferentes.'])}
<h2>7.3 Favorito ou azarão?</h2>{t73}
<p>Pelo Elo, a T1 era a {nf(rk_t1.loc[2022,'posicao'],0)}ª força entre os participantes em 2022, caiu para {nf(rk_t1.loc[2024,'posicao'],0)}ª em
2024 e voltou ao {nf(rk_t1.loc[2025,'posicao'],0)}º lugar em 2025. A Gen.G foi a mais forte nos quatro anos.</p>
<h2>7.4 A surpresa de cada série</h2>{t71}
{fig(res_png('parte7_surpresa_series.png'), 'Figura 7.1 — Surpresa de cada série eliminatória da T1 (barras) e a surpresa média esperada antes da série, H(P) (traços). Azul: a T1 venceu; vermelho: perdeu.')}
{ul([f'<b>2022, a favorita que perdeu.</b> Antes do Mundial a T1 tinha {pc(m22.p_serie_T1,0)} de chance contra a DRX numa final; na véspera da final, {pc(f22.p_serie_T1,0)}. A derrota custou {nf(f22.surpresa_bits,2)} bits (ou {nf(m22.surpresa_bits,2)} bits pela força de antes do Mundial), contra {nf(f22.entropia_bits,2)} esperados.',
     f'<b>2024, o azarão que venceu.</b> A semifinal contra a Gen.G é o resultado mais surpreendente de todos: {pc(s24.p_serie_T1,0)} de chance antes da série ({pc(m24s.p_serie_T1,0)} antes do Mundial), {nf(s24.surpresa_bits,2)} bits ({nf(m24s.surpresa_bits,2)} pela força de antes do Mundial). A final contra a BLG ({pc(f24.p_serie_T1,0)}) somou mais {nf(f24.surpresa_bits,2)} bits.',
     f'<b>2023 também teve zebra.</b> A semifinal contra a JD Gaming ({pc(sj23.p_serie_T1,0)}) valeu {nf(sj23.surpresa_bits,2)} bits; a campanha inteira acumulou {nf(c23.surpresa_total_bits,1)} bits, tanto quanto a de 2022 ({nf(c22.surpresa_total_bits,1)}).',
     f'<b>2025, azarão só na classificação.</b> A T1 entrou como quarta da Coreia, mas pelo Elo era a {nf(rk_t1.loc[2025,"posicao"],0)}ª força e foi favorita ou quase empatada em todas as séries (final contra a KT: {pc(f25.p_serie_T1,0)}).'])}
<h2>7.5 A surpresa das campanhas</h2>{t72}
{ul([f'<b>2024 foi a campanha mais improvável:</b> {nf(c24.surpresa_total_bits,1)} bits, uma chance de 1 em {nf(1/c24.p_do_que_aconteceu,0)}, contra {nf(c24.entropia_total_bits,1)} bits de surpresa média esperada.',
     f'<b>2025 foi menos surpreendente do que o esperado de séries equilibradas</b> ({nf(c25.surpresa_total_bits,1)} bits observados, {nf(c25.entropia_total_bits,1)} esperados): ganhar quatro séries de ~55 a 65% tinha chance de 1 em {nf(1/c25.p_do_que_aconteceu,0)}. O rótulo de “quarta da Coreia” vinha da liga nacional, não da força medida.',
     f'O mesmo quinteto (Zeus, Oner, Faker, Gumayusi e Keria) esteve nos dois extremos: perdeu como favorito em 2022 ({nf(c22.surpresa_total_bits,1)} bits na campanha) e venceu como azarão em 2024.'])}
<h2>7.6 O desempenho na temporada regular</h2>
{fig(res_png('parte7_volcano_kda.png'), 'Figura 7.2 — Volcano plot: para cada titular da LCK, o eixo x é o KDA médio menos o dos demais jogadores da mesma posição (em desvios-padrão) e o eixo y é −log₁₀ do p-valor (Mann-Whitney). Acima da linha tracejada, a diferença sobrevive à correção de Bonferroni.')}
{t74}
{ul([f'Em 2022, ano em que era favorita, a T1 teve o melhor desempenho entre os anos analisados: efeito médio de {nf(v22.d_medio,2)} dp e {nf(v22.acima_bonferroni,0)} titulares acima da linha de Bonferroni (Oner e Keria).',
     f'Em 2024 os titulares da T1 ficaram <b>acima</b> da média da posição, mas pouco ({nf(v24.d_medio,2)} dp; só {nf(v24.acima_bonferroni,0)} acima da linha), enquanto os cinco da Gen.G estavam todos acima da linha ({nf(g24.d_medio,2)} dp). A T1 não ficou abaixo da média: ficou abaixo da Gen.G, e é isso que a fazia azarão.',
     'O KDA final é em boa parte consequência das vitórias (Parte 6), então o volcano descreve a temporada; não é uma medida independente de força.'])}
{box('aprend','O que aprendemos',f'<p>A Teoria da Informação dá uma régua para a narrativa: a derrota de 2022 custou {nf(f22.surpresa_bits,1)} bits, a semifinal de 2024 contra a Gen.G valeu {nf(s24.surpresa_bits,1)} bits, e a campanha de 2024 inteira, {nf(c24.surpresa_total_bits,1)} bits (1 em {nf(1/c24.p_do_que_aconteceu,0)}). Com força medida só com o passado, 2024 foi o título improvável, e 2025, apesar da vaga de quarta colocada, foi o de um time forte.</p>')}
<h3>Limitações específicas da Parte 7</h3>
{ul(['<b>Elo é um modelo simples.</b> Não usa o draft nem o ouro, só resultados; a chance da série supõe jogos independentes, e K = 20 e a regressão de 25% ao ano são escolhas fixas. Com o Elo que recomeça todo ano (comparação no CSV), as conclusões sobre 2022, 2024 e 2025 se mantêm.',
     '<b>Ligas diferentes.</b> A comparação entre regiões depende dos poucos jogos internacionais; séries contra times chineses são as mais incertas.',
     '<b>Um time, quatro anos.</b> É um estudo de caso para ilustrar a medida, não uma estatística sobre azarões em geral.',
     '<b>Volcano plot</b> é descritivo: o KDA final reflete as vitórias, e os testes por jogador não são independentes.'])}
</section>"""


def sintese():
    S = tab(["Pergunta", "Resposta (em bits)", "Onde"], [
        ["Quanto o ouro informa ao longo do tempo?", f"{nf(P1.loc[10,'I_plugin'],2)} → {nf(P1.loc[15,'I_plugin'],2)} → {nf(P1.loc[20,'I_plugin'],2)} → {nf(P1.loc[25,'I_plugin'],2)} bit (min 10, 15, 20, 25). Aos 25 min ainda restam {nf(P1.loc[25,'H_V_dado_X'],2)} bit.", "Parte 1"],
        ["Quanto um campeão informa?", f"≈ {mb(t2('nenhum (marginal)'),2)} mbit. Os 10 picks, ≈ {nf(picks_mbit,1)} mbit (com bans, {nf(bans_mbit,1)}), quase tudo aditivo.", "Partes 2 e 4"],
        ["Menor erro possível no min 15?", f"≈ 25% na prática (46,8% de um palpite ingênuo); piso de Fano ≤ {pc(teto4)}. XP, CS e abates acrescentam &lt; 0,01 bit depois do ouro.", "Parte 3"],
        ["Os objetivos iniciais importam?", f"Isolados informam até ~0,10 bit (1ª torre), mas depois do ouro sobra ≈ 1 mbit ou menos, exceto o 1º dragão (≈ {nf(d5.loc['1º dragão','I_dado_ouro_corr_mbit'],0)} mbit).", "Parte 5"],
        ["Qual lane é mais decisiva?", f"Isolada: meio ≈ atirador ≈ selva. Dado as outras 4 lanes: atirador ({nf(s6('ouro').loc['bot','I_dado_outras'],3)} bit), meio, top; selva e suporte bem abaixo.", "Parte 6"],
        ["Quão surpreendente foi a T1?", f"Final de 2022 perdida como favorita: {nf(s7(2022,'final').surpresa_bits,1)} bits. Campanha de 2024: {nf(CA7.loc[2024,'surpresa_total_bits'],1)} bits (1 em {nf(1/CA7.loc[2024,'p_do_que_aconteceu'],0)}).", "Parte 7"]],
        "Tabela S.1 — As respostas, em uma tabela.")
    return f"""<section class="parte" id="sintese"><h1>Síntese e conclusões</h1>
<p>Em uma frase: <b>o resultado de uma partida profissional é decidido por quem está na frente em ouro, e o ouro já resume quase
tudo o que os objetivos iniciais, o KDA e os campeões mostram.</b> O que sobra de informação própria está em poucos lugares: o
1º dragão, o tempo (dos 10 aos 25 minutos) e, entre as lanes, sobretudo o atirador e o meio.</p>
{fig(os.path.join(FIG,'S1_regua.png'), 'Figura S.1 — Régua de tudo que foi medido (informação isolada, em milibits, escala logarítmica). Valores de amostras diferentes (Tabela B.1); servem para comparar ordens de grandeza. Vermelho: consequência do resultado; laranja: pode ocorrer depois do min 15.', '86%')}
{S}
<h2>Ideias para levar</h2>
{ul(['<b>O ouro resume quase tudo.</b> XP, CS, abates e objetivos iniciais são, em grande parte, o ouro chegando por outro caminho: isolados informam muito, depois do ouro quase nada (redundância).',
     '<b>Cada variável tem um momento.</b> Só vale como preditor o que se sabe antes da previsão. Totais do fim do jogo e objetivos tardios inflam a informação (vazamento).',
     '<b>Viés de estimativa pode ser maior que o sinal.</b> Sem correção por permutação, o campeão parece informar 300 mbit em vez de ~4. A correção também precisa respeitar a estrutura dos dados (um vencedor e um perdedor por partida).',
     '<b>Informação pequena não é efeito pequeno.</b> Um campeão, sozinho, diz pouco sobre uma partida; isso não mede o efeito causal dele no jogo, nem o valor do draft para uma estratégia.',
     '<b>Média não é caso.</b> O jogo profissional é muito incerto (aos 15 minutos o melhor palpite ainda erra ~25%), e é por isso que resultados específicos podem carregar muitos bits de surpresa, como a T1 em 2022 e 2024.'])}
<h2>Conclusões por parte</h2>
{ul(['<b>Quando a incerteza cai.</b> O ouro resolve 13%, 22%, 32% e 47% da incerteza nos minutos 10, 15, 20 e 25.',
     f'<b>Draft.</b> Os campeões diferem em força real (~{nf(tau_a*100,1)} a {nf(tau_b*100,1)} p.p.), mas a informação que carregam sobre a vitória é de ~{mb(t2("nenhum (marginal)"),2)} milibit; os 10 picks, ~{nf(picks_mbit,1)} milibits, quase só aditivos (limite inferior que não usa a força por patch). Quase tudo que um estimador ingênuo mostra nos contextos finos é viés.',
     '<b>Contexto.</b> Patch e liga não confundem a força do campeão (a vitória é balanceada dentro deles; dentro do patch o campeão informa mais); a força do time confunde campeões específicos (explica metade da força aparente de Twisted Fate), mas não reduz a informação agregada do campeão, e no draft inteiro tira só ~10%.',
     '<b>Previsão.</b> No minuto 15 o erro mínimo prático é ~25%, com o ouro respondendo por quase tudo; o 1º dragão é o único complemento relevante.',
     '<b>Objetivos.</b> Depois do ouro, só o 1º dragão acrescenta algo mensurável entre os iniciais (≈ 12 mbit).',
     '<b>Lanes.</b> O atirador é a lane com maior informação própria; selva e suporte, a menor.',
     '<b>T1.</b> Com a força medida só com o passado, a final de 2022 e a campanha de 2024 são as grandes surpresas; em 2025 a T1 era azarão na classificação, não na força.',
     '<b>Metodologia.</b> Correção de viés (permutação), cuidado com viés de seleção e com vazamento temporal são tão importantes quanto o cálculo da informação em si.'])}
</section>"""


def limitacoes():
    return f"""<section class="parte" id="lim"><h1>Limitações gerais</h1>
{ul(['<b>Seleção temporal.</b> A Parte 1 usa jogos que chegaram ao minuto 25; a informação “real” nos primeiros minutos é maior (Tabela 1.1). Só há quatro pontos no tempo (10, 15, 20, 25), então o “minuto decidido” é interpolado.',
     f'<b>Ligas sem dados de tempo.</b> {pc(EXC_TOT.sem_dados_min10/EXC_TOT.partidas,0)} das partidas (principalmente LDL e LPL) não têm as variáveis do minuto 10 em diante e ficam fora das Partes 1, 3, 5 e 6.',
     '<b>Dependência entre unidades.</b> Na Parte 2 os cinco picks de um time compartilham V, e os dois lados de uma partida têm V opostos. O teste de permutação embaralha unidades (partida, lado) inteiras e, quando o contexto não separa os lados, a partida inteira.',
     '<b>Contextos esparsos.</b> Com milhares de estratos, a informação corrigida da Parte 2 tende a ficar abaixo da verdade (células de 1 ou 2 picks não mostram o sinal).',
     '<b>Confundidores não modelados.</b> Time-ano é um proxy parcial: não separa a habilidade individual do jogador nem a posição, usa o ano inteiro e inclui o efeito dos próprios campeões.',
     '<b>Estimação do limite de Fano.</b> Depende da discretização e da correção de viés; o que se garante é um teto para o limite (h⁻¹ da log-loss), não o seu valor exato.',
     '<b>Valor “não nulo” ≠ relevante.</b> Muitos efeitos têm p &lt; 0,01 com N grande, mas são pequenos em bits. A relevância prática deve ser lida pela magnitude.',
     '<b>Posição não condicionada</b> na Parte 2: a força de um campeão pode estar numa só posição.',
     '<b>Objetivos sem horário.</b> “Primeiro” não diz quão cedo nem quanto ouro rendeu; o arauto e as placas de torre mudam de significado entre eras, e a coluna de placas muda de escala em 2026.',
     '<b>Limite inferior variacional.</b> Um modelo melhor pode achar mais informação no total; diferenças entre dois modelos (ganhos incrementais) não têm direção garantida.',
     '<b>Observacional e profissional.</b> Os resultados descrevem associações em jogos profissionais; não são efeitos causais e não valem automaticamente para partidas ranqueadas amadoras.'])}
</section>"""


def repro():
    t = tab(["Arquivo", "Conteúdo"], [
        ["it_utils.py", "entropia, I(X;V), I(X;V | Z), Miller-Madow, permutação, bootstrap, Fano"],
        ["data.py", "leitura dos CSVs; tabelas por partida (time azul), por pick, drafts, objetivos e lanes"],
        ["parte1_curva.py", "Parte 1: I(ouro em t ; V), t = 10, 15, 20, 25; exclusões por liga"],
        ["parte2_draft.py, parte2b_campeoes_extra.py, parte2c_picks_detalhe.py", "Parte 2: I(campeão ; V) vs. I(campeão ; V | contexto), com IC por meias-amostras; ruído vs. força real; contribuição por campeão"],
        ["parte3_limite.py", "Parte 3: Fano (estimado e teto), classificadores reais, regra da cadeia"],
        ["parte4_draft_completo.py", "Parte 4: escada de modelos, draft × time × ouro, robustez, pares"],
        ["parte5_objetivos.py", "Parte 5: objetivos isolados vs. depois do ouro, cadeia, vazamento"],
        ["parte6_lanes.py", "Parte 6: lanes isoladas, dado as outras 4, no tempo, razão de chances"],
        ["parte7_t1.py", "Parte 7: Elo contínuo, surpresa das séries e campanhas da T1, volcano plot"],
        ["main.py", "orquestra as partes 1 a 7"],
        ["relatorio/gerar_relatorio.py", "gera este relatório (HTML e PDF) a partir de resultados/"]],
        "Tabela R.1 — Mapa do código.")
    return f"""<section class="parte" id="repro"><h1>Reprodutibilidade</h1>
<p>O código está em Python (numpy, pandas, scipy, scikit-learn, matplotlib). Parâmetros desta execução: 100 permutações, 300
reamostragens bootstrap, 50 meias-amostras, semente 0, todos os anos (2014–2026), 5 repetições de placebo na Parte 4. Todas as
tabelas e figuras são produzidas por <code>main.py</code> e ficam na pasta <code>resultados/</code>.</p>{t}
<p>Exemplos de execução:</p>
<pre>python main.py --data dados.zip --recarregar
python main.py --data dados.zip --partes 2 --perm 200
python main.py --data dados.zip --partes 7
python relatorio/gerar_relatorio.py</pre>
<p class="nota">A opção <code>--recarregar</code> é necessária uma vez, porque as Partes 5, 6 e 7 leem colunas novas do CSV e o
cache antigo não as tem (o <code>main.py</code> avisa se faltar coluna). A Parte 2 leva alguns minutos a mais por causa das
meias-amostras; <code>--meias 0</code> desliga esse cálculo.</p>
</section>"""


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm;
  @bottom-left { content: "Teoria da Informação · League of Legends em bits"; font: 8pt 'Segoe UI', Arial, sans-serif; color: #2c4a73; }
  @bottom-right { content: counter(page); font: 8pt 'Segoe UI', Arial, sans-serif; color: #2c4a73; } }
@page :first { margin-top: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a202c; background: #fff; margin: 0; }
h1 { color: #1f3a5f; font-size: 20pt; margin: 0 0 8pt; }
h2 { color: #1f3a5f; font-size: 13pt; margin: 14pt 0 5pt; break-after: avoid; }
h3 { color: #2b6cb0; font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; }
ul { margin: 3pt 0 7pt; padding-left: 16pt; } li { margin: 2pt 0; text-align: justify; }
.eq { text-align: center; color: #1f3a5f; font-weight: 700; font-style: italic; font-size: 11pt; margin: 6pt 0; }
code, pre { font-family: Consolas, 'Courier New', monospace; font-size: 8.6pt; }
pre { background: #f1f5f9; padding: 6pt 9pt; border-radius: 3pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 1pt; font-size: 8.4pt; break-inside: avoid; }
table.pequena { font-size: 7.6pt; }
th { background: #1f3a5f; color: #fff; text-align: left; padding: 4pt 5pt; font-weight: 600; }
td { padding: 3pt 5pt; border-bottom: 1px solid #d5dde8; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f2f6fb; }
.cap, figcaption { font-size: 8pt; color: #1e3a8a; font-style: italic; margin: 2pt 0 9pt; text-align: left; }
figure { margin: 7pt 0 4pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { text-align: left; }
.box { border-left: 4px solid; border-radius: 3pt; padding: 6pt 10pt; margin: 8pt 0; break-inside: avoid; }
.box .bt { font-weight: 700; font-size: 9pt; margin-bottom: 3pt; }
.box p { margin: 3pt 0; }
.simples { border-color: #2b6cb0; background: #e8f0fa; } .simples .bt { color: #2b6cb0; }
.exemplo { border-color: #38a169; background: #e9f5ec; } .exemplo .bt { color: #2f855a; }
.ler { border-color: #718096; background: #edf1f6; } .ler .bt { color: #4a5568; }
.atencao { border-color: #d69e2e; background: #fdf4e3; } .atencao .bt { color: #975a16; }
.vaz { border-color: #c53030; background: #fcebea; } .vaz .bt { color: #c53030; }
.aprend { border-color: #2c7a7b; background: #e3f2f0; } .aprend .bt { color: #285e61; }
.resumo { border-color: #6b46c1; background: #f3eefb; } .resumo .bt { color: #553c9a; }
.parte, .quebra { break-before: page; }
.capa { padding-top: 0; }
.capa .faixa { height: 15mm; background: #1f3a5f; margin: 0 -16mm 30mm; }
.capa .sobre { color: #1f3a5f; font-size: 9pt; margin-bottom: 4pt; }
.capa .titulo { font-size: 26pt; line-height: 1.15; margin: 0 0 8pt; }
.capa .sub { color: #2b6cb0; font-size: 13pt; margin: 2pt 0; }
.capa .sub2 { color: #2b6cb0; font-size: 11pt; margin: 2pt 0 12pt; text-align: left; }
.rodape-capa { font-size: 8.4pt; color: #1e3a8a; margin-top: 12pt; text-align: left; }
.sum { font-size: 11pt; line-height: 1.9; list-style: none; padding-left: 0; } .sum a { color: #1f3a5f; text-decoration: none; }
.nota { font-size: 8.4pt; color: #1e3a8a; }
"""


def montar():
    figuras_conceituais()
    corpo = "".join([capa(), sumario(), uso(), cap_A(), cap_B(), parte1(), parte2(), parte3(), parte4(), parte5(), parte6(),
                     parte7(), sintese(), limitacoes(), repro()])
    html = (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"/>'
            f'<title>LoL em bits — relatório final</title><style>{CSS}</style></head><body>{corpo}</body></html>')
    caminho = os.path.join(AQUI, "relatorio.html")
    open(caminho, "w", encoding="utf-8").write(html)
    print("HTML:", caminho)
    return caminho


def pdf(html):
    chrome = next((c for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                               shutil.which("google-chrome") or "", shutil.which("chromium") or "") if c and os.path.exists(c)), None)
    if not chrome:
        print("Chrome/Edge não encontrado: abra relatorio.html no navegador e imprima em PDF.")
        return
    saida = os.path.join(AQUI, "relatorio_LoL_Teoria_da_Informacao_Partes_1_a_7.pdf")
    perfil = os.path.join(AQUI, ".chrome-perfil")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={perfil}",
                    "--no-pdf-header-footer", f"--print-to-pdf={saida}", "file:///" + html.replace("\\", "/")],
                   check=False, timeout=180)
    shutil.rmtree(perfil, ignore_errors=True)
    print("PDF:", saida if os.path.exists(saida) else "não gerado")


if __name__ == "__main__":
    pdf(montar())
