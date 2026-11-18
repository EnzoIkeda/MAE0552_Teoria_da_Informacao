# Quando e por quê uma partida de LoL deixa de ser incerta (em bits)

Projeto final de Teoria da Informação (MAE0552). Dados: Oracle's Elixir (2014 a 2026), ~103 mil partidas profissionais.
O relatório final está em `relatorio/relatorio_LoL_Teoria_da_Informacao_Partes_1_a_7.pdf` e é gerado a partir de `resultados/`.

## Como rodar

```bash
pip install -r requirements.txt
python main.py --data caminho/para/dados.zip --recarregar   # 1ª vez (lê os CSVs e refaz o cache)
python main.py --data dados.zip --partes 2 --perm 200        # só a parte 2, mais permutações
python main.py --data dados.zip --partes 2 --meias 0         # parte 2 sem o IC por meias-amostras (mais rápido)
python relatorio/gerar_relatorio.py                          # relatório HTML + PDF (usa o Chrome ou o Edge instalado)
```

Todos os comandos partem desta pasta (`projetos/02_projeto_final/`). O `relatorio/gerar_relatorio.py` exige Python >= 3.12
(usa barra invertida dentro de f-strings). Os dados brutos e o `cache/` não são versionados.

A 1ª execução lê os CSVs e guarda `cache/raw.pkl`. Se o cache for de uma versão antiga (sem as colunas das Partes 5 a 7),
o `main.py` avisa: rode com `--recarregar`. A execução completa leva ~25 min (a Parte 2, com as meias-amostras, é a mais lenta).
Saídas (figuras `.png` e tabelas `.csv`) vão para `resultados/`.

| arquivo | conteúdo |
|---|---|
| `it_utils.py` | entropia, I(X;V), I(X;V\|Z), Miller-Madow, permutação, bootstrap, Fano |
| `data.py` | leitura dos CSVs; tabelas por partida (time azul), por pick, drafts, objetivos e lanes |
| `parte1_curva.py` | I(ouro em t ; V), t = 10, 15, 20, 25; por que cada partida sai da amostra (`parte1_exclusoes_por_liga.csv`) |
| `parte2_draft.py` | I(campeão ; V) vs I(campeão ; V \| contexto), com IC por meias-amostras e diferenças contra o marginal |
| `parte2b_campeoes_extra.py` | ruído vs força real, ajuste por time-ano, heterogeneidade por patch |
| `parte2c_picks_detalhe.py` | contribuição de cada campeão para I(X;V), estabilidade (metades), posições |
| `parte3_limite.py` | Fano (estimado e teto pela log-loss), classificadores reais, regra da cadeia |
| `parte4_draft_completo.py` | draft completo: escada de modelos, draft x time x ouro, robustez, pares |
| `parte5_objetivos.py` | first blood, torres, dragão, arauto, barão, placas: isolado vs depois do ouro, cadeia, vazamento |
| `parte6_lanes.py` | qual lane é mais decisiva: ouro, XP, CS e KDA por lane; isolada vs dado as outras 4; no tempo |
| `parte7_t1.py` | estudo de caso: Elo contínuo, surpresa (bits) das séries e campanhas da T1 no Mundial 2022-2025, volcano plot |
| `relatorio/gerar_relatorio.py` | monta o relatório (HTML e PDF) a partir dos CSVs e figuras de `resultados/` |

## Notação e escolhas

* **V** = 1 se o time azul venceu (partida = unidade, Partes 1, 3, 4, 5 e 6). Na Parte 2 a unidade é o *pick* e V é a vitória
  do time que o fez. Na Parte 7 a unidade é a série (melhor de cinco).
* **Discretização**: quantis (faixas de mesma frequência). Há tabela de sensibilidade ao nº de faixas.
* **Todas as quantidades estão em bits.**

## Ressalvas importantes (leia antes de apresentar)

1. **Seleção na Parte 1.** `golddiffat25` só existe se o jogo durou ≥ 25 min; o gráfico principal usa a **amostra comum**
   (jogos que chegaram ao min 25). Além disso, ~10% das partidas não têm nenhuma variável de tempo (ligas com
   `datacompleteness = partial`, principalmente LDL e LPL): as Partes 1, 3, 5 e 6 valem para as ligas com dados completos.
2. **Patch/liga não são confundidores.** Em cada partida há um vencedor e um perdedor no mesmo patch e liga, então
   I(patch ; V) = 0 e condicionar só pode *aumentar* a informação do campeão. A força do time confunde campeões específicos
   (Twisted Fate, Bard), mas não reduz a informação agregada; no draft inteiro (Parte 4) tira ~10%. Time-ano é calculado com os
   mesmos resultados e inclui o efeito dos próprios campeões.
3. **Correção por permutação.** Nos contextos sem o lado (nenhum, patch, liga) as duas unidades de uma partida têm V opostos:
   o embaralhamento troca o resultado da **partida inteira** entre partidas do mesmo estrato. Embaralhar unidade por unidade
   (versão anterior) inflava a informação corrigida em ~5 graus de liberdade por estrato (~1 mbit em `patch`). Três partidas da
   base têm os dois lados como derrota e ficam fora do embaralhamento. Nos contextos com milhares de estratos
   (`patch+liga+lado`, `time-ano+lado`) as células têm 1 ou 2 picks e a corrigida tende a ser **subestimada**.
4. **Vazamento temporal.** Totais do fim do jogo (dragões, torres, barões) e objetivos tardios contêm o resultado. As linhas
   marcadas "[VAZAMENTO]"/"pós-fato" existem de propósito.
5. **Fano é estimado.** O limite de Fano usa H(V|X), que só se estima (discretizar aumenta, o plug-in diminui). Só um teto é
   garantido: limite ≤ h⁻¹(log-loss) (`parte3_classificadores.csv`). O IC do Fano usa o bootstrap "básico".
6. **Limite inferior variacional.** O total de cada modelo é limite inferior; o ganho entre dois modelos é diferença de
   limites e não tem direção garantida (um ganho negativo com IC fora do zero indica sobreajuste).
7. **Placas de torre.** Em 2026 a coluna `turretplates` muda de escala (máximo 45 por time, contra 15 até 2025); a análise de
   placas usa só 2022-2025.
8. **Parte 7.** "WLDs" no Oracle's Elixir inclui as seletivas regionais; o início do Mundial é o 1º jogo entre ligas
   diferentes. O Elo é contínuo desde 2014, com regressão à média da liga na virada do ano; o Elo que recomeça todo ano
   (usado na versão inicial) fica no CSV como comparação.

## Rodando no Spyder / Jupyter

Abra `rodar_spyder.py`, edite `DATA` na célula 0 e rode célula por célula. Alternativamente, preencha `DATA_PADRAO` no topo do `main.py`.
