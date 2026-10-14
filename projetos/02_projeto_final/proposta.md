# Proposta do projeto final: medindo em bits se um jogo é bem balanceado

MAE0552 Teoria da Informação, IME-USP, 2º semestre de 2026.
Grupo: Enzo Ikeda e mais quatro integrantes (a definir).

## Resumo

Propomos um método para diagnosticar, em bits, se um sistema de decisões
recompensa habilidade, não tem opções dominantes e precifica bem a
informação, separando o efeito real de cada opção do contexto em que ela é
usada.

O problema que motiva o trabalho é o balanceamento do **Métis**, um jogo de
cartas sobre gestão de uma cidade inteligente, desenvolvido como TCC de um dos
integrantes. Um jogo equilibrado deve ter resultado incerto (entropia alta do
resultado), deve premiar quem joga melhor (informação mútua positiva entre
habilidade e vitória), não deve ter uma carta que sozinha decide a partida
(informação mútua baixa entre carta e vitória, depois de descontado o
contexto) e deve dar valor à informação que o jogador tem. Todas essas
perguntas viram quantidades da disciplina: entropia, entropia condicional,
informação mútua, informação mútua condicional, regra da cadeia e
desigualdade de Fano.

Como os dados do Métis são simulados, a metodologia será validada em dados
reais de **xadrez (Lichess, janeiro de 2014, cerca de 610 mil partidas)**,
onde existe uma referência teórica independente: o modelo Elo prevê quanto a
diferença de rating informa sobre o resultado, e podemos comparar essa
previsão com a estimativa empírica. Por fim, aplicamos as mesmas ferramentas
a um problema real fora dos jogos, o **UCI Heart Disease (Cleveland)**, em
que cada exame tem um custo e a pergunta é quanta informação sobre o
diagnóstico cada exame traz por unidade de custo, e se escolher exames de
forma gulosa precifica mal a informação.

Os resultados preliminares já mostram que o método funciona e que ele muda
conclusões. Exemplos: no xadrez, a informação mútua empírica entre diferença
de rating e resultado (0,137 bit no blitz) bate com a prevista pelo Elo
(0,138 bit); no Métis, cartas que pareciam decisivas (cerca de 78 mbit de
informação sobre a vitória) caem para 2 a 20 mbit quando se condiciona ao
progresso da cidade; e saber as cartas futuras vale 19,5 pontos percentuais
de vitória, dos quais 15,6 são recuperados apenas revelando as cartas
permanentemente.

## 1. Motivação

Balancear um jogo costuma ser feito por intuição e por taxa de vitória
bruta. A taxa de vitória bruta tem dois problemas conhecidos:

1. **Confusão com o contexto.** Uma carta que só pode ser jogada em níveis
   avançados aparece muito em partidas vencedoras porque quem chega a esses
   níveis já está ganhando. A associação é real, mas o efeito causal da
   carta pode ser pequeno.
2. **Falta de escala comum.** Taxa de vitória, diversidade de escolhas e
   valor de informação são medidos em unidades diferentes. Em bits, tudo
   fica comparável.

A Teoria da Informação oferece uma linguagem única para as três propriedades
desejadas (incerteza do resultado, recompensa à habilidade, ausência de
opções dominantes) e para o valor da informação. O trabalho de referência é
o estudo de balanceamento de Hearthstone por Teoria da Informação, que
adaptamos ao Métis.

## 2. Casos e dados

| Papel | Caso | Dados | Tamanho | Licença |
|---|---|---|---|---|
| Problema | Métis (simulado) | Simulador do jogo, versão 1.0, semente 20261005 | 18.000 partidas por medição | Próprio |
| Validação | Lichess | Partidas ranqueadas de janeiro de 2014 | 697.600 partidas (610.564 após filtros) | CC0 |
| Aplicação real | UCI Heart Disease | `processed.cleveland.data` e arquivos de custo dos exames | 297 pacientes, 13 atributos | CC BY 4.0 |

Filtros do Lichess: resultado válido, ao menos 4 lances, |diferença de
ajuste de rating| < 40 e exclusão de partidas por correspondência.

### Fontes dos dados

- **Métis.** Partidas geradas pelo simulador do jogo, `Tools/Metis.Simulator`
  do repositório [EnzoIkeda/Metis](https://github.com/EnzoIkeda/Metis). Os
  parâmetros das cartas vêm de `balance_data_1.0.json` (versão 1.0, SHA-256
  `ef2c6b4e2b9e57f65b13ae247b3cd4759d7cc2795a963b0a40595608aaa3d3b4`). As
  partidas, junto com esse arquivo, estão na release
  [dados-metis-v1.0](https://github.com/EnzoIkeda/MAE0552_Teoria_da_Informacao/releases/tag/dados-metis-v1.0)
  deste repositório (`metis_dados_v1.0.tar.xz`, a extrair em
  `projetos/02_projeto_final/dados/`) e podem ser reproduzidas com a semente
  20261005.
- **Lichess.** Lichess Open Database, partidas ranqueadas de janeiro de 2014,
  arquivo `lichess_db_standard_rated_2014-01.pgn.zst`, disponível em
  <https://database.lichess.org/>. Domínio público (CC0).
- **UCI Heart Disease.** JANOSI, A.; STEINBRUNN, W.; PFISTERER, M.; DETRANO,
  R. *Heart Disease* [dataset]. UCI Machine Learning Repository, 1989.
  DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). Arquivo obtido
  em <https://archive.ics.uci.edu/dataset/45/heart+disease>. Usamos
  `processed.cleveland.data` e os custos dos exames em `costs/`. Licença
  CC BY 4.0.

Os dados brutos ficam fora do Git; versionamos apenas os scripts e as
tabelas derivadas.

## 3. Conceitos de Teoria da Informação utilizados

| Conceito | Onde aparece |
|---|---|
| Entropia H(V) | Incerteza do resultado (M1) |
| Entropia condicional H(V \| X) | Resultado dado rating ou nível (M1, M2) |
| Informação mútua I(X; V) | Habilidade, carta, abertura ou exame contra resultado (M2, M4, C1) |
| Entropia normalizada da escolha | Diversidade de jogadas dada a disponibilidade (M3) |
| Informação mútua condicional I(X; V \| Z) | Separar efeito da opção do contexto (M4, C2) |
| Regra da cadeia | Ganho incremental de cada exame (C2, C3) |
| Valor da informação | Diferença de vitória entre jogador informado e desinformado (M5, C3) |
| Desigualdade de Fano | Limite inferior para o erro de diagnóstico com um subconjunto de exames (C3) |

Todas as quantidades serão implementadas "na mão" a partir de
`utils/info_theory.py`.

## 4. Perguntas e resultados preliminares

### Métis (problema)

**M1. O resultado é incerto?** H(V) do resultado por fase.
Preliminar: H(V) = 0,600 bit na fase 1, 0,992 e 0,941 nas fases seguintes,
conforme a política.

**M2. O jogo recompensa habilidade?** I(política; V), usando políticas de
força diferente como proxy de habilidade.
Preliminar: I(política; V) = 0,0755 bit.

**M3. Há escolhas dominantes?** Entropia da carta escolhida dado o conjunto
de cartas disponíveis. Nota: em 3,9% dos turnos a carta jogada não aparece
entre as jogáveis; em todos esses casos trata-se da carta de busca, cuja
carta trazida está registrada, então o conjunto disponível é recuperável.

**M4. Condicionar ao contexto certo muda a conclusão?** I(carta; V) contra
I(carta; V | nível máximo alcançado), com análise de sensibilidade
condicionando ao nível que libera a carta.

| Carta | I bruta (mbit) | I \| nível máximo (mbit) |
|---|---|---|
| Smart Grid | 79,9 | 17,9 |
| Gêmeo Digital | 77,7 | 19,8 |
| Central de Monitoramento | 75,9 | 1,5 |
| Escola Técnica | 37,6 | 13,4 |

I(nível; V) = 0,240 bit. A Escola Técnica é uma carta básica que
provavelmente atua como mediadora (ajuda a cidade a subir de nível), e por
isso parte do seu efeito some ao condicionar ao nível. Uma análise anterior
afirmava que a informação dessa carta "quase zerava"; verificamos que isso
vinha de condicioná-la a um nível que ela não exige.

**M5. Quanto vale a informação?** Comparação pareada (mesmas sementes)
entre o jogador que conhece as cartas futuras, o desinformado e o
desinformado com revelação permanente. Medição nova, na versão 1.0 do jogo,
18.000 partidas, fases 2 em diante (4.500 partidas por política):

| Política | Vitória (fases 2+) | H(V) |
|---|---|---|
| Informado | 64,3% | 0,940 |
| Desinformado com revelação | 60,4% | 0,969 |
| Desinformado | 44,8% | 0,992 |

| Comparação | Diferença | IC 95% (bootstrap) | Discordantes | McNemar |
|---|---|---|---|---|
| Informado menos desinformado | +19,5 pontos | [+17,8; +21,3] | 1317 / 440 | p = 2e-101 |
| Revelação menos desinformado | +15,6 pontos | [+13,8; +17,5] | 1175 / 473 | p = 6e-69 |
| Informado menos revelação | +3,9 pontos | [+2,7; +5,2] | 496 / 321 | p = 1e-9 |

A revelação permanente recupera cerca de 80% do valor da informação
completa. O efeito é estável entre os três cenários iniciais (18,6 a 20,7
pontos). A reprodução foi conferida: as 1.200 partidas do jogador informado
que também existem na medição oficial da versão 1.0 saíram idênticas.

### Lichess (validação)

**M1.** A entropia do resultado é 1,21 bit com ratings iguais e cai para
0,30 bit quando a diferença passa de 400 pontos.

**M2.** I(diferença de rating; V) empírica contra a prevista pelo modelo Elo
com vantagem das brancas:

| Ritmo | I empírica (bit) | I pelo modelo (bit) |
|---|---|---|
| Blitz | 0,1371 | 0,1379 |
| Bullet | 0,1718 | 0,1684 |
| Clássico | 0,1668 | 0,1569 |

A calibração erra no máximo 2,3 pontos percentuais. O Lichess usa Glicko-2,
de modo que o Elo é uma aproximação; mesmo assim a concordância valida o
estimador.

**M3.** Nos 8 primeiros meio-lances do blitz, a entropia normalizada da
jogada cai de 0,57 para 0,41 com o aumento da faixa de rating: jogadores
mais fortes concentram suas escolhas. Só 55% das decisões estão em células
com amostra suficiente; pretendemos juntar meses ou reduzir o limite de
ocorrências.

**M4.** I(abertura ECO; V), com correção de Miller-Madow, cai de 6,02 para
0,96 mbit ao condicionar à diferença de rating, mas sobe para 8,56 mbit ao
condicionar à faixa absoluta. Isso mostra que o condicionamento pode reduzir
ou aumentar a informação, e que a correção de viés importa (sem ela, 11,30
mbit). Exemplo: a abertura C30 vence 62,8% das vezes, mas só 54,3% depois de
padronizar pela diferença de rating.

### Heart Disease (aplicação)

n = 297, prevalência de 46,1%. Exames mais informativos isoladamente: `thal`
(0,210 bit), `cp` (0,197 bit) e `ca` (0,185 bit).

**C1.** Informação mútua de cada exame com o diagnóstico, e a mesma
quantidade dividida pelo custo do exame.

**C2.** Informação mútua condicional e regra da cadeia: quanto cada exame
acrescenta dado o que já se sabe.

**C3.** Escada de políticas de escolha de exames: triagem fixa, aleatória,
gulosa de um passo, busca com antecipação de k passos e todos os exames.
Comparamos acurácia contra custo, com teste de McNemar entre políticas, e o
limite de Fano para subconjuntos pequenos de exames. A pergunta é a mesma
de M5: uma heurística de um passo subestima o valor de exames que só são
úteis em conjunto?

## 5. Métodos estatísticos

- Estimadores por contagem (plug-in) com correção de Miller-Madow.
- Intervalos de confiança por bootstrap.
- Teste-G para independência (2N·I ~ qui-quadrado).
- Teste de McNemar para comparações pareadas.
- No Métis, sementes comuns entre políticas, para que as comparações sejam
  pareadas partida a partida.
- Experimentos sintéticos com distribuição conhecida para medir o viés dos
  estimadores antes de aplicá-los aos dados.

## 6. Riscos e limitações

- **Associação não é causa.** A informação mútua condicional reduz a
  confusão, mas não a elimina; quando a variável de contexto é mediadora
  (caso da Escola Técnica), condicionar pode esconder um efeito real.
- **Simulação contra jogo real.** As políticas do Métis são agentes
  artificiais; a habilidade é aproximada pela força da política.
- **Amostra pequena no Heart Disease.** Com 297 pacientes, a informação
  mútua de muitos exames conjuntos é muito enviesada; por isso Fano e a
  busca ficam restritos a subconjuntos pequenos.
- **M3 restrita às aberturas** no xadrez, por falta de amostra em posições
  posteriores.
- **Glicko-2 contra Elo** na referência teórica do Lichess.
- **Ressalva clínica.** O caso Heart Disease é ilustrativo e não constitui
  recomendação médica.

## 7. Lista de tarefas

As frentes ainda não têm responsável definido. Itens marcados como
**pendente** dependem de decisão do grupo.

### Decisões pendentes

- [ ] **Pendente:** data da apresentação.
- [ ] **Pendente:** responsável por cada frente.
- [ ] **Pendente:** concordância e forma de crédito da coautora do Métis
      (Beatriz Gonzaga).

### Frente A: estimadores e experimentos sintéticos

- [ ] Criar `projetos/02_projeto_final/` com `src/`, `notebooks/` e `tabelas/`.
- [ ] Implementar Miller-Madow, bootstrap e teste-G sobre `utils/info_theory.py`.
- [ ] Experimento sintético: viés da informação mútua contra n e tamanho do alfabeto.
- [ ] Experimento sintético de confusão: I(X;V) contra I(X;V|Z) com Z conhecido, incluindo caso mediador.
- [ ] Implementar o limite de Fano e testá-lo em distribuição conhecida.

### Frente B: Métis

- [ ] Script que gera as tabelas derivadas a partir de `games.csv` e `turns.csv` da versão 1.0.
- [ ] M1 e M2 por fase, arquétipo e cenário, com IC.
- [ ] M3: entropia da escolha dada a disponibilidade, tratando a carta de busca.
- [ ] M4: tabela bruta contra condicional por carta, mais sensibilidade pelo nível que libera a carta.
- [x] M5: medição nova do valor da informação na versão 1.0 (18.000 partidas).
- [ ] M5: decompor o valor da informação por cenário e por fase.

### Frente C: Lichess

- [ ] Versionar o script de leitura do PGN e a tabela derivada.
- [ ] M1 e M2 com curva empírica contra Elo, por ritmo.
- [ ] M3: juntar meses ou reduzir o limite de células; entropia por faixa de rating.
- [ ] M4: abertura contra resultado, bruta e condicional, com e sem Miller-Madow.
- [ ] Sensibilidade aos filtros e à discretização do rating.

### Frente D: Heart Disease

- [ ] Leitura de `processed.cleveland.data`, discretização e tabela de custos.
- [ ] C1: informação mútua por exame e por custo.
- [ ] C2: informação condicional e regra da cadeia.
- [ ] C3: escada de políticas, McNemar e limite de Fano.
- [ ] Sensibilidade à discretização das variáveis contínuas.

### Frente E: síntese e apresentação

- [ ] Enviar este resumo ao professor.
- [ ] Definir o modelo de figuras e tabelas comum (`utils/plotting.py`).
- [ ] Tabela-síntese: a mesma pergunta respondida nos três casos.
- [ ] Relatório final.
- [ ] Slides e ensaio.
