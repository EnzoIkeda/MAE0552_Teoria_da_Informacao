# O que mudou em relação à versão de 16/11 (Partes 1 a 6 + Parte 7 do Lucas)

Todos os números das partes que não foram alteradas (2b, 2c, 3, 4 e 6) foram reproduzidos exatamente, até a precisão da
máquina, com os mesmos dados e a mesma semente. As mudanças abaixo estão no código e no relatório.

## Correções de método

1. **Permutação da Parte 2 (patch, liga e marginal).** O embaralhamento era feito lado a lado e desfazia o par
   vencedor/perdedor de cada partida; o ruído esperado ficava ~5 graus de liberdade por estrato abaixo do real e a
   informação corrigida saía inflada. Agora embaralha-se o resultado da partida inteira. Em dados sintéticos em que o
   campeão não influencia a vitória (verdade = 0), a versão antiga dava 14,9 mbit com p = 0,03 em "patch"; a nova, 1,1 ± 3,1.
   Três partidas da base têm os dois lados como derrota e ficam fora do embaralhamento. Nos dados reais: patch 3,71 → 2,72 mbit,
   liga 2,66 → 2,22, marginal 0,85 → 0,84 (a conta teórica previa ~1,0 e ~0,4 mbit a menos). Os contextos com o lado não mudam
   de método (patch + liga + lado 3,94; time-ano + lado 1,42, agora com p = 0,02).
2. **IC na Tabela 2.1.** Intervalo de 95% da informação corrigida por 50 meias-amostras e intervalo das diferenças contra o
   marginal. Os contextos com milhares de estratos tendem a ficar abaixo da verdade (células de 1 ou 2 picks).
3. **Fano é estimativa.** A log-loss da logística (0,7262 bit) é um teto para H(V|X), então o limite verdadeiro é no máximo
   20,2%; o 20,6% discretizado não é piso garantido. Novo `parte3_classificadores.csv` com o teto; IC do Fano pelo bootstrap
   "básico" (o percentil caía fora da própria estimativa).
4. **Placas de torre (Parte 5).** Em 2026 a coluna `turretplates` muda de escala (média 25, máximo 45 por time; até 2025 o
   máximo é 15). A análise de placas passa a usar só 2022-2025, e o resultado muda: as placas acrescentam 0,0001 bit
   depois do ouro (nada), contra 0,082 bit com 2026 misturado. O "maior efeito entre os objetivos iniciais" era artefato.
5. **Parte 7, início do Mundial.** No Oracle's Elixir a liga "WLDs" inclui as seletivas regionais (ex.: T1 x KT em 14/09/2024).
   O "pré-Worlds" do `story.py` usava a primeira data de "WLDs" (31/08/2024). Agora o início é o 1º jogo entre ligas diferentes.
6. **Parte 7, Elo.** O Elo que recomeça em 1500 todo ano coloca times de ligas fracas acima dos coreanos (DetonatioN FocusMe
   em 2º em 2022). O Elo agora é contínuo desde 2014, com regressão à média da liga na virada do ano; o Elo anual fica no CSV
   como comparação. Foram incluídas a semifinal de 2024 (Gen.G), a seletiva de 2024, o play-in e a final de 2025, e a surpresa
   acumulada de cada campanha.
7. **Parte 7, volcano plot.** Em 2024 a T1 não ficou abaixo da média: ficou acima, mas bem abaixo da Gen.G (efeito médio
   0,20 dp contra 0,52; 1 titular acima da linha de Bonferroni contra 5).

## Correções de texto (relatório)

* Limite de Fano chamado de "piso" virou estimativa, com o teto de 20,2% (A.7, A.12, Parte 3, Limitações).
* "Limite inferior nunca superestima" vale para o total, não para os ganhos entre modelos (A.9, 4.1, Parte 6).
* −0,57 ± 0,38 (ban × pick) não é "indistinguível de zero": é negativo, sobreajuste (4.2).
* "Todo vs soma das partes": a comparação justa é M4 contra M1 (+2,3 mbit); 8,9 contra 8,5 misturava estimadores (4.6).
* "Draft completo" = só picks; com os bans são 13,1 mbit (1,3% de H(V), ~20 vezes menos que o ouro). O modelo não tem força
  por patch, então é um limite inferior (4.2, 4.6, limitações).
* "O confundidor real é a força do time" virou "confunde campeões específicos, não o agregado" (2.1, 2.3, conclusões).
* Draft e ouro complementares: parte do aumento é automática quando as variáveis são independentes (A.5, 4.3).
* Ligas sem dados de tempo (~10% das partidas) declaradas no capítulo B e nas limitações.
* GBM com 33 árvores não prova falta de estrutura (4.4); Zeri → Udyr (2.3); "logística esparsa" → penalização L2 (4.1).
