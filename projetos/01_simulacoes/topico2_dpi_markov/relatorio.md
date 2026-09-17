# Relatório do Tópico 2: Desigualdade de Processamento de Dados e Cadeias de Markov

## Objetivo
Comprovar por simulação que nenhuma etapa de processamento cria informação nova
sobre a entrada: I(X; Y1) >= I(X; Y2) >= … ao longo da cadeia.

## Modelo definido
- Fonte X (nº de bits, distribuição).
- Cadeia de canais BSC: parâmetros p de cada elo.
- (Passo 7) classes, médias por classe, ruído gaussiano, etapa de compressão e
  regra de decisão.
- (Passo 9) processo de gerações para o colapso de modelo.

## Método
Propagação de N mensagens por todos os estágios; estimativa de I(X; Yj) para
cada j; gráfico I vs posição; variação da intensidade de ruído; cadeia
rótulo → imagem → representação → predição; gerações sucessivas com D(p_real || p_ger k).

## Resultados
- Curva I(X; Yj) decrescente (dentro da margem de estimação).
- Efeito do ruído na velocidade de degradação.
- DPI ao longo de imagem → representação → predição.
- I ou D(p||q) vs número de gerações.

## Discussão
Solavancos para cima na curva = ruído de estimação, não violação da DPI:
aumentar N e verificar se somem. Efeito do tamanho de amostra por geração.

## Conclusão

Passo 1: com P(X)=[0,5, 0,5], H(X)=1,0000 bit. É o teto teórico de I(X;Yj) ao
longo de toda a cadeia.

Passo 2: demonstração de `compose_bsc_chain` com 5 elos de p=0,10 cada. Um
bloco de exemplo (X=[0,1,1,0,1]) chega ao final da cadeia já alterado
(Y5=[0,1,1,1,0]).

Passo 3: com N=20000 mensagens propagadas pelos 5 elos, as marginais
empíricas de X e de todos os Yj ficam próximas de 0,50, coerente com um BSC
simétrico preservando P(bit=1)=0,5 em cada elo.

Passo 4: I(X;Yj) cai de 0,5255 bit em Y1 para 0,0784 bit em Y5, sempre abaixo
do teto H(X)=1 bit. A sequência é não crescente dentro do IC bootstrap.

| elo | estimado (bits) | IC 95% |
| --- | --- | --- |
| I(X;Y1) | 0,5255 | [0,5126, 0,5387] |
| I(X;Y2) | 0,3071 | [0,2960, 0,3198] |
| I(X;Y3) | 0,1923 | [0,1828, 0,2020] |
| I(X;Y4) | 0,1248 | [0,1169, 0,1327] |
| I(X;Y5) | 0,0784 | [0,0719, 0,0851] |

![Degradação de I(X;Yj) ao longo da cadeia](figuras/t2_passo5_degradacao_i.png)

Passo 5: o gráfico confirma a queda monotônica de I(X;Yj) com a posição j,
dentro da faixa de incerteza do bootstrap.

Passo 6: repetindo a propagação com p=0,02/0,05/0,10/0,20 por elo, quanto
maior o p de cada elo mais rápida a queda de I(X;Yj) ao longo da cadeia.

![Degradação de I(X;Yj) para diferentes intensidades de ruído](figuras/t2_passo6_ruido_intensidade.png)

Passo 7: na cadeia rótulo → imagem (8 dimensões) → representação comprimida
(2 dimensões) → predição, I(rótulo;estágio) cai de 0,8724 para 0,8685 e
depois para 0,8282 bit. Confirma a DPI nas três etapas. O classificador por
distância às médias acerta 97,45% dos rótulos.

| estágio | I(rótulo;estágio) (bits) |
| --- | --- |
| imagem completa | 0,8724 |
| representação comprimida | 0,8685 |
| predição final | 0,8282 |

Passo 8: em 15 gerações, cada uma treinada só na amostra finita (N=300) da
geração anterior, D(p_real‖p_geração_15) chega a 0,3203 bit. Mostra o
colapso de modelo por deriva acumulada de amostragem finita.

![Divergência D(p_real || p_geração) ao longo das gerações](figuras/t2_passo8_colapso_modelo.png)

Conclusão geral: a simulação confirma a desigualdade de processamento de
dados em três cenários distintos, canal ruidoso composto, cadeia
rótulo-imagem-predição e colapso de modelo por gerações sucessivas. A
informação mútua nunca cresce ao processar dados adiante, cai mais rápido
quanto maior o ruído por elo, e uma cadeia de estimativas em amostra finita
acumula divergência da distribuição real geração após geração mesmo sem
nenhum viés introduzido de propósito.
