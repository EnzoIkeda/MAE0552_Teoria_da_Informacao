# Relatório — Tópico 2: Desigualdade de Processamento de Dados e Cadeias de Markov

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
…
