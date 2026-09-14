# Relatório do Tópico 3: Compressão de Dados, AEP e Critério de Kelly

## Objetivo
Mostrar a relação entre entropia e a taxa ótima de crescimento de patrimônio
(critério de Kelly) e observar a AEP em ação.

## Modelo definido
- m resultados por partida, probabilidades verdadeiras p_i.
- Cotações o_i oferecidas pela casa.
- Estratégias comparadas: tudo no favorito, aposta uniforme, Kelly (b_i = p_i).
- (Passo 9) distribuição conjunta resultado × dica.
- (Passo 11) crença errada q ≠ p.

## Método
Simulação de n apostas em log (soma de log2(b_x o_x)); repetição em 500–1000
temporadas; taxa de crescimento empírica vs W(b, p); histograma de
(1/n) log2(S_n) para n crescente; Kelly condicional à dica; comparação do ganho
com I(X; Y); perda por usar q vs D(p||q).

## Resultados
- Taxa típica e dispersão do patrimônio final por estratégia.
- Empírico vs W(b, p).
- Concentração do histograma em torno de W* conforme n cresce (AEP).
- ΔW empírico vs I(X; Y).
- Perda de crescimento vs D(p||q).

## Discussão
Por que trabalhar em log; efeito de temporadas curtas; casos de quebra do
apostador.

## Conclusão
…
