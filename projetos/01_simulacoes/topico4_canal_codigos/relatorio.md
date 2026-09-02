# Relatório — Tópico 4: Capacidade de Canal e Códigos Corretores de Erro

## Objetivo
Testar na prática os limites e as promessas do teorema da codificação de canal
de Shannon para o canal binário simétrico.

## Modelo definido
- Canal BSC com parâmetro p = … (algo entre 0.05 e 0.2).
- Capacidade teórica C = 1 - H_b(p) = …
- Mensagem: sequência de bits (pixels de uma imagem simplificada).
- Códigos: sem codificação; repetição n_rep ∈ {3, 5, 7, …}; Hamming(7,4).

## Método
Transmissão direta e medição da BER vs p; repetição com voto de maioria;
Hamming(7,4) com decodificação por síndrome; tabela BER × taxa efetiva R = k/n
para o mesmo p; varredura de p; varredura de n_rep; cenário R > C.

## Resultados
- BER sem codificação vs p teórico.
- BER pós-decodificação: repetição (vários n_rep) e Hamming(7,4).
- Gráfico BER (escala log) vs tipo de código, com linha de referência sem codificação.
- Comparação da queda da BER de bloco com a proximidade entre R e C.
- O que acontece quando R > C.

## Discussão
Tamanho de amostra necessário para medir BER da ordem de p com confiança
(~100/p bits); conferência da BER de bloco da repetição com a cauda binomial
(`scipy.stats.binom`); trade-off proteção × eficiência.

## Conclusão
…
