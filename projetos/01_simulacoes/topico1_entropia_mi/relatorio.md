# Relatório — Tópico 1: Entropia, Informação Mútua e Teste de Independência

## Objetivo
Decidir se o teste rápido carrega informação real sobre o estado de saúde e
quantos bits, no mínimo, são necessários para transmitir o histórico de testes.

## Modelo definido
- Prevalência P(D=1) = …
- Falso positivo P(T=1 | D=0) = …
- Falso negativo P(T=0 | D=1) = …
- Conjunta p(d, t) resultante e valores teóricos de H(D), H(T), H(D,T), H(D|T), I(D;T).

## Método
Geração de N pares (D, T); estimativa das mesmas quantidades por contagem;
comparação com a teoria; varredura de N; entropia cruzada e D(p||q) para um
q(D) adivinhado; dimensão temporal via cadeia de Markov de 2 estados; teste-G
de independência.

## Resultados
- Tabela teoria x estimativa (N fixo).
- Curva do erro de I(D;T) vs N.
- H(p,q) - H(p) vs D(p||q).
- H(T_1..T_n) vs soma das condicionais vs n·H(T_1).
- Estatística do teste-G, g.l. e p-valor nos três cenários; taxa de rejeição.

## Discussão
Viés de amostra finita em I estimada; cenários com contagens esperadas < 5;
economia de bits ao transmitir de forma condicional.

## Conclusão
…
