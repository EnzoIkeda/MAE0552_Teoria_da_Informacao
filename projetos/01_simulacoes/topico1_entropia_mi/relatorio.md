# Relatório do Tópico 1: Entropia, Informação Mútua e Teste de Independência

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

Passo 1: com p(D)=0,08, fpr=0,03 e fnr=0,12, saem H(D)=0,4022, H(T)=0,4626,
H(D,T)=0,6234, H(D|T)=0,1607 e I(D;T)=0,2414 bits. Já mostra que o teste
carrega informação real sobre D.

Passo 2: amostra de N=2000 pares reproduz bem as marginais teóricas
(prevalência 0,0835 contra 0,08, positividade 0,0935 contra 0,098).

Passo 3: as estimativas por contagem ficam próximas da teoria, com erro
relativo entre 0,2% (H(D,T)) e 7,9% (H(D|T)).

| quantidade | estimado | teórico | erro relativo (%) |
| --- | --- | --- | --- |
| H(D) | 0,4144 | 0,4022 | 3,04 |
| H(T) | 0,4480 | 0,4626 | 3,15 |
| H(D,T) | 0,6215 | 0,6234 | 0,30 |
| H(D\|T) | 0,1735 | 0,1607 | 7,92 |
| I(D;T) | 0,2409 | 0,2414 | 0,22 |

Passo 4: o erro de I(D;T) cai de cerca de 23 a 51% em N entre 20 e 100 para
2,9% em N=10000. O IC bootstrap cobre o valor teórico em quase todos os N,
falhando só em N=500.

![Erro de I(D;T) em função de N](figuras/t1_passo4_erro_i_vs_n.png)

Passo 5: para q(D)=[0,85, 0,15], diferente do p(D) real, H(p,q)-H(p) = 0,0325
bits e D(p||q) = 0,0325 bits. Confirma a identidade entre excesso de custo de
codificação e divergência KL.

Passo 6: a cadeia de Markov de 2 estados gera a sequência diária de D e a
série de testes T correspondente. No exemplo simulado D fica concentrado nos
primeiros dias.

![Série de D e T ao longo dos dias](figuras/t1_passo6_serie_d_t.png)

Passo 7: para n=8 dias e 2000 pacientes, H(T_1,...,T_8) estimado (5,3475
bits) fica perto da soma das H(T_i|T_i-1) pela regra da cadeia (5,4836 bits)
e bem abaixo de 8×H(T_1) sob hipótese i.i.d. (7,9519 bits).

Passo 8: aproveitar a dependência temporal economiza 2,4684 bits por
paciente frente a transmitir cada dia isolado, cerca de 31% do custo.

![Economia de bits ao usar a regra da cadeia](figuras/t1_passo8_economia_bits.png)

Passo 9: o teste G rejeita independência tanto para D×T (G=667,95,
p≈2,8×10⁻¹⁴⁷) quanto para T_i×T_i-1 (G=45468,96, p≈0). Confirma dependência
estatisticamente significativa nos dois casos.

Passo 10: a taxa de rejeição nas repetições acompanha a força da dependência
real.

| cenário | I(D;T) teórico | taxa de rejeição |
| --- | --- | --- |
| dependência forte | 0,2414 | 1,000 |
| dependência fraca | 0,0137 | 0,649 |
| independência real | 0,0000 | 0,060 |

![Taxa de rejeição por cenário](figuras/t1_passo10_taxa_rejeicao.png)

Conclusão geral: o teste rápido carrega informação real sobre o estado de
saúde, com I(D;T) bem acima de zero e confirmado pelo teste G. O histórico de
testes ao longo dos dias tem dependência temporal explorável, reduzindo em
torno de 30% o número de bits necessários frente ao tratamento dia a dia como
independente. As estimativas por amostragem finita têm viés perceptível para
N pequeno, mas convergem para a teoria com N maior, e o teste G ganha poder
conforme a dependência real fica mais forte, mantendo a taxa de erro tipo I
próxima do nível nominal quando D e T são de fato independentes.
