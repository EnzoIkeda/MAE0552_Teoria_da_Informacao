# Relatório do Tópico 1: Entropia, Informação Mútua e Teste de Independência

## Objetivo
Decidir se o teste rápido carrega informação real sobre o estado de saúde e
quantos bits, no mínimo, são necessários para transmitir o histórico de testes.

## Modelo definido
- Prevalência P(D=1) = 0,08.
- Falso positivo P(T=1 | D=0) = 0,03.
- Falso negativo P(T=0 | D=1) = 0,12.

Conjunta p(d, t) resultante:

| | T=0 | T=1 | total |
| --- | --- | --- | --- |
| **D=0** | 0,8924 | 0,0276 | 0,9200 |
| **D=1** | 0,0096 | 0,0704 | 0,0800 |
| **total** | 0,9020 | 0,0980 | 1,0000 |

Valores teóricos: H(D) = 0,4022, H(T) = 0,4626, H(D,T) = 0,6234,
H(D|T) = 0,1607 e I(D;T) = 0,2414 bits.

Para a dimensão temporal (Passos 6 a 8), D segue uma cadeia de Markov de 2
estados com P(D=0→1) = 0,02 e P(D=1→0) = 0,23. Esses valores não são
arbitrários: a estacionária de uma cadeia de 2 estados é [b, a]/(a+b), então
b = a(1−prevalência)/prevalência faz a cadeia ter **exatamente a prevalência do
modelo acima** em regime, e a extensão temporal descreve a mesma população dos
Passos 1 a 5. A cadeia é iniciada na estacionária, não em um estado fixo.

## Método
Geração de N pares (D, T) e estimativa das mesmas quantidades por contagem;
varredura de N com R amostras independentes por N; entropia cruzada e D(p‖q)
para um q(D) adivinhado; dimensão temporal via cadeia de Markov de 2 estados
iniciada na estacionária; teste-G de independência escrito como
G = 2N ln2 · Î(D;T).

Nos Passos 7 e 8 as três quantidades comparadas também são calculadas de forma
**exata**, e não só estimadas: como D é Markov e T depende apenas do D do mesmo
dia, o par (D, T) é um modelo oculto de Markov, e p(t₁…tₙ) sai do algoritmo
forward somando sobre os caminhos de D. As estimativas por contagem são então
conferidas contra esse valor exato, em vez de contra outra estimativa.

## Resultados
- Tabela teoria × estimativa para N = 2000 fixo, com IC bootstrap de I(D;T).
- Curva do erro de I(D;T) vs N, média de 400 amostras independentes por N.
- H(p,q) − H(p) vs D(p‖q).
- H(T₁…Tₙ) vs soma das condicionais de ordem 1 vs n·H(T₁), estimados e exatos.
- Estatística do teste-G, g.l. e p-valor para D×T e T₂×T₁; taxa de rejeição nos
  três cenários.

## Discussão
**Viés de amostra finita.** A informação mútua estimada por contagem é
enviesada para cima, e o viés cresce com o tamanho do alfabeto e cai com N —
por isso o Passo 4 mede a média de 400 amostras por N, não uma tirada só. Em
N = 20 aparece um segundo efeito que o viés sozinho não explica: **21% das
amostras não contêm nenhum paciente infectado** (P(nenhum D=1) = 0,92²⁰ = 19%),
e nessas a tabela de contingência tem uma linha vazia e Î sai exatamente zero.
A distribuição de Î em N = 20 é bimodal, o que faz a média em N = 20 ficar
*abaixo* da média em N = 50, invertendo o que se esperaria de um viés puro.

**Contagens esperadas pequenas.** No Passo 10 a menor contagem esperada é
N·P(D=1)·P(T=1|D=1) ≈ 300 · 0,08 · 0,88 ≈ 21, acima do mínimo prático de 5 para
confiar na aproximação qui-quadrado. Com N = 300 e prevalência 0,08 a célula
crítica ainda tem folga; abaixo disso a taxa de erro tipo I deixaria de ser
confiável.

**Independência das observações no teste-G.** O teste-G pressupõe N observações
independentes, e G escala linearmente com N. Empilhar os 7 pares consecutivos
(T₁,T₂), (T₂,T₃), … de cada paciente daria N = 7M "observações" que se
sobrepõem duas a duas e vêm da mesma trajetória — a estatística sairia inflada
por pseudo-replicação, por um fator da ordem de 7, sem que nenhuma informação
nova tivesse entrado. O Passo 9 usa por isso **um único par por paciente**.

**Economia de bits.** A comparação entre o custo condicional e n·H(T₁) só mede
dependência temporal se a cadeia estiver em regime. Iniciada fora da
estacionária, H(T₁) reflete a condição inicial e não a distribuição do processo:
partindo de um estado uniforme, a "economia" chegaria a 2,49 bits, e partindo da
prevalência do Passo 1 com uma cadeia não calibrada ela chegaria a ser
**negativa**. Nenhum dos dois números diria nada sobre dependência temporal.

## Conclusão

Passo 1: com p(D)=0,08, fpr=0,03 e fnr=0,12, saem H(D)=0,4022, H(T)=0,4626,
H(D,T)=0,6234, H(D|T)=0,1607 e I(D;T)=0,2414 bits. Já mostra que o teste
carrega informação real sobre D: observar T reduz a incerteza sobre o estado de
saúde de 0,4022 para 0,1607 bit, ou seja, em 60%.

Passo 2: amostra de N=2000 pares reproduz bem as marginais teóricas
(prevalência 0,0835 contra 0,08, positividade 0,0935 contra 0,0980).

Passo 3: as estimativas por contagem ficam próximas da teoria, com erro
relativo entre 0,2% (I(D;T)) e 7,9% (H(D|T)).

| quantidade | estimado | teórico | erro relativo (%) |
| --- | --- | --- | --- |
| H(D) | 0,4144 | 0,4022 | 3,04 |
| H(T) | 0,4480 | 0,4626 | 3,15 |
| H(D,T) | 0,6215 | 0,6234 | 0,30 |
| H(D\|T) | 0,1735 | 0,1607 | 7,92 |
| I(D;T) | 0,2409 | 0,2414 | 0,22 |

O IC bootstrap de 95% para I(D;T) nessa amostra é [0,2004, 0,2820] e cobre o
valor teórico. A concordância em I(D;T) é melhor do que nas entropias que a
compõem, porque os erros de H(T) e H(D|T) apontam para o mesmo lado e se
cancelam na diferença.

Passo 4: com 400 amostras independentes por N, a média de Î(D;T) converge para
o valor teórico e a faixa entre os percentis 10 e 90 encolhe de [0,0000, 0,6098]
em N=20 para [0,2305, 0,2532] em N=10000.

| N | Î médio | erro relativo (%) | faixa 10–90% |
| --- | --- | --- | --- |
| 20 | 0,2639 | 9,3 | [0,0000, 0,6098] |
| 50 | 0,2700 | 11,8 | [0,1084, 0,4465] |
| 100 | 0,2467 | 2,2 | [0,1266, 0,3725] |
| 500 | 0,2401 | 0,6 | [0,1927, 0,2878] |
| 2000 | 0,2412 | 0,1 | [0,2154, 0,2670] |
| 10000 | 0,2418 | 0,1 | [0,2305, 0,2532] |

O percentil 10 igual a zero em N=20 é a assinatura das amostras sem nenhum
infectado, discutidas acima. A partir de N=500 o erro da média cai abaixo de 1%.

![Erro de I(D;T) em função de N](figuras/t1_passo4_erro_i_vs_n.png)

Passo 5: para q(D)=[0,85, 0,15], diferente do p(D) real, H(p,q)=0,4347 bits
contra H(p)=0,4022, e a diferença de 0,0325 bit bate exatamente com
D(p‖q)=0,0325. Codificar D com o modelo errado custa 8% a mais de bits, e esse
excesso é a divergência KL — não uma quantidade parecida com ela.

Passo 6: a cadeia calibrada tem matriz [[0,98, 0,02], [0,23, 0,77]] e
estacionária [0,92, 0,08], idêntica à prevalência do Passo 1. A série simulada
mostra longos trechos em D=0 interrompidos por episódios curtos de infecção, e
T acompanhando D com erros ocasionais.

![Série de D e T ao longo dos dias](figuras/t1_passo6_serie_d_t.png)

Passo 7: para n=8 dias e 20000 pacientes, as estimativas por contagem batem com
os valores exatos dentro de 1,3%:

| quantidade | estimado | exato | erro relativo (%) |
| --- | --- | --- | --- |
| H(T₁…T₈) | 2,9391 | 2,9773 | 1,28 |
| soma H(Tᵢ\|Tᵢ₋₁) | 3,0290 | 3,0571 | 0,92 |
| 8·H(T₁) | 3,6843 | 3,7010 | 0,45 |

O ponto conceitual do passo é a relação entre as duas primeiras linhas.
**T sozinho não é uma cadeia de Markov**: quem é Markov é D, e T é a observação
ruidosa dele, o que faz de (D, T) um modelo oculto de Markov. A regra da cadeia
condiciona em todo o passado, H(T¹⁻ⁿ) = Σ H(Tᵢ | T₁…Tᵢ₋₁); condicionar apenas em
Tᵢ₋₁ é uma aproximação de ordem 1 e portanto um **limite superior**. A folga
entre as duas é 0,0797 bit, 2,7% de H(T₁…T₈) — pequena, mas estrutural, e não
ruído de estimação: ela aparece igual no cálculo exato.

![Regra da cadeia: H(T_1..T_n), soma das condicionais e n·H(T_1)](figuras/t1_passo7_regra_cadeia.png)

Passo 8: transmitir os 8 dias como amostras independentes custa 3,7010 bits por
paciente. Usando o dia anterior como contexto o custo cai para 3,0571 bits
(economia de 0,6439 bit, 17,4%), e explorando todo o passado cai para 2,9773
bits (economia de 0,7236 bit, 19,6%). Ou seja, **a memória de um único dia já
captura 89% de toda a economia disponível** — o resto exige memória mais longa e
paga pouco por isso. Como a cadeia está em regime, T₁ tem a mesma distribuição
de qualquer outro dia e a comparação mede dependência temporal, não a condição
inicial.

![Economia de bits ao usar a dependência temporal](figuras/t1_passo8_economia_bits.png)

Passo 9: o teste G rejeita a independência nos dois casos.

| teste | N | Î (bits) | G | g.l. | p-valor |
| --- | --- | --- | --- | --- | --- |
| D × T | 2000 | 0,2409 | 667,95 | 1 | 2,8×10⁻¹⁴⁷ |
| T₂ × T₁ | 20000 | 0,0919 | 2547,46 | 1 | < 10⁻³⁰⁰ |

Os 20000 pares de T₂×T₁ são um por paciente, portanto independentes entre si.

Passo 10: a taxa de rejeição acompanha a força da dependência real, e sob
independência verdadeira fica no nível nominal.

| cenário | I(D;T) teórico | taxa de rejeição |
| --- | --- | --- |
| dependência forte | 0,2414 | 1,000 |
| dependência fraca | 0,0137 | 0,654 |
| independência real | 0,0000 | 0,049 |

Com R=1000 repetições e α=0,05, o erro de Monte Carlo da taxa sob H₀ é
√(0,05·0,95/1000) = 0,0069, então 0,049 está bem dentro do esperado: o teste
não é anticonservador nessas condições. Na dependência fraca (I = 0,0137 bit)
ele acerta em 65% das vezes com N=300 — o poder é limitado, mas a decisão está
correta na maioria das repetições, que é o que o passo pede para verificar.

![Taxa de rejeição por cenário](figuras/t1_passo10_taxa_rejeicao.png)

Conclusão geral: o teste rápido carrega informação real sobre o estado de
saúde — I(D;T) = 0,2414 bit, 60% da incerteza sobre D, confirmado pelo teste G
com p-valor da ordem de 10⁻¹⁴⁷. O histórico de testes ao longo dos dias tem
dependência temporal explorável, mas o ganho é modesto: cerca de 20% dos bits,
e quase todo ele já disponível olhando apenas o dia anterior. As estimativas por
amostragem finita convergem para a teoria, com dois efeitos distintos em N
pequeno — o viés para cima da informação mútua e a massa de amostras sem nenhum
paciente infectado — e o teste G ganha poder conforme a dependência real fica
mais forte, mantendo a taxa de erro tipo I no nível nominal quando D e T são de
fato independentes.
