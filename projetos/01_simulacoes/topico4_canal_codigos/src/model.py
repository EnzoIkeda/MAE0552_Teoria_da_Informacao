"""Tópico 4: canal binário simétrico e códigos corretores de erro.

Canal: `utils.simulation.bsc`. Capacidade C = 1 - H_b(p) via
`utils.info_theory.binary_entropy`.

Aqui ficam os dois codecs pedidos no roteiro: código de repetição (n_rep cópias
+ voto de maioria) e Hamming(7,4) com decodificação por síndrome. Toda a
aritmética do Hamming é módulo 2.
"""

from __future__ import annotations

import numpy as np

from utils.info_theory import binary_entropy


def _como_inteiro(bits: np.ndarray) -> np.ndarray:
    """`asarray` que preserva um dtype inteiro ja existente (ex.: uint8).

    So cai para `uint8` quando a entrada nao e inteira (lista de bool, por
    exemplo). Evita promover mensagens longas para o inteiro de 8 bytes
    padrao do numpy quando o chamador ja passou algo mais compacto.
    """
    bits = np.asarray(bits)
    if not np.issubdtype(bits.dtype, np.integer):
        bits = bits.astype(np.uint8)
    return bits


# ---------------------------------------------------------------- repetição ---


def repetition_encode(bits: np.ndarray, n_rep: int) -> np.ndarray:
    """Cada bit vira `n_rep` cópias (use `n_rep` ímpar para não ter empate)."""
    return np.repeat(_como_inteiro(bits), n_rep)


def repetition_decode(received: np.ndarray, n_rep: int) -> np.ndarray:
    """Voto de maioria em blocos de `n_rep` bits."""
    blocks = _como_inteiro(received).reshape(-1, n_rep)
    # soma em int64 mesmo com blocks em uint8: n_rep cabe sem estourar e evita
    # que o acumulador da soma sature em blocos muito longos.
    maioria = blocks.sum(axis=1, dtype=np.int64) > n_rep / 2
    return maioria.astype(np.uint8)


# ------------------------------------------------------------- Hamming(7,4) ---
# Forma padrão G = [I4 | P], H = [P^T | I3]; bit i do código = coluna i.
G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """Codifica múltiplos de 4 bits; retorna blocos de 7 bits concatenados."""
    m = _como_inteiro(bits).reshape(-1, 4)
    return ((m @ G) % 2).ravel()


# cada coluna de H e um padrao de 3 bits distinto e nao nulo -> a sindrome,
# lida como numero de 0 a 7, aponta direto para a posicao a corrigir. Tabela
# fixa de 8 entradas (-1 = sindrome 0, sem erro) monta essa busca uma vez so.
_PESOS_SINDROME = np.array([4, 2, 1])
_COLUNA_POR_SINDROME = np.full(8, -1, dtype=int)
for _i in range(7):
    _COLUNA_POR_SINDROME[H[:, _i] @ _PESOS_SINDROME] = _i


def hamming74_decode(received: np.ndarray) -> np.ndarray:
    """Decodifica blocos de 7 bits corrigindo até 1 erro por bloco (síndrome).

    Vetorizado: a sindrome de cada bloco vira um indice em 0..7 que consulta
    `_COLUNA_POR_SINDROME` em vez de percorrer os blocos em um laco Python.
    """
    r = _como_inteiro(received).reshape(-1, 7).copy()
    syndromes = (r @ H.T) % 2  # (n_blocos, 3)
    indices = syndromes @ _PESOS_SINDROME  # (n_blocos,), cada um em 0..7
    colunas = _COLUNA_POR_SINDROME[indices]  # -1 onde a sindrome e 0
    linhas = np.nonzero(colunas >= 0)[0]
    r[linhas, colunas[linhas]] ^= 1
    return r[:, :4].ravel()


# ------------------------------------------------------- valores teoricos ---


def bissecao(f, a: float, b: float, tol: float = 1e-10) -> float:
    """Raiz de `f` em [a, b] por bissecao (f(a) e f(b) com sinais opostos)."""
    fa = f(a)
    if fa * f(b) > 0:
        raise ValueError("f(a) e f(b) precisam ter sinais opostos")
    while b - a > tol:
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def binary_entropy_inverse(h: float) -> float:
    """p em [0, 1/2] tal que H_b(p) = h (o ramo de baixo da entropia binaria)."""
    if h <= 0.0:
        return 0.0
    if h >= 1.0:
        return 0.5
    return bissecao(lambda x: binary_entropy(x) - h, 1e-15, 0.5)


def ber_floor_above_capacity(rate: float, capacity: float) -> float:
    """Menor BER possivel para QUALQUER codigo de taxa `rate` num canal de capacidade `capacity`.

    Parte inversa do teorema de Shannon combinada com taxa-distorcao: com uma BER
    p_b tolerada, a taxa maxima alcancavel e C / (1 - H_b(p_b)) (MacKay, 2003,
    cap. 10). Para R > C isso da p_b >= H_b^-1(1 - C/R), um piso que nenhum
    comprimento de bloco remove. Para R <= C o piso e 0: a BER pode ir a zero.
    """
    if rate <= capacity:
        return 0.0
    return binary_entropy_inverse(1.0 - capacity / rate)


def hamming74_ber_exata(p: float) -> float:
    """BER exata dos 4 bits de informacao do Hamming(7,4) num BSC(p).

    O codigo e linear, entao o erro apos decodificar depende so do padrao de erro
    do canal, nao da mensagem: basta decodificar cada um dos 128 padroes como se
    a palavra enviada fosse a nula e ponderar pela probabilidade do padrao.
    """
    padroes = ((np.arange(128)[:, None] >> np.arange(6, -1, -1)) & 1).astype(np.uint8)
    erros_info = hamming74_decode(padroes.ravel()).reshape(128, 4).sum(axis=1)
    pesos = padroes.sum(axis=1)
    prob = p**pesos * (1.0 - p) ** (7 - pesos)
    return float((prob * erros_info).sum() / 4.0)
