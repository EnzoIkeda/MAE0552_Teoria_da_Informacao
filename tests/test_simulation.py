"""Verificacoes de `utils.simulation`.

Roda com pytest (`pytest`) ou diretamente (`python tests/test_simulation.py`).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.simulation import markov_chain, rng, stationary_distribution

TOL = 1e-9


def test_estacionaria_e_ponto_fixo_de_P():
    P = np.array([[0.95, 0.05], [0.20, 0.80]])
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=TOL)
    assert np.isclose(pi.sum(), 1.0, atol=TOL)


def test_estacionaria_bate_com_a_forma_fechada_de_2_estados():
    # P = [[1-a, a], [b, 1-b]]  ->  pi = [b, a] / (a + b)
    a, b = 0.05, 0.20
    P = np.array([[1 - a, a], [b, 1 - b]])
    assert np.allclose(stationary_distribution(P), np.array([b, a]) / (a + b), atol=TOL)


def test_estacionaria_de_matriz_maior():
    P = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=TOL)
    assert np.all(pi >= 0.0)


def test_cadeia_iniciada_na_estacionaria_nao_deriva():
    """A frequencia do estado 1 no primeiro dia bate com a do ultimo."""
    P = np.array([[0.95, 0.05], [0.20, 0.80]])
    pi = stationary_distribution(P)
    g = rng(7)
    n_dias, m = 12, 4000
    series = np.array([markov_chain(P, n_dias, g, x0=g.choice(2, p=pi)) for _ in range(m)])
    assert abs(series[:, 0].mean() - pi[1]) < 0.02
    assert abs(series[:, -1].mean() - pi[1]) < 0.02


def test_P_com_linha_que_nao_soma_1_e_recusada():
    try:
        stationary_distribution(np.array([[0.9, 0.2], [0.2, 0.8]]))
    except ValueError:
        return
    raise AssertionError("esperava ValueError para linha que nao soma 1")


def test_P_nao_quadrada_e_recusada():
    try:
        stationary_distribution(np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]]))
    except ValueError:
        return
    raise AssertionError("esperava ValueError para matriz nao quadrada")


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FALHOU {fn.__name__}: {exc!r}")
    raise SystemExit(failed)
