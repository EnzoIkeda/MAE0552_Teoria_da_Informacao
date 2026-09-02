"""Verificacoes de `utils.estimation`.

Roda com pytest (`pytest`) ou diretamente (`python tests/test_estimation.py`).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.estimation import bootstrap_ci

TOL = 1e-9


def test_bootstrap_ci_cobre_a_media_verdadeira():
    g = np.random.default_rng(0)
    x = g.normal(5.0, 2.0, 500)
    est, lo, hi = bootstrap_ci(x, np.mean, n_boot=1000, generator=g)
    assert lo < 5.0 < hi
    assert lo < est < hi
    assert lo < hi


def test_bootstrap_ci_estreita_com_mais_dados():
    g = np.random.default_rng(1)
    pequeno = bootstrap_ci(g.normal(size=50), np.mean, n_boot=800, generator=g)
    grande = bootstrap_ci(g.normal(size=5000), np.mean, n_boot=800, generator=g)
    assert (grande[2] - grande[1]) < (pequeno[2] - pequeno[1])


def test_bootstrap_pareado_preserva_correlacao():
    g = np.random.default_rng(2)
    x = g.normal(size=400)
    y = x + g.normal(scale=0.1, size=400)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    est, lo, hi = bootstrap_ci((x, y), corr, n_boot=800, generator=g)
    assert est > 0.9
    assert lo > 0.5


def test_bootstrap_valida_tamanhos_diferentes():
    try:
        bootstrap_ci((np.zeros(3), np.zeros(4)), lambda a, b: 0.0, n_boot=10)
    except ValueError:
        return
    raise AssertionError("esperava ValueError para arrays de tamanhos diferentes")


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
