"""Verificacoes das funcoes de `utils.info_theory`.

Roda com pytest (`pytest`) ou diretamente (`python tests/test_info_theory.py`).
Os testes com scipy servem de conferencia independente e sao pulados se scipy
nao estiver instalado.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.info_theory import (  # noqa: E402
    entropy,
    binary_entropy,
    joint_entropy,
    conditional_entropy,
    cross_entropy,
    kl_divergence,
    mutual_information,
    conditional_mutual_information,
    mi_gtest_statistic,
)
from utils.simulation import rng, sample_joint  # noqa: E402

TOL = 1e-9


def test_entropy_casos_limite():
    assert abs(entropy([0.5, 0.5]) - 1.0) < TOL
    assert abs(entropy([1.0, 0.0])) < TOL
    assert abs(entropy(np.ones(8)) - 3.0) < TOL  # dado justo de 8 faces
    assert abs(binary_entropy(0.5) - 1.0) < TOL
    assert abs(binary_entropy(0.0)) < TOL
    assert abs(binary_entropy(1.0)) < TOL


def test_mi_independente_e_zero():
    px = np.array([0.2, 0.3, 0.5])
    py = np.array([0.6, 0.4])
    joint = np.outer(px, py)
    assert abs(mutual_information(joint)) < TOL


def test_mi_perfeitamente_correlacionada_e_hx():
    joint = np.diag([0.1, 0.4, 0.5])
    hx = entropy(joint.sum(axis=1))
    assert abs(mutual_information(joint) - hx) < TOL


def test_identidades_de_entropia():
    g = rng(0)
    joint = g.random((4, 3))
    joint /= joint.sum()
    hx = entropy(joint.sum(axis=1))
    hy = entropy(joint.sum(axis=0))
    hxy = joint_entropy(joint)
    mi = mutual_information(joint)
    assert abs(mi - (hx + hy - hxy)) < TOL
    assert abs(mi - (hx - conditional_entropy(joint, given="y"))) < TOL
    assert abs(mi - (hy - conditional_entropy(joint, given="x"))) < TOL
    assert mi >= -TOL


def test_kl_e_entropia_cruzada():
    p = np.array([0.1, 0.2, 0.7])
    q = np.array([0.3, 0.3, 0.4])
    assert abs(kl_divergence(p, p)) < TOL
    assert kl_divergence(p, q) > 0.0
    assert kl_divergence(p, q) != kl_divergence(q, p)  # assimetria
    assert abs(cross_entropy(p, q) - (entropy(p) + kl_divergence(p, q))) < TOL
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")


def test_cmi_reduz_a_mi_quando_z_e_constante():
    joint_xy = np.array([[0.1, 0.2], [0.3, 0.4]])
    joint_xyz = joint_xy[:, :, None]  # Z degenerado (um unico valor)
    assert abs(conditional_mutual_information(joint_xyz) - mutual_information(joint_xy)) < TOL


def test_gtest_bate_com_scipy():
    scipy_stats = __import__("importlib").import_module("scipy.stats")
    g = rng(42)
    true_joint = np.array([[0.30, 0.10], [0.15, 0.45]])
    x, y = sample_joint(true_joint, size=5000, generator=g)
    counts = np.zeros((2, 2))
    np.add.at(counts, (x, y), 1.0)

    stat, dof, _ = mi_gtest_statistic(counts)
    res = scipy_stats.chi2_contingency(counts, lambda_="log-likelihood", correction=False)
    assert abs(stat - res.statistic) < 1e-6
    assert dof == res.dof


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FALHOU {fn.__name__}: {exc!r}")
    raise SystemExit(failed)
