"""
infotools.py
Small helper functions used by the Chapter 2 notebooks
(Cover and Thomas, Elements of Information Theory, 2nd edition).

Conventions used everywhere in these notebooks
----------------------------------------------
* A distribution is a plain numpy array of non-negative numbers summing to 1.
* A joint distribution of (X, Y) is a 2-D array P with P[i, j] = p(X = i, Y = j).
* All logarithms are base 2, so every quantity is measured in bits.
* The convention 0 log 0 = 0 is built into every function.

The functions are deliberately short so that you can read them
next to the definitions in the book.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. The basic quantities
# ----------------------------------------------------------------------

def entropy(p):
    """H(X) = - sum_x p(x) log2 p(x), in bits.

    Works for an array of any shape: the array is flattened first, so the
    same function computes the joint entropy H(X, Y) of a 2-D table.
    Zero entries are simply skipped (the convention 0 log 0 = 0).
    """
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(np.sum(-p * np.log2(p)))


def binary_entropy(p):
    """H(p) = -p log2 p - (1-p) log2 (1-p), the binary entropy function.

    Accepts a single number or an array of numbers.
    """
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p, dtype=float)
    inside = (p > 0) & (p < 1)
    q = p[inside]
    out[inside] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return out if out.ndim else float(out)


def joint_entropy(P):
    """H(X, Y) for a joint table P[i, j] = p(x_i, y_j). Same as entropy(P)."""
    return entropy(P)


def marginal_X(P):
    """p(x), obtained by summing the joint table over y."""
    return np.asarray(P, dtype=float).sum(axis=1)


def marginal_Y(P):
    """p(y), obtained by summing the joint table over x."""
    return np.asarray(P, dtype=float).sum(axis=0)


def cond_entropy_Y_given_X(P):
    """H(Y | X) = - sum_{x,y} p(x, y) log2 p(y | x), in bits."""
    P = np.asarray(P, dtype=float)
    pX = P.sum(axis=1)[:, None]          # column of p(x), shape (nx, 1)
    nz = P > 0                           # terms with p(x, y) = 0 contribute 0
    return float(np.sum(-P[nz] * np.log2(P[nz] / np.broadcast_to(pX, P.shape)[nz])))


def cond_entropy_X_given_Y(P):
    """H(X | Y). Same as cond_entropy_Y_given_X applied to the transposed table."""
    return cond_entropy_Y_given_X(np.asarray(P, dtype=float).T)


def relative_entropy(p, q):
    """D(p || q) = sum_x p(x) log2 (p(x) / q(x)), in bits.

    Returns np.inf when q(x) = 0 for some x with p(x) > 0, as the book requires.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if np.any((p > 0) & (q == 0)):
        return np.inf
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))


def mutual_information(P):
    """I(X; Y) = sum_{x,y} p(x, y) log2 ( p(x, y) / (p(x) p(y)) ), in bits."""
    P = np.asarray(P, dtype=float)
    pX = P.sum(axis=1)[:, None]
    pY = P.sum(axis=0)[None, :]
    product = pX * pY                    # the table p(x) p(y)
    nz = P > 0
    return float(np.sum(P[nz] * np.log2(P[nz] / product[nz])))


def cond_mutual_information(P3):
    """I(X; Y | Z) for a 3-D table P3[i, j, k] = p(x_i, y_j, z_k), in bits.

    Uses I(X;Y|Z) = sum p(x,y,z) log2 ( p(x,y,z) p(z) / (p(x,z) p(y,z)) ).
    """
    P = np.asarray(P3, dtype=float)
    pZ = P.sum(axis=(0, 1))[None, None, :]      # p(z)
    pXZ = P.sum(axis=1)[:, None, :]             # p(x, z)
    pYZ = P.sum(axis=0)[None, :, :]             # p(y, z)
    nz = P > 0
    num = (P * pZ)[nz]
    den = (pXZ * pYZ)[nz]
    return float(np.sum(P[nz] * np.log2(num / den)))


# ----------------------------------------------------------------------
# 2. Building random distributions (used to test inequalities)
# ----------------------------------------------------------------------

def random_pmf(n, rng):
    """A random distribution on n outcomes, drawn uniformly from the simplex."""
    p = rng.dirichlet(np.ones(n))
    return p


def random_joint(nx, ny, rng):
    """A random joint distribution on nx * ny outcomes."""
    return rng.dirichlet(np.ones(nx * ny)).reshape(nx, ny)


# ----------------------------------------------------------------------
# 3. Small display helpers
# ----------------------------------------------------------------------

def check(label, condition, tol_info=""):
    """Print a PASS / FAIL line. Used to verify the theorems numerically."""
    mark = "PASS" if bool(condition) else "FAIL"
    extra = f"   {tol_info}" if tol_info else ""
    print(f"[{mark}] {label}{extra}")
    return bool(condition)


def show_table(P, row_name="X", col_name="Y", fmt="{:7.4f}"):
    """Print a joint table with its marginals, as in the book's examples."""
    P = np.asarray(P, dtype=float)
    nx, ny = P.shape
    header = f"{row_name}\\{col_name}".ljust(8) + "".join(f"{j:>7}  " for j in range(1, ny + 1))
    print(header + "     p(x)")
    for i in range(nx):
        row = "".join(fmt.format(P[i, j]) + "  " for j in range(ny))
        print(f"{i + 1:>5}   {row}  {P[i, :].sum():7.4f}")
    tail = "".join(fmt.format(P[:, j].sum()) + "  " for j in range(ny))
    print(f" p(y)   {tail}  {P.sum():7.4f}")


def setup_plot_style():
    """One common look for every figure in these notebooks."""
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })


# A fixed palette so that all the notebooks look the same.
COLORS = {
    "H": "#1f77b4",       # entropy
    "cond": "#ff7f0e",    # conditional entropy
    "I": "#2ca02c",       # mutual information
    "D": "#d62728",       # relative entropy
    "bound": "#7f7f7f",   # theoretical bounds
    "extra": "#9467bd",
}
