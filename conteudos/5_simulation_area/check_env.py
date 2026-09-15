"""
check_env.py -- verify that the environment is complete and that the helper
functions reproduce the numbers printed in Cover and Thomas, Chapter 2.

Run it with:      .venv/bin/python check_env.py
"""

import sys

print("Python", sys.version.split()[0], "at", sys.executable)

missing = []
for name in ["numpy", "scipy", "matplotlib", "jupyterlab", "ipykernel", "ipywidgets"]:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "installed")
        print(f"  {name:12s} {version}")
    except ImportError:
        missing.append(name)
        print(f"  {name:12s} MISSING")

if missing:
    print("\nmissing packages:", ", ".join(missing))
    print("run ./setup_env.sh again")
    sys.exit(1)

import numpy as np
import infotools as it

# Example 2.2.1 of the book: the 4 by 4 joint distribution.
book_table = np.array([
    [1/8,  1/16, 1/32, 1/32],
    [1/16, 1/8,  1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4,  0,    0,    0   ],
])
P = book_table.T

expected = {
    "H(X)":   (it.entropy(it.marginal_X(P)),    7/4),
    "H(Y)":   (it.entropy(it.marginal_Y(P)),    2.0),
    "H(X|Y)": (it.cond_entropy_X_given_Y(P),    11/8),
    "H(Y|X)": (it.cond_entropy_Y_given_X(P),    13/8),
    "H(X,Y)": (it.joint_entropy(P),             27/8),
    "I(X;Y)": (it.mutual_information(P),        0.375),
}

# Example 2.3.1: two Bernoulli distributions.
expected["D(p||q)"] = (it.relative_entropy([0.5, 0.5], [0.75, 0.25]),
                       1 - 0.5*np.log2(3))
expected["D(q||p)"] = (it.relative_entropy([0.75, 0.25], [0.5, 0.5]),
                       0.75*np.log2(3) - 1)

# Example 2.6.1: conditioning reduces entropy on the average.
E = np.array([[0.0, 1/8], [3/4, 1/8]])
expected["H(X) of ex. 2.6.1"]   = (it.entropy(it.marginal_X(E)), 0.5435644431995964)
expected["H(X|Y) of ex. 2.6.1"] = (it.cond_entropy_X_given_Y(E), 0.25)

print("\nchecking the worked examples of Chapter 2:")
failures = 0
for label, (computed, book) in expected.items():
    ok = np.isclose(computed, book, atol=1e-12)
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:20s} computed {computed:.10f}"
          f"   book {book:.10f}")

if failures:
    print(f"\n{failures} check(s) failed")
    sys.exit(1)

print("\nEverything is in order. Open 00_start_here.ipynb.")
