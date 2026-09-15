# Chapter 2 simulations

A set of Jupyter notebooks that illustrate **Cover and Thomas, *Elements of Information Theory*, Chapter 2: entropy, relative entropy, and mutual information**.


Each notebook takes one section of the chapter, states the result in one or two lines in the notation of the book, and then computes it on a small example you can check by hand, plots it, and tests it numerically on thousands of random distributions. The Python is deliberately plain: `numpy` arrays, `for` loops, `matplotlib`. You can make it much better!

---

## 1. First installation

Open a terminal in this folder and run

```
./setup_env.sh
```

The script

1. finds a Python 3.10 or newer on the machine,
2. creates a virtual environment in `.venv` (nothing is installed system-wide),
3. installs `numpy`, `scipy`, `matplotlib`, `jupyterlab`, `ipykernel` and
   `ipywidgets`,
4. writes `requirements-lock.txt` with the exact versions it installed,
5. registers a Jupyter kernel called **Python (infotheo ch2)**,
6. runs `check_env.py`, which verifies that the helper functions reproduce the
   numbers printed in the book.

If no suitable Python is found, the script says so and tells you what to
install. The Python that ships with the Apple command line tools is usually too
old; install one from [https://www.python.org/downloads/](https://www.python.org/downloads/) or with
`brew install python@3.12`.

Then start the notebooks with

```
source .venv/bin/activate
jupyter lab
```

and open `00_start_here.ipynb`.

## 2. Reproducing the environment on a second Mac

This is the case of preparing on one machine and teaching from another, for
example a MacBook Air.

1. Copy this folder to the other machine **without the `.venv` directory**.
   A virtual environment contains absolute paths and does not survive a move.

   ```
   rsync -av --exclude .venv --exclude __pycache__ --exclude .ipynb_checkpoints \
         ~/infoTheo2026/simulation_area/  <destination>/simulation_area/
   ```

   A zip archive or a cloud folder works just as well, as long as `.venv` is
   left out. What must travel with the notebooks is `requirements-lock.txt`.
2. On the second machine, run

   ```
   ./setup_env.sh
   ```

   Because `requirements-lock.txt` is present, it installs **exactly** the same
   package versions as on the first machine rather than whatever is newest that
   day. This is what makes the two machines behave identically in class.
3. Confirm that everything runs end to end:

   ```
   .venv/bin/python run_all_notebooks.py
   ```

   This executes all nine notebooks from top to bottom and reports any error.
   It takes a couple of minutes and does not modify the notebooks.

If you later want to move to newer versions of the packages, run
`./setup_env.sh --update` on one machine, check the notebooks still run, and
copy the regenerated `requirements-lock.txt` to the other machine.

## 3. The notebooks

| File                                                 | Section of the book | Contents                                                                                                                                                                                                           |
| ---------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `00_start_here.ipynb`                              |                     | environment check, conventions, how to use the rest                                                                                                                                                                |
| `01_entropy.ipynb`                                 | 2.1                 | $H(X)$, the binary entropy curve (Figure 2.1), the 20-questions interpretation of Example 2.1.2, entropy as a long-run average, estimating $H$ from data                                                       |
| `02_joint_and_conditional_entropy.ipynb`           | 2.2                 | $H(X,Y)$ and $H(X\mid Y)$, Example 2.2.1 in full, the chain rule (Theorem 2.2.1) and its corollary                                                                                                             |
| `03_relative_entropy_and_mutual_information.ipynb` | 2.3, 2.4            | $D(p\Vert q)$ and its asymmetry (Example 2.3.1), the coding interpretation, $I(X;Y)$, Theorem 2.4.1, Figure 2.2, the binary symmetric channel                                                                  |
| `04_chain_rules.ipynb`                             | 2.5                 | chain rules for$H$, $I$ and $D$ (Theorems 2.5.1 to 2.5.3), conditional mutual information, when conditioning helps and when it does not                                                                      |
| `05_jensen_and_inequalities.ipynb`                 | 2.6, 2.7            | Jensen, the information inequality,$H \le \log\lvert\mathcal X\rvert$, conditioning reduces entropy (with Example 2.6.1), the independence bound, log sum, convexity of $D$, concavity of $H$, Theorem 2.7.4 |
| `06_data_processing.ipynb`                         | 2.8                 | Theorem 2.8.1 and its corollaries, cascades of noisy channels, quantizers, the$Z = X+Y$ counterexample                                                                                                           |
| `07_sufficient_statistics.ipynb`                   | 2.9                 | $I(\theta;T) \le I(\theta;X)$ with equality exactly for sufficient statistics, the Bernoulli example by exact enumeration, equation (2.125), minimal sufficiency                                                 |
| `08_fano.ipynb`                                    | 2.10                | Theorem 2.10.1 and its corollaries, the Fano region, the sharpness family (2.144), Lemma 2.10.1                                                                                                                    |

Run them in order the first time. After that each one is self-contained.

## 4. The other files

| File                      | What it is                                                                                                                                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `infotools.py`          | the helper functions:`entropy`, `joint_entropy`, `cond_entropy_X_given_Y`, `cond_entropy_Y_given_X`, `relative_entropy`, `mutual_information`, `cond_mutual_information`, plus small display and plotting helpers. About a hundred lines, meant to be read next to the book. |
| `requirements.txt`      | the packages, as version ranges. Used the first time.                                                                                                                                                                                                                                      |
| `requirements-lock.txt` | the exact versions installed, written by`setup_env.sh`. This is the file that makes the environment reproducible.                                                                                                                                                                        |
| `setup_env.sh`          | builds`.venv`, installs, registers the kernel, runs the checks.                                                                                                                                                                                                                          |
| `check_env.py`          | verifies the installation and that the helper functions reproduce the book's worked examples.                                                                                                                                                                                              |
| `run_all_notebooks.py`  | executes every notebook and reports failures.                                                                                                                                                                                                                                              |

## 5. Using the notebooks in class

Every notebook follows the same rhythm: a short statement, one cell computing
it on a concrete example, one cell with a simulation or a picture, and a few
`[PASS]` lines testing the result on random distributions.

The numbers meant to be changed during a lecture are at the top of a cell and
carry a comment saying so, for example

```python
q = np.array([0.4, 0.3, 0.2, 0.1])          # <-- change these four numbers
```

Five cells have sliders (`ipywidgets`): the binary entropy function, the binary
symmetric channel in two places, and Fano's inequality. If `ipywidgets` is not
available the cells still run and draw a single fixed value, so nothing breaks.

The `[PASS]` lines are numerical tests, not proofs. They are there so that a
broken edit announces itself immediately.

## 6. VS Code

Open this folder in VS Code, then choose the interpreter `.venv/bin/python`
(command palette, "Python: Select Interpreter"), or pick the kernel
**Python (infotheo ch2)** in the notebook toolbar. Nothing else is needed.

## 7. If something goes wrong

* **`./setup_env.sh: permission denied`** — run `chmod +x setup_env.sh` first.
* **The notebook cannot find `infotools`** — the notebook must be opened from
  this folder, since `infotools.py` lives here. In JupyterLab, start `jupyter lab` from this folder.
* **Sliders do not appear** — `ipywidgets` is missing, or JupyterLab was started
  from a different environment. Check with `.venv/bin/python check_env.py`.
* **The numbers do not match the book** — run `.venv/bin/python check_env.py`.
  It prints a `[PASS]` or `[FAIL]` line for each worked example of the chapter.
* **A different machine gives different results** — the notebooks fix their
  random seeds (`np.random.default_rng(...)`), so the simulated numbers should
  agree to the last digit across machines with the same numpy version. That is
  another reason to install from `requirements-lock.txt`.
