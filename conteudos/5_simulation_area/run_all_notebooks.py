"""
run_all_notebooks.py -- execute every notebook in this folder from top to
bottom and report which ones finished without an error.

Use it after moving the folder to another machine, to confirm that the
environment reproduces. It does not modify the notebooks: the executed copies
are written to a temporary folder and discarded.

Run it with:      .venv/bin/python run_all_notebooks.py
"""

import glob
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
notebooks = sorted(glob.glob(os.path.join(here, "*.ipynb")))

if not notebooks:
    print("no notebooks found in", here)
    sys.exit(1)

failures = []
with tempfile.TemporaryDirectory() as out_dir:
    for path in notebooks:
        name = os.path.basename(path)
        print(f"running {name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute",
             "--ExecutePreprocessor.timeout=600",
             "--output-dir", out_dir, path],
            capture_output=True, text=True, cwd=here)
        if result.returncode == 0:
            print("ok")
        else:
            print("FAILED")
            failures.append((name, result.stderr[-2000:]))

print()
if failures:
    for name, message in failures:
        print(f"--- {name} ---")
        print(message)
    print(f"{len(failures)} of {len(notebooks)} notebooks failed")
    sys.exit(1)

print(f"all {len(notebooks)} notebooks ran without an error")
