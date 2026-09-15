#!/usr/bin/env bash
#
# setup_env.sh -- build the Python environment for the Chapter 2 notebooks.
#
#   ./setup_env.sh              install (uses requirements-lock.txt if present)
#   ./setup_env.sh --update     ignore the lock file, re-resolve, write a new lock
#   ./setup_env.sh --no-kernel  do not register a Jupyter kernel
#
# The environment is created in a folder .venv next to this script, so nothing
# is installed system-wide and nothing outside this folder is touched.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"
LOCK="$HERE/requirements-lock.txt"
REQ="$HERE/requirements.txt"
KERNEL_NAME="infotheo-ch2"
KERNEL_LABEL="Python (infotheo ch2)"

UPDATE=0
REGISTER_KERNEL=1
for arg in "$@"; do
  case "$arg" in
    --update)    UPDATE=1 ;;
    --no-kernel) REGISTER_KERNEL=0 ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- 1. Python
# Prefer a recent interpreter. Apple's built-in python3 is often too old.
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${version%%.*}"; minor="${version##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  cat <<'MSG'
No Python 3.10 or newer was found on this machine.

The version that ships with the Apple command line tools is usually too old.
Install a current Python in one of these ways and run this script again:

  * download the macOS installer from https://www.python.org/downloads/
  * or, if you use Homebrew:   brew install python@3.12

MSG
  exit 1
fi

echo "using $PY ($("$PY" --version 2>&1))"

# ---------------------------------------------------------------- 2. venv
if [ ! -d "$VENV" ]; then
  echo "creating the virtual environment in .venv"
  "$PY" -m venv "$VENV"
else
  echo ".venv already exists, reusing it"
fi

VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip setuptools wheel >/dev/null

# ---------------------------------------------------------------- 3. packages
if [ -f "$LOCK" ] && [ "$UPDATE" -eq 0 ]; then
  echo "installing the exact versions listed in requirements-lock.txt"
  "$VPY" -m pip install -r "$LOCK"
else
  echo "installing from requirements.txt"
  "$VPY" -m pip install -r "$REQ"
  echo "writing requirements-lock.txt with the exact versions that were installed"
  "$VPY" -m pip freeze > "$LOCK"
fi

# ---------------------------------------------------------------- 4. kernel
if [ "$REGISTER_KERNEL" -eq 1 ]; then
  echo "registering the Jupyter kernel \"$KERNEL_LABEL\""
  "$VPY" -m ipykernel install --user \
      --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL" >/dev/null
fi

# ---------------------------------------------------------------- 5. check
echo
"$VPY" "$HERE/check_env.py"

cat <<MSG

Done.

To start JupyterLab:

    source .venv/bin/activate
    jupyter lab

then open 00_start_here.ipynb.

In VS Code, open this folder and choose the interpreter .venv/bin/python
(command palette, "Python: Select Interpreter"), or pick the kernel
"$KERNEL_LABEL" in the notebook toolbar.
MSG
