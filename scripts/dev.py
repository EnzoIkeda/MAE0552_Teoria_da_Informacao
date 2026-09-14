#!/usr/bin/env python3
"""Comandos de desenvolvimento, multiplataforma (Windows / Linux / macOS).

    python scripts/dev.py setup           # cria .venv, instala tudo, ativa os hooks
    python scripts/dev.py test            # pytest (testes de utils/)
    python scripts/dev.py test-notebooks  # pytest --nbmake (executa os notebooks)
    python scripts/dev.py lint            # ruff check + ruff format --check
    python scripts/dev.py format          # ruff format + ruff check --fix
    python scripts/dev.py check           # lint + test
    python scripts/dev.py lab             # jupyter lab
    python scripts/dev.py clean           # remove caches locais
    python scripts/dev.py relatorio [...] # relatorio.md -> PDF (repassa argumentos)

No Linux/macOS o `make` chama este script; no Windows rode-o direto
(`python scripts\\dev.py <comando>`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"


def venv_bin(name: str) -> str:
    """Caminho de um executavel dentro do .venv, resolvendo bin/ vs Scripts/."""
    d = VENV / ("Scripts" if os.name == "nt" else "bin")
    if os.name == "nt":
        exe = d / f"{name}.exe"
        return str(exe if exe.exists() else d / name)
    return str(d / name)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_setup() -> None:
    if not VENV.exists():
        run([sys.executable, "-m", "venv", str(VENV)])
    pip = venv_bin("pip")
    run([pip, "install", "--upgrade", "pip"])
    run([pip, "install", "-r", "requirements.txt"])
    run([pip, "install", "-e", "."])
    run([venv_bin("pre-commit"), "install"])


def cmd_test() -> None:
    run([venv_bin("python"), "-m", "pytest"])


def cmd_test_notebooks() -> None:
    run([venv_bin("python"), "-m", "pytest", "--nbmake", "projetos/"])


def cmd_lint() -> None:
    run([venv_bin("ruff"), "check", "."])
    run([venv_bin("ruff"), "format", "--check", "."])


def cmd_format() -> None:
    run([venv_bin("ruff"), "format", "."])
    run([venv_bin("ruff"), "check", "--fix", "."])


def cmd_check() -> None:
    cmd_lint()
    cmd_test()


def cmd_lab() -> None:
    run([venv_bin("jupyter"), "lab"])


def cmd_relatorio(extra: list[str]) -> None:
    run([venv_bin("python"), str(ROOT / "scripts" / "relatorio_pdf.py"), *extra])


def cmd_clean() -> None:
    caches = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".ruff_cache"}
    for path in ROOT.rglob("*"):
        parts = set(path.parts)
        if path.name in caches and path.is_dir() and ".venv" not in parts and ".git" not in parts:
            shutil.rmtree(path, ignore_errors=True)
            print("removido", path.relative_to(ROOT))
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
        print("removido", egg.relative_to(ROOT))


COMMANDS = {
    "setup": cmd_setup,
    "test": cmd_test,
    "test-notebooks": cmd_test_notebooks,
    "lint": cmd_lint,
    "format": cmd_format,
    "check": cmd_check,
    "lab": cmd_lab,
    "clean": cmd_clean,
    "relatorio": cmd_relatorio,
}

# Comandos que aceitam argumentos extras (repassados adiante).
PASSTHROUGH = {"relatorio"}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("comandos:", ", ".join(COMMANDS))
        return 2
    cmd, rest = argv[0], argv[1:]
    if rest and cmd not in PASSTHROUGH:
        print(f"'{cmd}' nao aceita argumentos extras: {rest}")
        return 2
    try:
        COMMANDS[cmd](rest) if cmd in PASSTHROUGH else COMMANDS[cmd]()
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
