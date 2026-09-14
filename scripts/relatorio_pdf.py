#!/usr/bin/env python3
"""Converte um `relatorio.md` em PDF (multiplataforma).

    python scripts/relatorio_pdf.py                      # todos os projetos/**/relatorio.md
    python scripts/relatorio_pdf.py projetos/01_simulacoes/topico1_entropia_mi
    python scripts/relatorio_pdf.py caminho/relatorio.md -o saida.pdf

O pipeline e Markdown -> HTML autocontido -> PDF:

- Markdown por `mistune` (tabelas, listas de tarefas, notas de rodape, ...);
- matematica entre ``$ ... $`` (inline) e ``$$ ... $$`` (bloco) e pre-renderizada
  com o `matplotlib` (mathtext) e embutida como imagem -- funciona offline e em
  qualquer motor de PDF;
- figuras locais (``figuras/foo.png``) sao embutidas em base64, entao o HTML
  intermediario nao depende de arquivos externos;
- codigo com realce de sintaxe via `pygments`, se instalado.

Motores de PDF, tentados nesta ordem com ``--engine auto``:

1. `weasyprint`, se importavel (``pip install weasyprint``) -- inclui numero de pagina;
2. um navegador Chromium headless (Google Chrome / Microsoft Edge / Chromium)
   via ``--headless --print-to-pdf``;
3. se nenhum estiver disponivel, grava so o ``.html`` ao lado e explica como
   imprimir para PDF pelo navegador.
"""

from __future__ import annotations

import argparse
import base64
import io
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402  (precisa vir depois de use("Agg"))
from matplotlib.font_manager import FontProperties  # noqa: E402

try:
    import mistune
except ImportError:  # pragma: no cover - depende do ambiente
    sys.exit("erro: falta o pacote 'mistune' (rode 'make setup' ou 'pip install mistune').")

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    _PYGMENTS = True
except ImportError:  # pragma: no cover
    _PYGMENTS = False

ROOT = Path(__file__).resolve().parents[1]

_DPI = 200
_SCALE = _DPI / 96.0  # px do matplotlib -> px de CSS

DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_RE = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?!\d)")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)

# O mathtext do matplotlib cobre um subconjunto do LaTeX. Estes apelidos comuns
# nao existem la, mas tem equivalente direto -- traduzimos antes de renderizar.
_TEX_ALIASES = {
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\to": r"\rightarrow",
    r"\implies": r"\Rightarrow",
    r"\iff": r"\Leftrightarrow",
    r"\|": r"\parallel",
    r"\text": r"\mathrm",
    r"\eqref": r"\ref",
}
_TEX_ALIAS_RE = re.compile(
    "|".join(re.escape(k) + r"(?![a-zA-Z])" for k in sorted(_TEX_ALIASES, key=len, reverse=True))
)


def warn(msg: str) -> None:
    print(f"  aviso: {msg}", file=sys.stderr)


def _rel(path: Path) -> str:
    for base in (Path.cwd(), ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


# --------------------------------------------------------------------------- #
# Matematica -> imagem                                                         #
# --------------------------------------------------------------------------- #


def _png_size(data: bytes) -> tuple[int, int]:
    """Largura e altura de um PNG lidas direto do cabecalho IHDR."""
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _normalize_tex(expr: str) -> str:
    return _TEX_ALIAS_RE.sub(lambda m: _TEX_ALIASES[m.group(0)], expr)


def _render_math(expr: str, display: bool) -> str:
    expr = _normalize_tex(expr.strip())
    size = 16 if display else 11
    buf = io.BytesIO()
    try:
        try:
            mathtext.math_to_image(
                f"${expr}$",
                buf,
                prop=FontProperties(size=size),
                dpi=_DPI,
                format="png",
                color="#1c1c1c",
            )
        except TypeError:  # matplotlib < 3.7 nao aceita color=
            mathtext.math_to_image(
                f"${expr}$", buf, prop=FontProperties(size=size), dpi=_DPI, format="png"
            )
    except (ValueError, RuntimeError) as exc:
        warn(f"expressao matematica nao reconhecida: {expr!r} ({exc})")
        return f'<code class="math-fallback">{mistune.util.escape(expr)}</code>'

    data = buf.getvalue()
    w, h = _png_size(data)
    cw, ch = w / _SCALE, h / _SCALE
    src = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    alt = mistune.util.escape(expr)
    if display:
        return f'<img src="{src}" style="width:{cw:.1f}px;height:{ch:.1f}px" alt="{alt}">'
    shift = ch * 0.22
    return (
        f'<img class="math-inline" src="{src}" alt="{alt}" '
        f'style="width:{cw:.1f}px;height:{ch:.1f}px;vertical-align:-{shift:.1f}px">'
    )


# --------------------------------------------------------------------------- #
# Pre-processamento do Markdown                                                #
# --------------------------------------------------------------------------- #


def _protect_code(text: str) -> tuple[str, list[str]]:
    """Troca blocos de codigo cercados por marcadores, para o passo de math nao os tocar."""
    out: list[str] = []
    store: list[str] = []
    buf: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            buf = [line]
        elif fence is None:
            out.append(line)
        else:
            buf.append(line)
            if stripped.startswith(fence):
                out.append(f"@@CODE{len(store)}@@")
                store.append("\n".join(buf))
                fence = None
                buf = []
    if buf:  # cerca nao fechada: devolve as linhas como estavam
        out.extend(buf)
    return "\n".join(out), store


def _restore_code(text: str, store: list[str]) -> str:
    for i, block in enumerate(store):
        text = text.replace(f"@@CODE{i}@@", block)
    return text


def _extract_math(text: str) -> tuple[str, list[str], list[bool]]:
    frags: list[str] = []
    display: list[bool] = []

    def _add(expr: str, is_disp: bool) -> str:
        idx = len(frags)
        frags.append(_render_math(expr, is_disp))
        display.append(is_disp)
        return f"@@MATH{idx}@@"

    text = DISPLAY_RE.sub(lambda m: _add(m.group(1), True), text)
    text = INLINE_RE.sub(lambda m: _add(m.group(1), False), text)
    return text, frags, display


def _inject_math(html: str, frags: list[str], display: list[bool]) -> str:
    def _block(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if display[idx]:
            return f'<div class="math-display">{frags[idx]}</div>'
        return f"<p>{frags[idx]}</p>"

    html = re.sub(r"<p>\s*@@MATH(\d+)@@\s*</p>", _block, html)
    for i, frag in enumerate(frags):
        html = html.replace(f"@@MATH{i}@@", frag)
    return html


# --------------------------------------------------------------------------- #
# Renderer do mistune                                                          #
# --------------------------------------------------------------------------- #


class ReportRenderer(mistune.HTMLRenderer):
    """HTML do mistune + realce de codigo e figuras embutidas em base64."""

    def __init__(self, base_dir: Path) -> None:
        super().__init__()
        self.base_dir = base_dir
        self._fmt = HtmlFormatter(cssclass="highlight") if _PYGMENTS else None

    def block_code(self, code: str, info: str | None = None) -> str:
        if self._fmt is not None and info:
            lang = info.strip().split(None, 1)[0]
            try:
                lexer = get_lexer_by_name(lang)
            except ClassNotFound:
                lexer = None
            if lexer is not None:
                return highlight(code, lexer, self._fmt)
        return f'<pre class="highlight"><code>{mistune.util.escape(code)}</code></pre>\n'

    def image(self, text: str, url: str, title: str | None = None) -> str:
        src = self._as_data_uri(url)
        attr_t = f' title="{mistune.util.escape(title)}"' if title else ""
        return f'<img src="{src}" alt="{mistune.util.escape(text or "")}"{attr_t}>'

    def _as_data_uri(self, url: str) -> str:
        if url.startswith(("http://", "https://", "data:")):
            return url
        path = (self.base_dir / url).expanduser()
        try:
            raw = path.read_bytes()
        except OSError:
            warn(f"figura nao encontrada: {url}")
            return mistune.util.escape(url)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------- #
# Montagem do HTML                                                             #
# --------------------------------------------------------------------------- #

_CSS = """
:root {
  --ink: #1c1c1c; --muted: #555; --rule: #d5d5dd; --accent: #2f2a6b; --soft: #f4f4f8;
}
* { box-sizing: border-box; }
@page {
  size: A4;
  margin: 20mm 18mm 22mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font: 9pt Georgia, serif; color: #8a8a8a;
  }
  @top-right {
    content: "MAE0552 \\2014 Teoria da Informa\\E7\\E3o";
    font: 8.5pt Georgia, serif; color: #b0b0b0;
  }
}
html { font-size: 11pt; }
body {
  margin: 0; color: var(--ink); line-height: 1.5;
  font-family: Georgia, "Times New Roman", "DejaVu Serif", serif;
}
.relatorio { max-width: 40em; margin: 0 auto; }
h1, h2, h3, h4 {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.25; break-after: avoid;
}
h1 {
  font-size: 1.7rem; margin: 0 0 .8rem;
  padding-bottom: .35rem; border-bottom: 2px solid var(--accent);
}
h2 { font-size: 1.25rem; margin: 1.8rem 0 .6rem; }
h3 { font-size: 1.05rem; margin: 1.3rem 0 .4rem; }
h4 { font-size: 1rem; margin: 1.1rem 0 .3rem; color: var(--muted); }
p { margin: 0 0 .7rem; }
a { color: var(--accent); }
ul, ol { margin: 0 0 .7rem 1.4rem; padding: 0; }
li { margin: .15rem 0; }
strong { color: #000; }
blockquote {
  margin: .8rem 0; padding: .1rem 0 .1rem .9rem;
  border-left: 3px solid var(--rule); color: var(--muted);
}
code {
  font-family: "SF Mono", "Cascadia Code", "DejaVu Sans Mono", Consolas, monospace;
  font-size: .86em; background: var(--soft); padding: .08em .34em; border-radius: 3px;
}
pre, .highlight {
  background: var(--soft); border: 1px solid var(--rule); border-radius: 5px;
  padding: .7rem .85rem; overflow-x: auto; break-inside: avoid; margin: 0 0 .8rem;
}
pre code, .highlight pre {
  background: none; border: 0; padding: 0; font-size: .82em; line-height: 1.45;
}
.highlight pre { margin: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 .9rem;
  font-size: .9em; break-inside: avoid;
}
th, td {
  border: 1px solid var(--rule); padding: .35rem .55rem;
  text-align: left; vertical-align: top;
}
thead th { background: var(--soft); }
tbody tr:nth-child(even) td { background: #fafafb; }
img { max-width: 100%; }
p > img:not(.math-inline) { display: block; margin: 1rem auto; }
.math-inline { display: inline; }
.math-display { text-align: center; margin: 1rem 0; break-inside: avoid; }
.math-display img { max-width: 100%; }
.math-fallback { color: #a1381f; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.5rem 0; }
""".strip()


def _title_from(md_text: str, fallback: str) -> str:
    m = HEADING_RE.search(md_text)
    return m.group(1).strip() if m else fallback


def build_html(md_text: str, md_path: Path) -> str:
    protected, code_store = _protect_code(md_text)
    protected, frags, display = _extract_math(protected)
    source = _restore_code(protected, code_store)

    md = mistune.create_markdown(
        renderer=ReportRenderer(md_path.parent),
        plugins=["table", "strikethrough", "task_lists", "footnotes", "def_list", "url"],
    )
    body = _inject_math(md(source), frags, display)

    css = _CSS
    if _PYGMENTS:
        css += "\n" + HtmlFormatter(cssclass="highlight").get_style_defs(".highlight")
    title = mistune.util.escape(_title_from(md_text, md_path.stem))
    return (
        '<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>\n{css}\n</style>\n</head>\n"
        f'<body>\n<main class="relatorio">\n{body}\n</main>\n</body>\n</html>\n'
    )


# --------------------------------------------------------------------------- #
# Motores de PDF                                                               #
# --------------------------------------------------------------------------- #

_BROWSER_NAMES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
    "microsoft-edge-stable",
    "msedge",
]


def _find_browser() -> str | None:
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    elif os.name == "nt":
        roots = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for root in filter(None, roots):
            candidates.append(str(Path(root) / "Google/Chrome/Application/chrome.exe"))
            candidates.append(str(Path(root) / "Microsoft/Edge/Application/msedge.exe"))
    return next((c for c in candidates if Path(c).exists()), None)


def _engine_weasyprint(html: str, out: Path, base: Path) -> bool:
    try:
        from weasyprint import HTML
    except ImportError:
        return False
    HTML(string=html, base_url=str(base)).write_pdf(str(out))
    return out.exists() and out.stat().st_size > 0


def _engine_browser(html: str, out: Path, keep_html: bool) -> bool:
    browser = _find_browser()
    if not browser:
        return False
    html_path = out.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    stderr = ""
    try:
        for headless in ("--headless=new", "--headless"):
            if out.exists():
                out.unlink()
            try:
                proc = subprocess.run(
                    [
                        browser,
                        headless,
                        "--disable-gpu",
                        "--no-sandbox",
                        "--no-pdf-header-footer",
                        "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=15000",
                        f"--print-to-pdf={out}",
                        html_path.as_uri(),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                stderr = "tempo esgotado ao gerar o PDF"
                continue
            stderr = (proc.stderr or "").strip()
            if out.exists() and out.stat().st_size > 0:
                return True
    finally:
        if not keep_html:
            html_path.unlink(missing_ok=True)
    if stderr:
        warn(stderr.splitlines()[-1][:300])
    return False


def _write_html(html: str, out: Path) -> Path:
    path = out.with_suffix(".html")
    path.write_text(html, encoding="utf-8")
    return path


def convert_one(md_path: Path, saida: Path | None, engine: str, keep_html: bool) -> str:
    """Devolve 'pdf', 'html' ou 'fail'."""
    try:
        html = build_html(md_path.read_text(encoding="utf-8"), md_path)
    except Exception as exc:  # o pipeline pode falhar de varias formas
        warn(f"falha ao processar {md_path.name}: {exc}")
        return "fail"

    out = saida or md_path.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    if engine == "html":
        print(f"  html  {_rel(_write_html(html, out))}")
        return "html"

    plano = {
        "auto": ("weasyprint", "browser"),
        "weasyprint": ("weasyprint",),
        "chrome": ("browser",),
    }[engine]
    for eng in plano:
        try:
            if eng == "weasyprint":
                done = _engine_weasyprint(html, out, md_path.parent)
            else:
                done = _engine_browser(html, out, keep_html)
        except Exception as exc:
            warn(f"{eng}: {exc}")
            done = False
        if done:
            print(f"  pdf   {_rel(out)}  ({eng})")
            return "pdf"

    path = _write_html(html, out)
    warn(f"nenhum motor de PDF disponivel; gerado HTML: {_rel(path)}")
    print("  dica: 'pip install weasyprint' ou instale o Google Chrome / Microsoft Edge;")
    print("        ou abra o HTML e use 'Imprimir' -> 'Salvar como PDF'.")
    return "html"


# --------------------------------------------------------------------------- #
# Alvos e CLI                                                                  #
# --------------------------------------------------------------------------- #


def resolve_targets(alvo: str | None) -> list[Path]:
    if alvo is None:
        achados = sorted(ROOT.glob("projetos/**/relatorio.md"))
        return [p for p in achados if "_template" not in p.parts]
    path = Path(alvo).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir():
        cand = path / "relatorio.md"
        if not cand.exists():
            sys.exit(f"erro: nao ha relatorio.md em {alvo}")
        return [cand]
    if path.is_file():
        return [path]
    sys.exit(f"erro: caminho inexistente: {alvo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relatorio_pdf",
        description="Converte um relatorio.md em PDF (Markdown + matematica + figuras).",
    )
    parser.add_argument(
        "alvo",
        nargs="?",
        help="arquivo .md, pasta do topico, ou vazio para todos os projetos/**/relatorio.md",
    )
    parser.add_argument("-o", "--saida", type=Path, help="caminho do PDF (apenas com um alvo)")
    parser.add_argument(
        "--engine",
        choices=("auto", "weasyprint", "chrome", "html"),
        default="auto",
        help="motor de PDF (padrao: auto)",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="mantem o HTML intermediario ao lado do PDF",
    )
    args = parser.parse_args(argv)

    targets = resolve_targets(args.alvo)
    if not targets:
        print("nada a converter: nenhum relatorio.md encontrado em projetos/.")
        return 0
    if args.saida is not None and len(targets) > 1:
        parser.error("-o/--saida so funciona com um unico alvo")

    resultados = []
    for md_path in targets:
        print(_rel(md_path))
        resultados.append(convert_one(md_path, args.saida, args.engine, args.keep_html))

    gerados = resultados.count("pdf")
    print(f"\n{gerados}/{len(resultados)} PDF(s) gerado(s).")
    if "fail" in resultados:
        return 1
    if gerados == 0 and args.engine != "html":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
