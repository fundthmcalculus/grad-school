#!/usr/bin/env python3
"""Render paper.md to PDF.

Same approach as AnalyticalDynamics/chaos/build_pdf.py (this file is its
sibling, not the proposal's): LaTeX math rendered by a real TeX pipeline,
in this order of preference:

  1. pandoc + a LaTeX engine (xelatex / lualatex / pdflatex / tectonic)
     -- the correct, publication-grade path. Full LaTeX support.
  2. pandoc --mathml + WeasyPrint -- works offline with no TeX install; math
     quality depends on the renderer's MathML support.

paper.md is already a single, whole document, so there is no chapter
assembly. Unlike chaos, nothing is stripped -- the TODO markers are live
research notes and stay in the draft render -- but two mechanical defects in
paper.md are fixed here rather than at the source, so the build stays
reproducible while the author decides how to fix the text:

  * ``\\gte`` is a MathJax spelling; LaTeX's is ``\\ge``. xelatex dies with
    "Undefined control sequence" on it (and pandoc passes it through
    untouched, so the failure is at the TeX stage, not the parse stage).
  * The stacked-matrix display in the Derivation section is closed with a
    stray ``##`` line instead of ``$$``. pandoc then leaves the opening
    ``$$`` as literal text and turns the ``##`` into an empty subsection --
    the matrix renders as raw LaTeX source in the PDF. The paper has no
    ``##`` headings (only ``###``), so a bare ``##`` line is unambiguous.

Usage (from the repo root):
    .venv/Scripts/python papers/tribble-base/build_pdf.py

Outputs:
    papers/tribble-base/build/paper-fixed.md
    papers/tribble-base/build/paper.pdf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# paper.md carries em dashes, times signs, Greek letters and degree signs; the
# platform default is cp1252 on Windows, which cannot decode them -- same failure
# shape build_pdf.py documents for the proposal build.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SRC = os.path.join(HERE, "paper.md")

LATEX_ENGINES = ["xelatex", "lualatex", "pdflatex", "tectonic"]

TITLE_BLOCK = r"""---
documentclass: article
papersize: letter
geometry: margin=1in
fontsize: 11pt
linestretch: 1.08
numbersections: false
colorlinks: true
header-includes: |
  \usepackage{xurl}
---

"""


def fix_undefined_macros(md):
    r"""`\gte` is not a LaTeX macro -- it is the MathJax spelling of `\ge`.
    pandoc passes it through into the .tex verbatim, and the TeX engine then
    dies with "Undefined control sequence \gte". The negative lookahead keeps
    a longer macro that happens to start with `\gte` from being clobbered."""
    return re.sub(r"\\gte(?![a-zA-Z])", r"\\ge", md)



def fix_inline_math_spacing(md):
    r"""pandoc's tex_math_dollars requires a non-space character immediately
    after the opening `$` and immediately before the closing `$` for INLINE
    math (display `$$` allows the space). The Derivation section writes
    `$ M \gte R(N+1) $` with spaces inside the delimiters, so pandoc emits
    literal `\$ ... \$` and the TeX engine then dies on `\ge` in text mode
    ("Missing $ inserted") -- which xelatex papers over and still exits 0.
    Strip the spaces just inside single-dollar spans. Pairing is sequential
    over lone `$` delimiters (a regex that just looks for `$ ... $` mis-pairs
    the closing `$` of an earlier span as an opener and eats prose spaces);
    spans crossing a newline are left alone."""
    lone = re.compile(r"(?<!\\)\$(?!\$)")
    marks = list(lone.finditer(md))
    edits = []
    for k in range(0, len(marks) - 1, 2):
        start, end = marks[k].end(), marks[k + 1].start()
        span = md[start:end]
        stripped = span.strip(" ")
        if stripped != span and "\n" not in span:
            edits.append((start, end, stripped))
    for start, end, stripped in reversed(edits):
        md = md[:start] + stripped + md[end:]
    return md

def fix_unclosed_matrix_block(md):
    r"""The stacked-matrix display at the end of the Derivation section is
    closed with a bare `##` line where the author meant `$$`. pandoc's tex_math_dollars
    needs a closing `$$`; without one it leaves the opening `$$` as literal
    text and turns the `##` into an empty `\subsection{}`, so the matrix
    renders as raw LaTeX source. The paper uses only `###` headings, so a
    line that is exactly `##` is unambiguous -- replace it with `$$`."""
    return re.sub(r"(?m)^##[ \t]*$", "$$", md)


def assemble():
    os.makedirs(BUILD, exist_ok=True)
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    fixed = fix_unclosed_matrix_block(
        fix_inline_math_spacing(fix_undefined_macros(md))
    )
    out_path = os.path.join(BUILD, "paper-fixed.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"  wrote {out_path}")
    return out_path


def pandoc_bin():
    if shutil.which("pandoc"):
        return shutil.which("pandoc")
    try:
        import pypandoc

        pypandoc.ensure_pandoc_installed()
        return "pandoc"
    except Exception:
        return None


def find_latex_engine():
    for eng in LATEX_ENGINES:
        if shutil.which(eng):
            return eng
    return None


def build_with_latex(md_path, pandoc, engine):
    pdf = os.path.join(BUILD, "paper.pdf")
    src = os.path.join(BUILD, "paper-titled.md")
    with open(md_path, encoding="utf-8") as f:
        body = f.read()
    with open(src, "w", encoding="utf-8") as f:
        f.write(TITLE_BLOCK + body)

    cmd = [
        pandoc,
        src,
        "-o",
        pdf,
        f"--pdf-engine={engine}",
        "--from",
        "markdown+tex_math_dollars+pipe_tables+fenced_code_blocks+autolink_bare_uris",
        "-V",
        "linkcolor=blue",
        "--wrap=preserve",
        "--resource-path",
        HERE,
    ]
    print(f"  pandoc + {engine} ...")
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:])
        return None
    return pdf


def build_with_weasyprint(md_path, pandoc):
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        print("  [fallback] weasyprint not installed")
        return None

    html_path = os.path.join(BUILD, "paper.html")
    cmd = [
        pandoc,
        md_path,
        "-o",
        html_path,
        "--standalone",
        "--mathml",
        "--from",
        "markdown+tex_math_dollars+pipe_tables",
        "--wrap=preserve",
        "--resource-path",
        HERE,
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if res.returncode != 0:
        print(res.stderr[-1500:])
        return None

    css = CSS(
        string="""
    @page { size: letter; margin: 1in;
            @bottom-center { content: counter(page); font-size: 9.5pt; color:#555; } }
    body { font-family: Georgia, "Times New Roman", serif; font-size: 10.8pt;
           line-height: 1.4; text-align: justify; hyphens: auto; }
    h1 { font-size:17pt; } h2 { font-size:13pt; } h3 { font-size:11.4pt; font-style:italic; }
    table { border-collapse:collapse; width:100%; font-size:9pt; margin:.8em 0 1em; }
    th,td { border-bottom:.6pt solid #ccc; padding:4pt 6pt; vertical-align:top; }
    """
    )
    pdf = os.path.join(BUILD, "paper.pdf")
    doc = HTML(filename=html_path, base_url=HERE).render(stylesheets=[css])
    doc.write_pdf(pdf)
    return pdf, len(doc.pages)


def page_count(pdf):
    try:
        result = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
    except Exception:
        pass
    try:
        with open(pdf, "rb") as f:
            content = f.read()
            matches = re.findall(rb"/Count\s+(\d+)", content)
            if matches:
                return int(matches[0])
    except Exception:
        pass
    return None


def main():
    print("Assembling paper.md ...")
    md_path = assemble()

    pandoc = pandoc_bin()
    if not pandoc:
        sys.exit("pandoc not found. Install it, or: pip install pypandoc-binary")

    engine = find_latex_engine()
    if engine:
        pdf = build_with_latex(md_path, pandoc, engine)
        if pdf:
            n = page_count(pdf)
            print(f"  wrote {pdf}  ({n} pages)  [pandoc + {engine}: full LaTeX]")
            return
        print("  [warn] LaTeX build failed; falling back to WeasyPrint")
    else:
        print("  [note] No LaTeX engine found (xelatex/lualatex/pdflatex/tectonic).")
        print("         Falling back to pandoc --mathml + WeasyPrint.")

    out = build_with_weasyprint(md_path, pandoc)
    if not out:
        sys.exit("PDF build failed.")
    pdf, pages = out
    print(f"  wrote {pdf}  ({pages} pages)  [pandoc --mathml + WeasyPrint]")


if __name__ == "__main__":
    main()
