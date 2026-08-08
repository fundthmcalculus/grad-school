#!/usr/bin/env python3
"""Render paper.md to PDF.

Same approach as research/proposal-defense/build_pdf.py: LaTeX math rendered by
a real TeX pipeline, in this order of preference:

  1. pandoc + a LaTeX engine (xelatex / lualatex / pdflatex / tectonic)
     -- the correct, publication-grade path. Full LaTeX support.
  2. pandoc --mathml + WeasyPrint -- works offline with no TeX install; math
     quality depends on the renderer's MathML support.

paper.md is already a single, whole document (unlike the proposal's per-chapter
prose/), so there is no chapter assembly here -- just one editorial strip (the
"Paper Action Items" section, which the document itself flags "to be removed
before publication") before handing it to pandoc.

Usage (from chaos/):
    ../../.venv/Scripts/python build_pdf.py

Outputs:
    build/paper-stripped.md
    build/paper.pdf
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


def strip_editorial(md):
    """Drop the "Paper Action Items" section -- the document's own author note
    says it is "to be removed before publication", so a submission render
    shouldn't carry it."""
    return re.sub(
        r"\n### Paper Action Items.*\Z", "\n", md, flags=re.DOTALL
    )


def break_before_urls(md):
    """Force a hard line break before every bare URL.

    Each reference has its DOI line immediately followed by a bare URL line in
    the source, but Markdown joins soft-wrapped lines within a list item into
    one paragraph, so the URL was flowing inline as a continuation of the
    citation text instead of starting its own line. A Markdown hard break is
    two trailing spaces before the newline; this is a rendering-only concern,
    so it is done here rather than by editing paper.md's actual line endings.
    Each URL line in the source is itself indented (list-item continuation,
    e.g. three spaces), which put it after the newline the first version of
    this regex matched against -- `(?=https?://)` right after `\n` never
    matched, so the break was never actually inserted. The whitespace is
    consumed too, so the break lands immediately before the URL text.
    """
    return re.sub(r"(?<=\S)\n[ \t]*(?=https?://)", "  \n", md)


def fix_bold_greek(md):
    """`\\mathbf{\\theta}` and `\\mathbf{\\phi}` render as blank glyphs under
    this xelatex setup's default math font -- \\mathbf has no bold shape for
    a Greek letter here, unlike for a bare Latin letter such as `\\mathbf{x}`.
    `\\boldsymbol` (amsmath) does have one. Only patterns of the form
    `\\mathbf{\\<command>}` are touched, so `\\mathbf{x}`, `\\mathbf{a}_r`, and
    `\\dot{\\mathbf{x}}` are untouched -- \\boldsymbol conflicts with \\dot in
    this setup, so those need to stay exactly as \\mathbf.
    """
    return re.sub(r"\\mathbf\{(\\[a-zA-Z]+)\}", r"\\boldsymbol{\1}", md)


def assemble():
    os.makedirs(BUILD, exist_ok=True)
    with open(SRC, encoding="utf-8") as f:
        md = f.read()
    stripped = fix_bold_greek(break_before_urls(strip_editorial(md)))
    out_path = os.path.join(BUILD, "paper-stripped.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stripped)
    print(f"  wrote {out_path}")
    return out_path


def pandoc_bin():
    if shutil.which("pandoc"):
        return "pandoc"
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
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

    css = CSS(string="""
    @page { size: letter; margin: 1in;
            @bottom-center { content: counter(page); font-size: 9.5pt; color:#555; } }
    body { font-family: Georgia, "Times New Roman", serif; font-size: 10.8pt;
           line-height: 1.4; text-align: justify; hyphens: auto; }
    h1 { font-size:17pt; } h2 { font-size:13pt; } h3 { font-size:11.4pt; font-style:italic; }
    table { border-collapse:collapse; width:100%; font-size:9pt; margin:.8em 0 1em; }
    th,td { border-bottom:.6pt solid #ccc; padding:4pt 6pt; vertical-align:top; }
    """)
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
