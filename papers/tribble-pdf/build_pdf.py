#!/usr/bin/env python3
"""Generic pandoc -> PDF converter for paper.md documents.

Single shared converter used by any paper that needs a rendered PDF
(e.g. papers/tribble-base/paper.md and AnalyticalDynamics/chaos/paper.md).

LaTeX math is rendered by a real TeX pipeline (pandoc + xelatex/lualatex/
pdflatex/tectonic, with pandoc --mathml + WeasyPrint fallback), same as the
per-paper converters previously living in each paper directory.

paper.md is never modified in place; a preprocessed copy is written to the
output build directory, plus paper-titled.md and paper.pdf.

Usage (from repo root):
    python papers/tribble-pdf/build_pdf.py --paper <path to paper.md>

Options:
    --paper PATH         path to paper.md (required)
    --title TEXT         override the document title (else parsed from paper.md)
    --author TEXT        override the author line (else parsed from paper.md)
    --out-dir DIR        output directory (default: papers/tribble-pdf/<paper_dir>/build)
    --engine ENG         override LaTeX engine
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

# paper.md carries em dashes, times signs, Greek letters and degree signs; the
# platform default is cp1252 on Windows, which cannot decode them -- same failure
# shape the per-paper converters document for the proposal build.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LATEX_ENGINES = ["xelatex", "lualatex", "pdflatex", "tectonic"]


def yaml_scalar(s):
    """Return s as a YAML double-quoted scalar (safe for title/author)."""
    s = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return '"' + s + '"'


def title_block(title, author):
    """YAML front matter: documentclass + title + author for a standalone article."""
    title_yaml = yaml_scalar(title)
    author_yaml = yaml_scalar("\n".join(author))
    return f"""---
documentclass: article
papersize: letter
geometry: margin=1in
fontsize: 11pt
linestretch: 1.08
numbersections: false
colorlinks: true
header-includes: |
  \\usepackage{{xurl}}
title: {title_yaml}
author: {author_yaml}
---

"""


def strip_editorial(md):
    """Drop the 'Paper Action Items' draft section -- the document itself flags
    it 'to be removed before publication', so a submission render shouldn't carry
    it. No-op if absent."""
    return re.sub(r"\n### Paper Action Items.*\Z", "\n", md, flags=re.DOTALL)


def fix_bold_greek(md):
    """`\\mathbf{\\theta}` and `\\mathbf{\\phi}` render as blank glyphs under
    the xelatex math font -- `\\mathbf` has no bold shape for a Greek letter
    here, unlike a bare Latin letter such as `\\mathbf{x}`. `\\boldsymbol` (amsmath)
    does have one. Only `\\mathbf{<command>}` patterns are touched, so
    `\\mathbf{x}`, `\\mathbf{a}_r` and `\\dot{\\mathbf{x}}` (which conflict with
    `\\dot` in this setup) are untouched."""
    return re.sub(r"\\mathbf\{(\\[a-zA-Z]+)\}", r"\\boldsymbol{\1}", md)


def break_before_urls(md):
    """Force a hard line break before every bare URL (references list)."""
    return re.sub(r"(?<=\S)\n[ \t]*(?=https?://)", "  \n", md)


def fix_undefined_macros(md):
    r"""`\gte` is a MathJax spelling; LaTeX's is `\ge`. pandoc passes it through
    into the .tex verbatim and the TeX engine dies with 'Undefined control
    sequence \gte' (which xelatex papers over and still exits 0). The
    negative lookahead keeps a longer macro that happens to start with `\gte`
    from being clobbered."""
    return re.sub(r"\\gte(?![a-zA-Z])", r"\\ge", md)


def fix_unclosed_matrix_block(md):
    r"""A display block closed with a stray `##` line instead of `$$` (pandoc
    leaves the opening `$$` literal and turns `##` into an empty
    `\subsection{}`, so the matrix renders as raw LaTeX source). The papers use
    only `###` headings, so a bare `##` line is unambiguous -- replace it with
    `$$`."""
    return re.sub(r"(?m)^##[ \t]*$", "$$", md)


def fix_inline_math_spacing(md):
    r"""pandoc's tex_math_dollars requires a non-space character immediately
    after the opening `$` and before the closing `$` for INLINE math (display
    `$$` allows the space). Spaces inside single-dollar spans are stripped.
    Lone `$` delimiters are paired sequentially (a regex that matches `$ ... $`
    mis-pairs the closing `$` of an earlier span as an opener and eats prose
    spaces). Spans crossing a newline are left alone."""
    lone = re.compile(r"(?<![$\\])\$(?![$\\])")
    marks = list(lone.finditer(md))
    edits = []
    for k in range(0, len(marks) - 1, 2):
        start, end = marks[k].end(), marks[k + 1].start()
        span = md[start:end]
        stripped = span.strip()
        if stripped != span and "\n" not in span:
            edits.append((start, end, stripped))
    for start, end, stripped in reversed(edits):
        md = md[:start] + stripped + md[end:]
    return md


def preprocess(md):
    md = strip_editorial(md)
    md = fix_bold_greek(md)
    md = break_before_urls(md)
    md = fix_undefined_macros(md)
    md = fix_unclosed_matrix_block(md)
    md = fix_inline_math_spacing(md)
    return md


def parse_meta(md):
    """Extract (title, author_lines) from paper.md."""
    title = None
    lines = md.splitlines()
    for line in lines:
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            title = m.group(1).strip()
            break
    first_h2 = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            first_h2 = i
            break
    author = []
    if first_h2 is not None:
        for line in lines[:first_h2]:
            if line.startswith("#") or not line.strip():
                continue
            author.append(line.strip())
    return title, author


def resolve_paper(paper_arg):
    if os.path.isabs(paper_arg):
        return os.path.abspath(paper_arg)
    return os.path.join(REPO_ROOT, paper_arg)


def default_out_dir(paper_path):
    paper_dir = os.path.dirname(os.path.abspath(paper_path))
    return os.path.join(HERE, paper_dir, "build")


def pandoc_bin():
    if shutil.which("pandoc"):
        return shutil.which("pandoc")
    try:
        import pypandoc

        pypandoc.ensure_pandoc_installed()
        return "pandoc"
    except Exception:
        return None


def find_latex_engine(engines):
    for eng in engines:
        if shutil.which(eng):
            return eng
    return None


def build_with_latex(body, pandoc, engine, out_dir, title, author, resource_path):
    pdf = os.path.join(out_dir, "paper.pdf")
    src = os.path.join(out_dir, "paper-titled.md")
    with open(src, "w", encoding="utf-8") as f:
        f.write(title_block(title, author) + body)

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
        resource_path,
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


def build_with_weasyprint(body, pandoc, out_dir, title, author, resource_path):
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        print("  [fallback] weasyprint not installed")
        return None

    html_path = os.path.join(out_dir, "paper.html")
    with open(os.path.join(out_dir, "paper-body.md"), "w", encoding="utf-8") as f:
        f.write(body)

    cmd = [
        pandoc,
        os.path.join(out_dir, "paper-body.md"),
        "-o",
        html_path,
        "--standalone",
        "--mathml",
        "--from",
        "markdown+tex_math_dollars+pipe_tables",
        "--wrap=preserve",
        "--resource-path",
        resource_path,
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
    pdf = os.path.join(out_dir, "paper.pdf")
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
    parser = argparse.ArgumentParser(
        description="Render paper.md to PDF (pandoc + LaTeX engine, WeasyPrint fallback)."
    )
    parser.add_argument("--paper", required=True, help="path to paper.md")
    parser.add_argument("--title", default=None, help="override document title")
    parser.add_argument("--author", default=None, help="override author line")
    parser.add_argument("--out-dir", default=None, help="output directory")
    parser.add_argument("--engine", default=None, help="override LaTeX engine")
    args = parser.parse_args()

    paper_path = resolve_paper(args.paper)
    paper_dir = os.path.dirname(os.path.abspath(paper_path))
    out_dir = args.out_dir or default_out_dir(paper_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(paper_path, encoding="utf-8") as f:
        md = f.read()

    title, author = parse_meta(md)
    if args.title:
        title = args.title
    if args.author:
        author = args.author.splitlines()

    fixed = preprocess(md)
    with open(os.path.join(out_dir, "paper-fixed.md"), "w", encoding="utf-8") as f:
        f.write(fixed)

    pandoc = pandoc_bin()
    if not pandoc:
        sys.exit("pandoc not found. Install it, or: pip install pypandoc-binary")

    engines = [args.engine] if args.engine else LATEX_ENGINES
    engine = find_latex_engine(engines)
    if engine:
        pdf = build_with_latex(fixed, pandoc, engine, out_dir, title, author, paper_dir)
        if pdf:
            n = page_count(pdf)
            print(f"  wrote {pdf}  ({n} pages)  [pandoc + {engine}: full LaTeX]")
            return
        print("  [warn] LaTeX build failed; falling back to WeasyPrint")
    else:
        print("  [note] No LaTeX engine found (xelatex/lualatex/pdflatex/tectonic).")
        print("         Falling back to pandoc --mathml + WeasyPrint.")

    out = build_with_weasyprint(fixed, pandoc, out_dir, title, author, paper_dir)
    if not out:
        sys.exit("PDF build failed.")
    pdf, pages = out
    print(f"  wrote {pdf}  ({pages} pages)  [pandoc --mathml + WeasyPrint]")


if __name__ == "__main__":
    main()
