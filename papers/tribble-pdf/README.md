# Shared PDF Build Converter

Single top-level converter used by every paper that needs a rendered PDF from
`paper.md` (pandoc + LaTeX engine, with WeasyPrint fallback).

## What it does

Renders `paper.md` to PDF via a real TeX pipeline:

1. pandoc + xelatex / lualatex / pdflatex / tectonic (preferred; full LaTeX)
2. pandoc --mathml + WeasyPrint (offline fallback; math quality depends on the
   renderer's MathML support)

The converter applies standardized preprocessing fixes to `paper.md` before
rendering. The papers carry known defects that break a naive pandoc/LaTeX
build; each fix is documented in the function docstring:

- `fix_undefined_macros`: `\gte` (MathJax spelling) → `\ge` (LaTeX), so xelatex
  does not die with "Undefined control sequence" (which xelatex papers over
  and still exits 0).
- `fix_unclosed_matrix_block`: a stray `##` line that closes a display block
  with `$$` (pandoc leaves the opening `$$` literal and turns `##` into an
  empty `\subsection{}`).
- `fix_inline_math_spacing`: strips spaces inside inline `$...$` delimiters
  (pandoc requires a non-space character immediately after the opening `$` and
  before the closing `$` for inline math).
- `strip_editorial`: removes the "Paper Action Items" draft section flagged by
  the document for publication.
- `fix_bold_greek`: `\mathbf{\theta}` → `\boldsymbol\theta` (bold Greek renders
  as blank glyphs under the xelatex math font).
- `break_before_urls`: forces a line break before bare URLs in reference lists.

## Usage

From the repo root:

```
python papers/tribble-pdf/build_pdf.py --paper <path to paper.md>
```

Options:

- `--paper PATH` (required): path to `paper.md`.
- `--title TEXT`: override the document title (else parsed from `paper.md`).
- `--author TEXT`: override the author line (else parsed from `paper.md`).
- `--out-dir DIR`: output directory (default: `build/` beside `paper.md`).
- `--engine ENG`: override the LaTeX engine.

Output (in `--out-dir`, default):

- `paper-fixed.md` — preprocessed paper body
- `paper-titled.md` — body with title/author front matter
- `paper.pdf` — rendered PDF

## Papers using this converter

- `papers/tribble-base/paper.md`
- `AnalyticalDynamics/chaos/paper.md`

Each paper's `build_pdf.py` is a thin wrapper that delegates to this script.
