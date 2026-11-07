#!/usr/bin/env bash
# Build main.pdf. Requires pdflatex and bibtex (texlive-latex-base, texlive-publishers).
# Figures are PDFs in fig/, converted from the EPS the reproduce/figures pipeline writes.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | tail -1
echo "built main.pdf"
