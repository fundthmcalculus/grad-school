#!/usr/bin/env python3
"""Render resume.md to a clean PDF with Pandoc and a TeX engine."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "resume.md"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_PDF = OUTPUT_DIR / "resume.pdf"
LATEX_ENGINES = ("xelatex", "lualatex", "pdflatex", "tectonic")


def pandoc_bin():
    """Find system Pandoc, or a binary provided by pypandoc."""
    found = shutil.which("pandoc")
    if found:
        return found
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except Exception:
        return None


def latex_engine():
    """Return the first supported TeX engine available on PATH."""
    return next(
        (shutil.which(engine) for engine in LATEX_ENGINES if shutil.which(engine)), None
    )


def build_pdf():
    """Build the PDF and atomically replace the previous output on success."""
    pandoc = pandoc_bin()
    engine = latex_engine()
    if not pandoc:
        raise RuntimeError("Pandoc not found. Install Pandoc or the pypandoc package.")
    if not engine:
        raise RuntimeError(
            "No TeX engine found. Install one of: " + ", ".join(LATEX_ENGINES)
        )
    if not SOURCE.is_file():
        raise RuntimeError(f"Resume source not found: {SOURCE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".resume-build-", dir=OUTPUT_DIR) as tmp:
        build_dir = Path(tmp)
        tex_file = build_dir / "resume.tex"
        pandoc_cmd = [
            pandoc,
            str(SOURCE),
            "--from=markdown",
            "--to=latex",
            "--standalone",
            "--output",
            str(tex_file),
            "--resource-path",
            str(HERE),
            "-V",
            "geometry:margin=0.65in",
            "-V",
            "fontsize=10pt",
            "-V",
            "colorlinks=true",
            "-V",
            "urlcolor=blue",
        ]
        subprocess.run(pandoc_cmd, check=True, capture_output=True, text=True)

        if Path(engine).name == "tectonic":
            engine_cmd = [engine, "--outdir", str(build_dir), str(tex_file)]
        else:
            engine_cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(build_dir),
                str(tex_file),
            ]
        subprocess.run(engine_cmd, check=True, capture_output=True, text=True)

        built_pdf = build_dir / "resume.pdf"
        if not built_pdf.is_file() or not built_pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("The renderer completed without producing a valid PDF.")
        os.replace(built_pdf, OUTPUT_PDF)

    return OUTPUT_PDF


def main():
    try:
        pdf_path = build_pdf()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        message = detail.strip() if detail else str(exc)
        print(f"PDF build failed: {message}", file=sys.stderr)
        return 1
    print(f"PDF written to {pdf_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
