from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import build_pdf


@pytest.fixture
def output_paths(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    output_pdf = output_dir / "resume.pdf"
    monkeypatch.setattr(build_pdf, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_pdf, "OUTPUT_PDF", output_pdf)
    return output_dir, output_pdf


def test_missing_pandoc_leaves_existing_pdf_untouched(
    output_paths, monkeypatch, capsys
):
    _, output_pdf = output_paths
    output_pdf.write_bytes(b"previous pdf")
    monkeypatch.setattr(build_pdf, "pandoc_bin", lambda: None)

    assert build_pdf.main() == 1
    assert output_pdf.read_bytes() == b"previous pdf"
    assert "Pandoc not found" in capsys.readouterr().err


def test_missing_tex_engine_leaves_existing_pdf_untouched(
    output_paths, monkeypatch, capsys
):
    _, output_pdf = output_paths
    output_pdf.write_bytes(b"previous pdf")
    monkeypatch.setattr(build_pdf, "pandoc_bin", lambda: "pandoc")
    monkeypatch.setattr(build_pdf, "latex_engine", lambda: None)

    assert build_pdf.main() == 1
    assert output_pdf.read_bytes() == b"previous pdf"
    assert "No TeX engine found" in capsys.readouterr().err


def test_renderer_failure_leaves_existing_pdf_untouched(
    output_paths, monkeypatch, capsys
):
    _, output_pdf = output_paths
    output_pdf.write_bytes(b"previous pdf")
    monkeypatch.setattr(build_pdf, "pandoc_bin", lambda: "pandoc")
    monkeypatch.setattr(build_pdf, "latex_engine", lambda: "/usr/bin/xelatex")
    monkeypatch.setattr(
        build_pdf.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0], stderr="renderer failed")
        ),
    )

    assert build_pdf.main() == 1
    assert output_pdf.read_bytes() == b"previous pdf"
    assert "renderer failed" in capsys.readouterr().err


def _renderer_available():
    return bool(build_pdf.pandoc_bin() and build_pdf.latex_engine())


@pytest.mark.skipif(
    not (_renderer_available() and shutil.which("pdftotext")),
    reason="Pandoc, a TeX engine, and pdftotext are required for the real-renderer check",
)
def test_builds_searchable_pdf_when_invoked_from_another_directory(tmp_path):
    script = Path(build_pdf.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    pdf = script.parent / "outputs" / "resume.pdf"
    assert "PDF written to" in result.stdout
    assert pdf.read_bytes().startswith(b"%PDF-")
    extracted = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert "[Full Name]" in extracted
    assert "Professional Summary" in extracted
