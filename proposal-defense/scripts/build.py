#!/usr/bin/env python3
"""
Build script for proposal defense slides.
Converts markdown slides to HTML presentation and PDF.
"""

import os
import subprocess
import sys
from pathlib import Path

SLIDES_DIR = Path(__file__).parent.parent / "slides"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SCRIPT_DIR = Path(__file__).parent


def combine_slides():
    """Combine individual markdown slides into a single presentation file."""
    slides_files = sorted(SLIDES_DIR.glob("*.md"))

    if not slides_files:
        print("No markdown slides found in", SLIDES_DIR)
        return None

    combined = []
    for slide_file in slides_files:
        print(f"Adding {slide_file.name}...")
        with open(slide_file, "r") as f:
            combined.append(f.read())

    combined_md = "\n\n".join(combined)
    combined_file = OUTPUT_DIR / "presentation.md"

    with open(combined_file, "w") as f:
        f.write(combined_md)

    print(f"Combined slides saved to {combined_file}")
    return combined_file


def build_html(md_file):
    """Build HTML presentation using pandoc with reveal.js."""
    html_file = OUTPUT_DIR / "proposal-defense.html"

    try:
        cmd = [
            "pandoc",
            str(md_file),
            "-t",
            "revealjs",
            "-s",
            "-o",
            str(html_file),
            "--variable",
            "revealjs-url=https://unpkg.com/reveal.js@4.5.0",
            "--variable",
            "transition=slide",
            "--variable",
            "theme=black",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"HTML presentation created: {html_file}")
        return html_file
    except FileNotFoundError:
        print(
            "Warning: pandoc not found. Install with: apt-get install pandoc",
            file=sys.stderr,
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error building HTML: {e.stderr}", file=sys.stderr)
        return None


def build_pdf(md_file):
    """Build PDF presentation using pandoc with LaTeX."""
    pdf_file = OUTPUT_DIR / "proposal-defense.pdf"

    try:
        cmd = [
            "pandoc",
            str(md_file),
            "-t",
            "beamer",
            "-o",
            str(pdf_file),
            "-V",
            "theme:Madrid",
            "-V",
            "aspectratio:16:9",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"PDF presentation created: {pdf_file}")
        return pdf_file
    except FileNotFoundError:
        print(
            "Warning: pandoc not found. Install with: apt-get install pandoc",
            file=sys.stderr,
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error building PDF: {e.stderr}", file=sys.stderr)
        return None


def main():
    """Main build function."""
    print("Building proposal defense artifacts...")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Combine slides
    combined_md = combine_slides()
    if not combined_md:
        print("Failed to combine slides")
        return 1

    # Build HTML
    html_result = build_html(combined_md)

    # Build PDF
    pdf_result = build_pdf(combined_md)

    # Success if at least one format was built, or if slides were combined
    if html_result or pdf_result:
        print("\n✓ Build completed successfully")
        if not html_result:
            print("  Note: HTML presentation not generated (pandoc required)")
        if not pdf_result:
            print("  Note: PDF presentation not generated (pandoc + LaTeX required)")
        return 0
    elif combined_md:
        print("\n⚠ Slides combined but presentation formats require pandoc+LaTeX")
        return 0
    else:
        print("\n✗ Build failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
