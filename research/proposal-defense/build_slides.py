#!/usr/bin/env python3
"""Build the proposal defense slide deck from `slides/*.md`.

Outputs land in `output/` (gitignored): a self-contained reveal.js HTML
deck and, when a TeX engine is available, a Beamer PDF of the same
content.

The HTML deck vendors its assets so it works offline: `build` downloads
reveal.js and KaTeX into `slides/vendor/` on first use and references
them relatively, so the built file is portable as long as `slides/`
stays next to `output/`. Figures are copied from `prose/fig/` into the
output directory and referenced by name.

Run from anywhere:

    python research/proposal-defense/build_slides.py            # html + pdf
    python research/proposal-defense/build_slides.py --html     # html only
    python research/proposal-defense/build_slides.py --pdf      # pdf only
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SLIDES_DIR = HERE / "slides"
PROSE_FIG_DIR = HERE / "prose" / "fig"
VENDOR_DIR = SLIDES_DIR / "vendor"
OUTPUT_DIR = HERE / "output"

# The deck's reveal.js + KaTeX assets, pinned. First build downloads them;
# later builds reuse the cache. revealjs-url points at the vendored package
# root (the pandoc template expects $revealjs-url$/dist/...), and the KaTeX
# header include is injected via --include-in-header because the pandoc
# revealjs template has no KaTeX hook of its own.
REVEAL_VERSION = "4.5.0"
KATEX_VERSION = "0.16.11"
JSDELIVR = "https://cdn.jsdelivr.net/npm"
VENDOR_FILES = {
    # relpath under VENDOR_DIR -> url
    "revealjs/dist/reset.css": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/dist/reset.css",
    "revealjs/dist/reveal.css": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/dist/reveal.css",
    "revealjs/dist/theme/black.css": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/dist/theme/black.css",
    "revealjs/dist/reveal.js": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/dist/reveal.js",
    "revealjs/plugin/notes/notes.js": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/plugin/notes/notes.js",
    "revealjs/plugin/search/search.js": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/plugin/search/search.js",
    "revealjs/plugin/zoom/zoom.js": f"{JSDELIVR}/reveal.js@{REVEAL_VERSION}/plugin/zoom/zoom.js",
    "katex/katex.min.css": f"{JSDELIVR}/katex@{KATEX_VERSION}/dist/katex.min.css",
    "katex/katex.min.js": f"{JSDELIVR}/katex@{KATEX_VERSION}/dist/katex.min.js",
    "katex/auto-render.min.js": f"{JSDELIVR}/katex@{KATEX_VERSION}/dist/contrib/auto-render.min.js",
}
# KaTeX's CSS references fonts/ relative to itself; pull the woff2 set.
KATEX_FONTS = [
    "KaTeX_AMS-Regular",
    "KaTeX_Caligraphic-Bold",
    "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold",
    "KaTeX_Fraktur-Regular",
    "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic",
    "KaTeX_Main-Italic",
    "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic",
    "KaTeX_Math-Italic",
    "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic",
    "KaTeX_SansSerif-Regular",
    "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular",
    "KaTeX_Size2-Regular",
    "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular",
    "KaTeX_Typewriter-Regular",
]

KATEX_HEADER = """\
<link rel="stylesheet" href="{katex_url}/katex.min.css">
<script src="{katex_url}/katex.min.js" defer></script>
<script src="{katex_url}/auto-render.min.js" defer></script>
<script>
document.addEventListener("DOMContentLoaded", function () {{
  renderMathInElement(document.body, {{
    delimiters: [
      {{left: "$$", right: "$$", display: true}},
      {{left: "$", right: "$", display: false}}
    ]
  }});
}});
</script>
"""

# reveal.js ships with a large default font; shrink headings, tables and the
# timeline grid so slides read like the proposal, not a document.
CUSTOM_CSS = """\
.reveal .slides section { text-align: left; }
.reveal .slides section:first-child { text-align: center; }
.reveal h1 { font-size: 1.7em; }
.reveal h2 { font-size: 1.25em; }
.reveal table { font-size: 0.55em; margin: 0.4em 0; }
.reveal th { font-weight: 600; }
.reveal img { display: block; margin: 0.6em auto 0; max-width: 95%; }
.reveal pre { font-size: 0.42em; width: 100%; overflow-x: auto; }
"""


def _ensure_vendor():
    """Download the pinned reveal.js/KaTeX assets into VENDOR_DIR (idempotent)."""
    missing = [rel for rel in VENDOR_FILES if not (VENDOR_DIR / rel).exists()] + [
        f"katex/fonts/{name}.woff2"
        for name in KATEX_FONTS
        if not (VENDOR_DIR / "katex" / "fonts" / f"{name}.woff2").exists()
    ]
    if not missing:
        return
    for rel in missing:
        if rel in VENDOR_FILES:
            url = VENDOR_FILES[rel]
        else:
            name = rel.rsplit("/", 1)[-1]
            url = f"{JSDELIVR}/katex@{KATEX_VERSION}/dist/fonts/{name}"
        dest = VENDOR_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  vendoring {rel}")
        urllib.request.urlretrieve(url, dest)


def pandoc_bin():
    """System pandoc, else the pypandoc-bundled binary (same contract as build_pdf.py)."""
    found = shutil.which("pandoc")
    if found:
        return found
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except Exception:
        return None


def combine_slides():
    """Concatenate slides/*.md in filename order, one `---` per file boundary."""
    files = sorted(SLIDES_DIR.glob("*.md"))
    if not files:
        print(f"No markdown slides found in {SLIDES_DIR}", file=sys.stderr)
        return None
    for f in files:
        print(f"  + {f.name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = OUTPUT_DIR / "presentation.md"
    combined.write_text(
        "\n\n---\n\n".join(f.read_text(encoding="utf-8").rstrip() for f in files)
        + "\n",
        encoding="utf-8",
    )
    print(f"Combined {len(files)} slides -> {combined}")
    return combined


def copy_figures(md_file):
    """Copy the figures the deck references into OUTPUT_DIR; return any missing."""
    refs = set(
        re.findall(r"!\[[^\]]*\]\(([^)\s]+)", md_file.read_text(encoding="utf-8"))
    )
    missing = []
    for ref in sorted(refs):
        if not ref.startswith("fig/"):
            continue
        src = PROSE_FIG_DIR / ref[len("fig/") :]
        dest = OUTPUT_DIR / ref
        if not src.exists():
            missing.append(ref)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return missing


def build_html(md_file):
    """pandoc -> self-contained reveal.js HTML (assets vendored, offline)."""
    html_file = OUTPUT_DIR / "proposal-defense.html"
    pandoc = pandoc_bin()
    if not pandoc:
        print("pandoc not found; skipping HTML build", file=sys.stderr)
        return None
    _ensure_vendor()
    (OUTPUT_DIR / "katex-header.html").write_text(
        KATEX_HEADER.format(
            katex_url=os.path.relpath(VENDOR_DIR / "katex", OUTPUT_DIR)
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "custom.css").write_text(CUSTOM_CSS, encoding="utf-8")
    reveal_url = os.path.relpath(VENDOR_DIR / "revealjs", OUTPUT_DIR)
    cmd = [
        pandoc,
        str(md_file),
        "-t",
        "revealjs",
        "-s",
        "-o",
        str(html_file),
        "--variable",
        f"revealjs-url={reveal_url}",
        "--variable",
        "transition=slide",
        "--variable",
        "theme=black",
        "--variable",
        "slideNumber=true",
        "--css",
        "custom.css",
        "--include-in-header",
        str(OUTPUT_DIR / "katex-header.html"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building HTML: {e.stderr}", file=sys.stderr)
        return None
    print(f"HTML deck: {html_file}")
    return html_file


def build_pdf(md_file):
    """pandoc -> Beamer PDF (requires a LaTeX engine)."""
    pdf_file = OUTPUT_DIR / "proposal-defense.pdf"
    tex_file = OUTPUT_DIR / "proposal-defense.tex"
    pandoc = pandoc_bin()
    engine = next(
        (e for e in ("xelatex", "lualatex", "pdflatex", "tectonic") if shutil.which(e)),
        None,
    )
    if not pandoc or not engine:
        print("pandoc/LaTeX engine not found; skipping PDF build", file=sys.stderr)
        return None
    cmd = [
        pandoc,
        str(md_file),
        "-t",
        "beamer",
        "--standalone",
        "-o",
        str(tex_file),
        "--resource-path",
        str(OUTPUT_DIR),
        "-V",
        "theme:Madrid",
        "-V",
        "aspectratio:169",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        subprocess.run(
            [engine, "-interaction=nonstopmode", str(tex_file)],
            check=True,
            capture_output=True,
            text=True,
            cwd=OUTPUT_DIR,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building PDF: {e.stderr}", file=sys.stderr)
        return None
    print(f"PDF deck: {pdf_file}")
    return pdf_file


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Build the proposal defense slide deck.")
    ap.add_argument("--html", action="store_true", help="build only the HTML deck")
    ap.add_argument("--pdf", action="store_true", help="build only the Beamer PDF")
    args = ap.parse_args(argv)
    want_html = args.html or not args.pdf
    want_pdf = args.pdf or not args.html

    print("Building proposal defense deck...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = combine_slides()
    if not combined:
        return 1

    missing = copy_figures(combined)
    if missing:
        print(
            f"WARNING: figures not found in {PROSE_FIG_DIR}: {missing}", file=sys.stderr
        )

    html = build_html(combined) if want_html else None
    pdf = build_pdf(combined) if want_pdf else None

    if (want_html and html) or (want_pdf and pdf):
        print("\nBuild completed.")
        return 0
    print("\nSlides combined, but no renderer produced output.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
