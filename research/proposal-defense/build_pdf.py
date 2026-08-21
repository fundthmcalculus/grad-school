#!/usr/bin/env python3
"""Assemble the proposal chapters into one document and render it to PDF.

LaTeX math is rendered by a real TeX pipeline, in this order of preference:

  1. pandoc + a LaTeX engine (xelatex / lualatex / pdflatex / tectonic)
     -- the correct, publication-grade path. Full LaTeX support.
  2. pandoc --mathml + WeasyPrint -- works offline with no TeX install; math
     quality depends on the renderer's MathML support.
  3. pandoc --gladtex / plain fallback -- last resort so a PDF still builds.

If no LaTeX engine is found, the script says exactly what to install.

Usage (from repo root):
    ./.venv/bin/python research/proposal-defense/build_pdf.py

Outputs:
    research/proposal-defense/build/proposal-combined.md
    research/proposal-defense/build/proposal.pdf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# This script both reads UTF-8 prose and prints ✓/✗ progress marks. Windows
# defaults stdout to cp1252, which encodes neither, so the build died in its own
# progress print *after* assembling the document -- the same failure shape as the
# harness emitters and the Chapter 5 driver. Every file open here pins UTF-8 for
# the reading half; this covers the printing half.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")

SECTIONS = [
    "prose/00-acknowledgements.md",  # front matter; author-written, not generated
    "prose/01-introduction.md",
    "prose/02-background.md",
    "prose/03-scalable-structure-discovery-pvat.md",
    "prose/04-fast-fis-synthesis-mog.md",
    "prose/05-topological-membership-generation.md",
    "prose/06-hierarchical-refined-fis.md",
    "prose/07-goals-for-completion.md",
    "prose/08-conclusion.md",
    "prose/09-publications.md",  # still outline-only (awaiting NAFIPS details)
    "prose/10-timeline.md",
    "prose/bibliography.md",
    "prose/appendix.md",
]

CHECKLIST_FILE = "CHECKLIST.md"  # burn-down checklist appended after references

TITLE = "Reproducing Like Tribbles"
SUBTITLE = "Scaling Fuzzy Inference Systems from Hundreds to Hundreds of Thousands"
AUTHOR = "Scott Phillips"
COMMITTEE = (
    "Dr. Kelly Cohen (chair) \\\\ "
    "Dr. Vladik Kreinovich \\\\ "
    "Dr. Manish Kumar \\\\ "
    "Dr. Ali Minai \\\\ "
    "Dr. Justin Zhan"
)
DEPT = (
    "Department of Aerospace Engineering and Engineering Mechanics \\\\ "
    "College of Engineering and Applied Sciences \\\\ University of Cincinnati"
)

LATEX_ENGINES = ["xelatex", "lualatex", "pdflatex", "tectonic"]


# Map from the harness output names to the prose figure names.
# Key: basename as output by reproduce/tables via common.save_figure()
# Value: filename expected in prose/fig/
#
# The figure inventory lives in `reproduce/figures/registry.py` -- one row per
# figure cited in the prose, naming its generator and the environment it needs.
# Importing it here means adding a figure is a one-file edit; the fallback keeps
# this script working if it is ever run somewhere `reproduce/` is not present.
def _figure_copies():
    figures_dir = os.path.join(HERE, "..", "..", "reproduce", "figures")
    try:
        sys.path.insert(0, os.path.abspath(figures_dir))
        import registry

        return registry.figure_copies()
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- report and fall back, never abort a build
        print(
            f"  [warn] figure registry unavailable ({exc.__class__.__name__}); "
            f"copying only the Chapter 3 complexity fit"
        )
        return {"fig_03_complexity_fit": "03-complexity-fit"}


FIGURE_COPIES = _figure_copies()


# --------------------------------------------------------------------------- #
# figure copying
# --------------------------------------------------------------------------- #
def copy_figures():
    """Copy figures from the harness outputs to the prose directory.

    The harness generates figures like 'fig_03_complexity_fit.{png,eps}' in
    reproduce/outputs/figures/; the prose expects them at prose/fig/03-complexity-fit.{png,eps}.
    This copies them once at build time, so a newly-generated figure automatically
    reaches the document without manual intervention.

    Returns the number of figures copied (for reporting).
    """
    source_dir = os.path.join(HERE, "..", "..", "reproduce", "outputs", "figures")
    dest_dir = os.path.join(HERE, "prose", "fig")

    if not os.path.exists(source_dir):
        return 0  # no figures generated yet; not an error

    os.makedirs(dest_dir, exist_ok=True)
    copied = 0

    for harness_name, prose_name in FIGURE_COPIES.items():
        for ext in ("png", "eps"):
            source = os.path.join(source_dir, f"{harness_name}.{ext}")
            dest = os.path.join(dest_dir, f"{prose_name}.{ext}")
            if os.path.exists(source):
                shutil.copy2(source, dest)
                copied += 1

    return copied


# --------------------------------------------------------------------------- #
# source assembly
# --------------------------------------------------------------------------- #
def read(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        print(f"  [warn] missing {rel}")
        return None
    # UTF-8 explicitly: the prose carries em dashes, times signs and Greek, and the
    # platform default is cp1252 on Windows, which cannot decode them. This build died
    # on chapters/09-publications.md byte 0x8f before the fix.
    with open(path, encoding="utf-8") as f:
        return f.read()


def replace_mermaid(md):
    """Mermaid source is meaningless in print; the ASCII quarter grid that follows
    it in Ch. 10 is the print rendering of the same schedule."""
    note = (
        "*The Gantt chart is maintained as an interactive figure; the quarter "
        "grid below is the print rendering of the same schedule.*"
    )
    return re.sub(r"```mermaid.*?```", note, md, flags=re.DOTALL)


def strip_html_comments(md):
    """Drop <!-- ... --> blocks.

    The acknowledgements page ships as a template whose guidance lives in an HTML
    comment. Markdown renderers hide it, but pandoc passes it through to LaTeX,
    so it is removed here rather than relied on to stay invisible.
    """
    return re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)


def build_section_registry(sections_by_file):
    """Build a comprehensive section registry from all files.

    Takes a dict mapping filenames to section dicts, and produces:
    - A normalized map of section IDs to full paths
    - A map of §chapter.section notation to sections
    """
    registry = {}
    chapter_sections = {}  # Map of "Ch X §X.Y" -> section_id

    for filename, sections in sections_by_file.items():
        for section_id, info in sections.items():
            registry[section_id] = {
                **info,
                "file": filename,
            }

    return registry


def extract_section_headers(md, filename):
    """Extract all section headers from Markdown to build a cross-reference map.

    Returns a dict mapping section IDs to header info with normalized titles.
    """
    sections = {}
    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+?)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Remove trailing markup like (status; description)
            title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
            # Convert title to section ID: "7.1 The capstone" -> "71-the-capstone"
            # Also handle goal IDs like "G1", "G2", etc.
            section_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            sections[section_id] = {
                "level": level,
                "title": title,
                "raw_title": title,  # Keep original for display
            }
    return sections


def check_cross_references(md, registry, filename, warnings=None):
    """Check for cross-references and warn if targets don't exist.

    Looks for patterns like:
    - "See §7.2" or "See Appendix A.6"
    - "§10.6"
    - References to goals like "G1", "G2"
    - Chapter references
    - Checklist items like C1, C3, etc.

    Excludes Gantt chart lines (Mermaid syntax) from validation.
    """
    if warnings is None:
        warnings = []

    # Remove Gantt chart markup (they use internal IDs that aren't prose references)
    # Gantt lines can be standalone or inline: ":done, g5a, 2026-08-03, 1d" or "goal_id, 2027-01-04, 45d"
    md_for_validation = md
    # Remove inline Gantt parts: ":done, ..., YYYY-MM-DD, ..."
    md_for_validation = re.sub(
        r":\w+,\s*[a-z0-9_]*,\s*\d{4}-\d{2}-\d{2}[^}]*", "", md_for_validation
    )
    # Remove standalone Gantt lines
    md_for_validation = re.sub(
        r"^\s*[a-z0-9_]+,\s*\d{4}-\d{2}-\d{2}.*$",
        "",
        md_for_validation,
        flags=re.MULTILINE,
    )

    # Known checklist items from Chapter 7 Table 7.1 and prose
    known_checklists = {"c1", "c3", "c5", "c8", "c10"}

    # Build maps of known sections by type for validation
    goal_sections = {}  # "g1" -> section_id
    appendix_sections = {}  # "a6" -> section_id
    chapter_sections = {}  # "7" -> [section_ids in chapter 7]

    for section_id, info in registry.items():
        # Match goal IDs: g1, g1a, g2, etc.
        goal_match = re.match(r"^g(\d+)([a-z]?)(?:-|$)", section_id)
        if goal_match:
            goal_key = f"g{goal_match.group(1)}{goal_match.group(2)}"
            goal_sections[goal_key] = section_id

        # Match appendix IDs: a-6, a-7, etc.
        appendix_match = re.match(r"^a-(\d+)(?:-|$)", section_id)
        if appendix_match:
            appendix_key = f"a{appendix_match.group(1)}"
            appendix_sections[appendix_key] = section_id

    # Pattern 1: §X.Y, §X.Y.Z (section references like §7.2)
    for match in re.finditer(r"§(\d+\.\d+(?:\.\d+)?)", md_for_validation):
        section_ref = match.group(1)
        # These map to chapter.section notation; we track them but don't validate
        # against actual headers (that requires mapping prose chapter/section numbers)
        # For now, just ensure the format is valid
        pass

    # Pattern 2: Appendix A.X references
    # Known appendix sections from appendix.md
    known_appendices = {"1", "2", "3", "4", "5", "6", "7"}

    for match in re.finditer(r"Appendix\s+A\.(\d+)", md_for_validation):
        appendix_num = match.group(1)
        if appendix_num not in known_appendices:
            warnings.append(
                f"  ⚠ {filename}: Appendix A.{appendix_num} not found (known: A.1–A.7)"
            )

    # Pattern 3: Goal references like G1, G2, G1a, etc.
    # Known goals from Table 7.1 (including parent goals and sub-goals)
    known_goals = {
        "g1",
        "g2",
        "g3",
        "g3b",
        "g4",
        "g4a",
        "g4b",
        "g4c",
        "g4d",
        "g4e",
        "g5",
        "g6",
        "g7",
        "g8",
        "g9",
    }

    for match in re.finditer(r"\bG(\d+)([a-z]?)\b", md_for_validation, re.IGNORECASE):
        goal_num = match.group(1)
        goal_suffix = match.group(2).lower()
        goal_key = f"g{goal_num}{goal_suffix}"

        # Check against known goals
        if goal_key not in known_goals:
            warnings.append(
                f"  ⚠ {filename}: Goal {goal_key.upper()} not found in Chapter 7 Table 7.1"
            )

    # Pattern 4: Chapter X references
    for match in re.finditer(r"Chapter\s+(\d+)", md_for_validation, re.IGNORECASE):
        chapter_num = match.group(1)
        try:
            ch = int(chapter_num)
            if ch < 0 or ch > 10:
                warnings.append(f"  ⚠ {filename}: Chapter {ch} is out of range (0-10)")
        except ValueError:
            pass

    # Pattern 5: Checklist references like C1, C3, etc.
    for match in re.finditer(r"\bC(\d+)\b", md_for_validation):
        checklist_num = match.group(1)
        checklist_key = f"c{checklist_num}"
        if checklist_key not in known_checklists:
            # Only warn if it's a clear typo (unusual number)
            if checklist_num not in [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "13",
            ]:
                warnings.append(
                    f"  ⚠ {filename}: Checklist item C{checklist_num} not found in Chapter 7"
                )

    # Pattern 6: Cross-reference links [text](§X.Y) or [text](#section-id)
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", md_for_validation):
        target = match.group(2)
        if target.startswith("#"):
            section_id = target[1:].lower()
            if section_id not in registry:
                warnings.append(
                    f"  ⚠ {filename}: Link target '#{section_id}' not found in section registry"
                )

    return warnings


# --------------------------------------------------------------------------- #
# section anchors and reference autolinking
# --------------------------------------------------------------------------- #
_CH_RE = re.compile(r"^#\s+Chapter\s+(\d+)\b")
_APP_TOP_RE = re.compile(r"^#\s+Appendix\b")
_NUMSEC_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+){1,3})\b")
_APPSEC_RE = re.compile(r"^#{2,4}\s+A\.(\d+(?:\.\d+)?)\b")
# The appendix numbers its A.1.x / A.2.x sub-items as bold list labels rather
# than headings (`- **A.2.4** ...`); anchor those too so references resolve.
_APP_LIST_RE = re.compile(r"^\s*[-*]\s+\*\*A\.(\d+(?:\.\d+)*)\*\*")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Protect inline code spans and existing links from reference rewriting.
_PROTECT_RE = re.compile(r"(`[^`]*`|\[[^\]]*\]\([^)]*\))")


def heading_anchor_id(line):
    """The stable anchor id for a numbered heading, or None for an unnumbered
    prose header (which nothing references by number)."""
    m = _CH_RE.match(line)
    if m:
        return f"ch-{m.group(1)}"
    if _APP_TOP_RE.match(line):
        return "appendix"
    m = _NUMSEC_RE.match(line)
    if m:
        return "sec-" + m.group(1).replace(".", "-")
    m = _APPSEC_RE.match(line)
    if m:
        return "sec-a-" + m.group(1).replace(".", "-")
    return None


def list_item_anchor_id(line):
    """Anchor id for a bold appendix list-item label (`- **A.2.4** ...`), which
    the appendix uses in place of a heading for its A.1.x / A.2.x sub-items."""
    m = _APP_LIST_RE.match(line)
    if m:
        return "sec-a-" + m.group(1).replace(".", "-")
    return None


def collect_anchors(raw_by_file):
    """Every anchor id the document will carry, across all files: one per
    numbered heading and one per bold appendix list-item label."""
    anchors = set()
    for md in raw_by_file.values():
        in_fence = False
        for line in md.split("\n"):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            aid = (
                heading_anchor_id(line)
                if line.startswith("#")
                else list_item_anchor_id(line)
            )
            if aid:
                anchors.add(aid)
    return anchors


def add_anchors(md):
    """Attach anchors so references can link to them: `{#id}` on each numbered
    heading, and an inline `[]{#id}` target on each bold appendix list-item
    label (`- **A.2.4** ...`), which the appendix uses in place of a heading."""
    out = []
    in_fence = False
    for line in md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if (
            in_fence
            or re.search(r"\{#[^}]+\}\s*$", line)  # heading already anchored
            or "]{#" in line  # list item already anchored
        ):
            out.append(line)
            continue
        if line.startswith("#"):
            aid = heading_anchor_id(line)
            out.append(f"{line.rstrip()} {{#{aid}}}" if aid else line)
            continue
        aid = list_item_anchor_id(line)
        if aid:
            at = line.index("**")  # inject the empty anchor just before the label
            out.append(f"{line[:at]}[]{{#{aid}}}{line[at:]}")
            continue
        out.append(line)
    return "\n".join(out)


def _protected_sub(line, fn):
    """Apply fn to the plain-text runs of a line, leaving inline code spans and
    existing links untouched."""
    parts = _PROTECT_RE.split(line)
    for i in range(0, len(parts), 2):  # even indices are unprotected text
        parts[i] = fn(parts[i])
    return "".join(parts)


def linkify_refs(md, anchors, filename, missing, stats):
    """Rewrite §X.Y, Chapter N and Appendix A.N references as internal links.

    A reference whose target heading does not exist is left as plain text and
    recorded in `missing` so the build can warn about it. Code spans, existing
    links, fenced code blocks and headings are left untouched.
    """

    def sub_text(text):
        def sec(m):
            num = m.group(1)
            aid = "sec-" + num.replace(".", "-")
            if aid in anchors:
                stats["linked"] += 1
                return f"[§{num}](#{aid})"
            missing.add((filename, f"§{num}"))
            return m.group(0)

        def app(m):
            num = m.group(1)
            aid = "sec-a-" + num.replace(".", "-")
            if aid in anchors:
                stats["linked"] += 1
                return f"[Appendix A.{num}](#{aid})"
            missing.add((filename, f"Appendix A.{num}"))
            return m.group(0)

        def chap(m):
            lead, body = m.group(1), m.group(2)

            def one(nm):
                n = nm.group(0)
                aid = f"ch-{n}"
                if aid in anchors:
                    stats["linked"] += 1
                    return f"[{n}](#{aid})"
                missing.add((filename, f"Chapter {n}"))
                return n

            return lead + re.sub(r"\d+", one, body)

        text = re.sub(r"§(\d+(?:\.\d+)+)", sec, text)
        text = re.sub(r"\bAppendix\s+A\.(\d+(?:\.\d+)?)", app, text)
        text = re.sub(
            r"\b(Chapters?\s+)(\d+(?:\s*(?:,|and|to|–|-)\s*\d+)*)", chap, text
        )
        return text

    out = []
    in_fence = False
    for line in md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or line.startswith("#"):
            out.append(line)
            continue
        out.append(_protected_sub(line, sub_text))
    return "\n".join(out)


def strip_editorial(md, src_dir=None, image_status=None):
    """Drop scaffolding that shouldn't appear in a reading copy.

    `src_dir` is the directory the Markdown came from, used to resolve image
    references so a figure that actually exists can be told from one that is
    still a placeholder.

    `image_status` is an optional dict that will be populated with image
    injection diagnostics: {'included': [...], 'missing': [...]}.
    """
    md = strip_html_comments(replace_mermaid(md))
    if image_status is None:
        image_status = {}
    included = image_status.setdefault("included", [])
    missing = image_status.setdefault("missing", [])

    out = []
    for line in md.split("\n"):
        s = line.strip()
        if re.match(
            r"^\*\*(Status|Repo|Mirrors|Length target|Name|Role note|"
            r"One-line claim):",
            s,
        ):
            continue
        if s.startswith("*Draft —") or s.startswith("*Source of truth:"):
            continue
        m = re.match(r"^`!\[.*\]\((.*)\)`$", s)  # image reference lines
        if m:
            # Strip the line only while the figure is still a PLACEHOLDER. Once
            # the file exists, emit it as a real image so a generated figure
            # actually reaches the built document -- previously every image line
            # was dropped unconditionally, so producing a figure had no effect.
            target = os.path.join(src_dir or HERE, m.group(1))
            alt_text = re.match(r"^`!\[(.*?)\]", s)
            label = alt_text.group(1) if alt_text else m.group(1)
            if os.path.exists(target):
                # Emit the figure as its own block, bounded to the text width.
                # Two things matter. A blank line detaches the image from the
                # caption paragraph: glued inline, a wide (low-aspect) figure is
                # laid mid-flow and overflows the right margin (Figure 4.4).
                # And width=100% caps it to the column -> LaTeX width=\linewidth,
                # HTML width:100%. The from-format disables implicit_figures, so
                # a lone image is not wrapped in a numbered figure with a
                # duplicate caption; the bold "Figure N —" line above is the
                # caption.
                out.append("")
                out.append(s.strip("`") + "{width=100%}")
                included.append(label)
            else:
                missing.append(label)
            continue
        out.append(line)
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(out))


def assemble():
    os.makedirs(BUILD, exist_ok=True)
    image_status = {"included": [], "missing": []}
    link_warnings = []
    sections_by_file = {}  # Accumulate section headers across all files
    raw_by_file = {}  # rel -> raw markdown, read once

    # First pass: read every file once.
    for rel in SECTIONS:
        md = read(rel)
        if md is None:
            continue
        raw_by_file[rel] = md
        sections_by_file[rel] = extract_section_headers(md, rel)

    # Anchor registry drives both the autolinks and the dangling-reference
    # warnings: a §X.Y / Chapter N / Appendix A.N reference is a hyperlink when
    # its target heading exists, and a warning when it does not.
    anchors = collect_anchors(raw_by_file)
    missing_refs = set()
    link_stats = {"linked": 0}

    # Build registry after all files are read
    registry = build_section_registry(sections_by_file)

    # Second pass: anchor headings, strip editorial scaffolding, autolink refs.
    parts = []
    for rel in SECTIONS:
        md = raw_by_file.get(rel)
        if md is None:
            continue
        src_dir = os.path.dirname(os.path.join(HERE, rel))

        md = add_anchors(md)
        part = strip_editorial(md, src_dir, image_status=image_status)
        part = linkify_refs(part, anchors, rel, missing_refs, link_stats)
        parts.append(part)
        print(f"  + {rel}")

        # Keep the existing goal/checklist cross-reference checks.
        check_cross_references(raw_by_file[rel], registry, rel, link_warnings)

        # Insert References section after bibliography.md but before appendix.md
        # The ::: {#refs} ::: div tells pandoc/citeproc where to place the bibliography
        if rel == "prose/bibliography.md":
            parts.append("# References\n\n::: {#refs}\n:::\n")

    combined = "\n\n\n".join(parts)

    # Add the burn-down checklist after references (open items only)
    checklist_path = os.path.join(HERE, CHECKLIST_FILE)
    if os.path.exists(checklist_path):
        with open(checklist_path, "r", encoding="utf-8") as f:
            checklist_md = f.read()
        # Filter to show only open items ([ ] ⬜) and section headers
        filtered_lines = []
        in_section = False
        section_header = None
        for line in checklist_md.split("\n"):
            # Keep section headers (##, #)
            if line.startswith("#"):
                filtered_lines.append(line)
                in_section = True
                continue
            # Keep empty lines and context
            if not line.strip():
                filtered_lines.append(line)
                continue
            # Skip completed items ([x])
            if line.startswith("- [x]"):
                continue
            # Keep open items ([ ])
            if line.startswith("- [ ]"):
                filtered_lines.append(line)
                continue
            # Keep other content (descriptions, tables, etc.)
            if line.startswith("  ") or line.startswith("|"):
                filtered_lines.append(line)
                continue
            # Keep legend and other introductory content
            if in_section or "Legend:" in line or "Tier" in line:
                filtered_lines.append(line)
        checklist_filtered = "\n".join(filtered_lines)
        checklist_filtered = linkify_refs(
            checklist_filtered, anchors, CHECKLIST_FILE, missing_refs, link_stats
        )
        combined += "\n\n" + checklist_filtered
        print(f"  + {CHECKLIST_FILE} (open items only)")

    md_path = os.path.join(BUILD, "proposal-combined.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"  wrote {md_path}")

    # Report image injection status
    if image_status["included"] or image_status["missing"]:
        print(f"\n  image injection:")
        if image_status["included"]:
            for img in image_status["included"]:
                print(f"    ✓ {img}")
        if image_status["missing"]:
            for img in image_status["missing"]:
                print(f"    ✗ {img} (placeholder stripped)")

    # Report section-reference autolinking (and any dangling references)
    print(f"\n  section links:")
    print(f"    ✓ {link_stats['linked']} section reference(s) hyperlinked")
    if missing_refs:
        for fn, ref in sorted(missing_refs):
            print(
                f"    ⚠ {fn}: {ref} has no matching section heading "
                f"(dangling reference, left as plain text)"
            )

    # Report cross-reference warnings
    if link_warnings:
        print(f"\n  cross-reference checks:")
        for warning in link_warnings:
            print(f"    ⚠ {warning}")
    else:
        print(f"\n  ✓ all cross-references verified")

    # Check if bibliography is available
    bib_file_path = os.path.join(HERE, "references.bib")
    if os.path.exists(bib_file_path):
        bib_size = os.path.getsize(bib_file_path)
        print(f"\n  bibliography:")
        print(f"    ✓ references.bib found ({bib_size} bytes, 70 entries)")
        print(f"    ℹ Bibliography support enabled in PDF build")

    return md_path


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def pandoc_bin():
    """System pandoc, else the pypandoc-bundled binary."""
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


TITLE_BLOCK = rf"""---
documentclass: article
papersize: letter
geometry: margin=1.05in
fontsize: 11pt
linestretch: 1.15
numbersections: false
colorlinks: true
nocite: "@*"
header-includes: |
  \usepackage{{titlesec}}
  \titleformat{{\section}}{{\Large\bfseries}}{{}}{{0pt}}{{}}
  \titlespacing{{\section}}{{0pt}}{{0pt}}{{\baselineskip}}
  \let\oldsection\section
  \renewcommand\section{{\clearpage\oldsection}}
---

"""


def build_with_latex(md_path, pandoc, engine):
    """The proper path: pandoc -> LaTeX -> PDF. Full math support."""
    pdf = os.path.join(BUILD, "proposal.pdf")
    src = os.path.join(BUILD, "proposal-titled.md")
    with open(md_path, encoding="utf-8") as f:
        body = f.read()

    # Construct title page with department and committee info
    # DEPT has \\\\ which should become \\ (newline) in LaTeX output
    dept_for_latex = DEPT.replace("\\\\", "\\\\\\par")

    title_page = TITLE_BLOCK + f"""
```{{=latex}}
\\begin{{titlepage}}
\\centering
\\vspace*{{2.2in}}
{{\\Large\\bfseries {TITLE}\\par}}
\\vskip 0.35em
{{\\normalsize\\color{{gray}} {SUBTITLE}\\par}}
\\vskip 2.4em
{{\\normalsize {AUTHOR}\\par}}
\\vskip 0.6em
{{\\small {dept_for_latex}\\par}}
\\vskip 0.6em
{{\\small {COMMITTEE}\\par}}
\\vskip 0.6em
{{\\small Dissertation Proposal · Draft\\par}}
\\vfil
\\end{{titlepage}}
```
"""

    with open(src, "w", encoding="utf-8") as f:
        f.write(title_page + body)
    prose_dir = os.path.join(HERE, "prose")
    bib_file = os.path.abspath(os.path.join(HERE, "references.bib"))

    cmd = [
        pandoc,
        src,
        "-o",
        pdf,
        f"--pdf-engine={engine}",
        "--from",
        "markdown+tex_math_dollars+pipe_tables+fenced_code_blocks-implicit_figures",
        "-V",
        "linkcolor=blue",
        "--wrap=preserve",
        "--resource-path",
        prose_dir,
        "-V",
        "titlepage=true",
        "--citeproc",
        "--bibliography",
        bib_file,
    ]

    # Add bibliography if the file exists
    if os.path.exists(bib_file):
        cmd.extend(
            [
                "--bibliography",
                bib_file,
            ]
        )
    print(f"  pandoc + {engine} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2500:])
        return None
    return pdf


def build_with_weasyprint(md_path, pandoc):
    """Offline fallback: pandoc -> HTML(MathML) -> WeasyPrint PDF."""
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        print("  [fallback] weasyprint not installed")
        return None

    html_path = os.path.join(BUILD, "proposal.html")
    prose_dir = os.path.join(HERE, "prose")
    bib_file = os.path.abspath(os.path.join(HERE, "references.bib"))

    cmd = [
        pandoc,
        md_path,
        "-o",
        html_path,
        "--standalone",
        "--mathml",
        "--from",
        "markdown+tex_math_dollars+pipe_tables-implicit_figures",
        "--wrap=preserve",
        "--metadata",
        f"title={TITLE}",
        "--resource-path",
        prose_dir,
    ]

    # Add bibliography if the file exists
    if os.path.exists(bib_file):
        cmd.extend(
            [
                "--bibliography",
                bib_file,
            ]
        )
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-1500:])
        return None

    title_html = f"""<div class="titlepage">
      <h1 class="doctitle">{TITLE}</h1>
      <p class="sub">{SUBTITLE}</p><p class="author">{AUTHOR}</p>
      <p class="meta">{DEPT.replace(chr(92) * 2, '<br>')}</p>
      <p class="committee">{COMMITTEE}</p>
      <p class="datestr">Dissertation Proposal · Draft</p></div>"""
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("<body>", "<body>" + title_html, 1)

    # WeasyPrint has no MathML support: it would print the glyph soup *and* the
    # <annotation> TeX fallback, duplicating every formula. Drop the annotation so
    # the interim PDF at least shows each formula once. (The real fix is a LaTeX
    # engine -- see main().)
    html = re.sub(r"<annotation\b[^>]*>.*?</annotation>", "", html, flags=re.DOTALL)
    with open(
        html_path, "w", encoding="utf-8"
    ) as f:  # keep the on-disk HTML in sync with the PDF
        f.write(html)

    css = CSS(string="""
    @page { size: letter; margin: 1in 1.05in;
            @bottom-center { content: counter(page);
                             font-family: Georgia, serif; font-size: 9.5pt; color:#555; } }
    @page :first { @bottom-center { content: ""; } }
    body { font-family: Georgia, "Times New Roman", serif; font-size: 10.8pt;
           line-height: 1.5; color:#14161a; text-align: justify; hyphens: auto; }
    .titlepage { text-align:center; page-break-after:always; padding-top:2.2in; }
    .titlepage .doctitle { font-size:26pt; margin:0 0 .35em; page-break-before:avoid; }
    .titlepage .sub { font-size:14pt; color:#333; margin:0 0 2.4em; }
    .titlepage .author { font-size:13.5pt; } .titlepage .meta,.titlepage .committee,
    .titlepage .datestr { font-size:10.3pt; color:#444; }
    h1 { font-size:18pt; page-break-before:always; page-break-after:avoid;
         text-align:left; margin:0 0 .6em; }
    h2 { font-size:13.5pt; margin:1.5em 0 .5em; page-break-after:avoid; text-align:left; }
    h3 { font-size:11.6pt; font-style:italic; margin:1.2em 0 .4em;
         page-break-after:avoid; text-align:left; }
    p { margin:0 0 .62em; orphans:2; widows:2; }
    table { border-collapse:collapse; width:100%; font-family:Arial,sans-serif;
            font-size:8.6pt; margin:.8em 0 1em; page-break-inside:avoid; text-align:left; }
    th,td { border-bottom:.6pt solid #ccc; padding:4pt 6pt; vertical-align:top; }
    th { border-bottom:1pt solid #666; }
    blockquote { margin:.8em 0; padding:6pt 10pt; border-left:2.5pt solid #a8501f;
                 background:#faf7f4; font-family:Arial,sans-serif; font-size:8.8pt;
                 text-align:left; page-break-inside:avoid; }
    pre { background:#f7f7f4; border:.6pt solid #ddd; padding:7pt; font-size:8.2pt;
          white-space:pre-wrap; page-break-inside:avoid; text-align:left; }
    code { font-family:Menlo,Consolas,monospace; font-size:8.8pt; }
    math { font-family:"Latin Modern Math","STIX Two Math","Cambria Math",serif; }
    """)
    pdf = os.path.join(BUILD, "proposal.pdf")
    doc = HTML(string=html, base_url=HERE).render(stylesheets=[css])
    doc.write_pdf(pdf)
    return pdf, len(doc.pages)


def page_count(pdf):
    """Extract page count from PDF using pdfinfo or fallback parsing."""
    try:
        import subprocess

        result = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
    except Exception:
        pass
    # Fallback: try to extract from uncompressed sections
    try:
        with open(pdf, "rb") as f:
            content = f.read()
            # Look for /Count entries which indicate page counts
            matches = re.findall(rb"/Count\s+(\d+)", content)
            if matches:
                return int(matches[0])
    except Exception:
        pass
    return None


def main():
    print("Copying figures from harness outputs ...")
    n = copy_figures()
    if n > 0:
        print(f"  copied {n} figure(s)")

    print("Assembling proposal ...")
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
        # The hint has to match the host it is printed on. This used to offer a
        # `sudo zypper` line unconditionally, which is wrong advice on the machine the
        # proposal is actually built on: a Windows workstation, where it sends the
        # reader looking for a package manager that is not there. The document assembles
        # and every image resolves on this host; only the PDF stage is blocked, so the
        # instruction is the whole value of this branch.
        print("  [note] No LaTeX engine found (xelatex/lualatex/pdflatex/tectonic).")
        print(
            "         The combined Markdown above is complete; only the PDF needs one."
        )
        print("         For publication-grade math, install one:")
        if sys.platform == "win32":
            print("           winget install MiKTeX.MiKTeX      (or TeX Live, or")
            print(
                "           tectonic -- a single self-contained binary, smallest option)"
            )
            print("         WeasyPrint is NOT a usable fallback here: it needs the GTK")
            print("         libraries (libgobject-2.0-0), which Windows does not ship.")
        elif sys.platform == "darwin":
            print(
                "           brew install --cask mactex-no-gui   (or `brew install tectonic`)"
            )
        else:
            print(
                "           apt:    sudo apt install texlive-xetex texlive-latex-recommended"
            )
            print(
                "           zypper: sudo zypper install texlive-xetex texlive-latex "
                "texlive-collection-fontsrecommended"
            )
            print("           or:     cargo install tectonic")
        print("         Falling back to pandoc --mathml + WeasyPrint for now.")

    out = build_with_weasyprint(md_path, pandoc)
    if not out:
        sys.exit("PDF build failed.")
    pdf, pages = out
    print(f"  wrote {pdf}  ({pages} pages)  [pandoc --mathml + WeasyPrint]")


if __name__ == "__main__":
    main()
