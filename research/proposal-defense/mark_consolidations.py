#!/usr/bin/env python3
"""Mark the passages that state a result and then walk it back.

The draft repeatedly reports a number, then explains that an earlier pass of the
same work reported a different one, then withdraws that. Each such passage was
right to write while the finding was moving. None of them has been shared
outside this repository, so for a reader meeting the document for the first time
they are archaeology: the retracted value is the only place the wrong number
still appears.

This inserts `<!-- CONSOLIDATE: ... -->` above each one. `build_pdf.py` strips
HTML comments (see `strip_html_comments`), so the markers are invisible in the
rendered PDF and greppable in the source:

    grep -rn "CONSOLIDATE" research/proposal-defense/prose/

Marking, not rewriting: which of these to compress and which to keep is an
authorial call. Several *should* stay — §3.4's retraction of machine-invariance
and §6.3.5's superseded optimizer comparison are live methodological points, not
archaeology, and are deliberately not marked here.

Idempotent: a passage already carrying its marker is skipped.

    python research/proposal-defense/mark_consolidations.py [--check]
"""

from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROSE = os.path.join(HERE, "prose")

# (file, anchor that begins the passage, what to consolidate and why)
MARKS = [
    (
        "03-scalable-structure-discovery-pvat.md",
        "**Fuzzy C-Means is a 1.2–3.7× device win, not a 30× one.**",
        "states the correct figure by contrast with a superseded one. The "
        "30-56x version appears nowhere else in the document and was never "
        "published, so the parenthetical retraction below is the only place a "
        "reader meets it. Consider stating the matched figure and the "
        "matched-vs-unmatched distinction directly, without the history.",
    ),
    (
        "04-fast-fis-synthesis-mog.md",
        "This is a smaller claim than the one an earlier pass of this work recorded.",
        "a full paragraph re-deriving a withdrawn measurement (0.014 +/- 0.195) "
        "and the two explanations ruled out on the way to withdrawing it. The "
        "mechanism it lands on -- the output partition, not the transform -- is "
        "worth keeping; the withdrawn numbers are the only trace of a value the "
        "document does not otherwise contain. Appendix A.6 already holds the "
        "investigation, so this can point there in a sentence.",
    ),
    (
        "04-fast-fis-synthesis-mog.md",
        "The zeroth-order figure carries a caveat worth stating, because an earlier version",
        "same shape as the paragraph above and about the same finding: it "
        "quotes -0.434 to explain that -0.434 was an artifact. Consider merging "
        "the two into one statement of what the partition does, kept in one "
        "place rather than two.",
    ),
    (
        "05-topological-membership-generation.md",
        "> That paragraph's earlier, stronger version needs correcting.",
        "a reproduction note that spends a paragraph on a two-week driver crash "
        "(`NameError: name 'row' is not defined`) which changed no number -- "
        "`results.json` came back byte-identical. The lesson is real and belongs "
        "in the reproducibility appendix; in a chapter's Reproduction block it "
        "reads as a correction to a claim no reader saw.",
    ),
    (
        "05-topological-membership-generation.md",
        "**`many_scale` is the clean confirmation the single n = 96 run could not be.**",
        "the finding is stated three times against its own history -- \"the "
        "nine-line paragraph below used to describe as untested\", \"used to call "
        "unwritten is now a real, ten-seed-verified table entry, not a retracted "
        "one\". State the ten-seed result once.",
    ),
    (
        "06-hierarchical-refined-fis.md",
        "Every arm here clears its own spread, which was not true under the previous output partition",
        "carries the superseded spreads (+/-0.241, +/-0.210) to make the point "
        "that the current arms clear theirs. The current spreads already show "
        "that on their own.",
    ),
    (
        "appendix.md",
        "An earlier belief about this arm is withdrawn",
        "withdrawal of a belief the document never asserted elsewhere. If the "
        "corrected account is the only one a reader needs, the withdrawal can go.",
    ),
]

MARKER = "<!-- CONSOLIDATE: {} -->"


def main() -> int:
    check_only = "--check" in sys.argv
    marked = skipped = missing = 0

    for fname, anchor, why in MARKS:
        path = os.path.join(PROSE, fname)
        text = io.open(path, encoding="utf-8").read()
        if anchor not in text:
            print(f"  MISSING anchor in {fname}: {anchor[:60]}...")
            missing += 1
            continue
        idx = text.index(anchor)
        # Insert at the start of the PARAGRAPH containing the anchor, not at the
        # anchor itself. Several anchors are mid-paragraph sentences, and a
        # comment dropped in front of one splits the paragraph in two: the
        # comment is stripped at build time but the blank line it leaves is not,
        # so the rendered PDF gains a paragraph break that nothing in the source
        # asks for. Found by doing exactly that to §6.4.
        para = text.rfind("\n\n", 0, idx)
        insert_at = 0 if para == -1 else para + 2

        # Already marked? The marker is the first thing in the paragraph.
        if text[insert_at:].startswith("<!-- CONSOLIDATE:"):
            skipped += 1
            continue
        if not check_only:
            marker = MARKER.format(" ".join(why.split()))
            text = text[:insert_at] + marker + "\n" + text[insert_at:]
            io.open(path, "w", encoding="utf-8", newline="\n").write(text)
        marked += 1
        print(f"  {'would mark' if check_only else 'marked'}: {fname} -- {anchor[:55]}...")

    print(f"\n{marked} marked, {skipped} already marked, {missing} anchors not found")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
