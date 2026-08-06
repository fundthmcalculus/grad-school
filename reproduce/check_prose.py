#!/usr/bin/env python3
"""Check every number in the proposal against the harness output that should produce it.

    python reproduce/check_prose.py full-2026-08-03
    python reproduce/check_prose.py full-2026-08-03 --against full-14900hx-r2
    python reproduce/check_prose.py full-2026-08-03 --chapter 06 --show-ok
    python reproduce/check_prose.py --since HEAD~1        # did an edit lose a fact?

There are two checks here, against two different failures, and using the wrong one reads
as a disaster.

The archive modes ask whether the document quotes the run it claims to. Use them after a
re-quote, where digits are *supposed* to move: the number to watch is drifted plus
untraceable falling toward a residue you can give a reason for.

`--since <git-ref>` asks whether an editing pass changed the writing and nothing else. Use
it after a voice or compression pass, which should come back identical. Do NOT use it
across a re-quote -- every superseded value will report GONE, correctly and uselessly,
because replacing them was the point. See `since()` for what it cannot tell you.

`compare_runs.py` answers "did the harness move between two runs". This answers the
other half, which the project has so far done by hand: **does the document quote what
the harness actually produced.** `PROVENANCE_MAP.md` tracks that per table, in prose,
maintained by a person -- and the failure it exists to catch has recurred at least six
times (notes 2, 3, 4, 5, 6, 16). A file whose one job is to say "the prose matches the
harness" cannot itself be the thing that goes stale.

## What it checks

Every `mean ± std` pair in `research/proposal-defense/prose/*.md`, in any of the forms
the document actually uses -- `0.877 ± 0.037`, `$0.877 \\pm 0.037$`, `0.84 ± 0.01 s`,
`R2=0.780 ± 0.029`, `7.78 ± 0.50 MPa` -- is looked up in every CSV of the named archive.
Each pair comes back as one of:

  ok          the pair appears verbatim in some archive CSV
  drifted     the mean appears with a DIFFERENT spread, or a near-miss mean exists
  untraceable the pair appears in no CSV of this archive at all

A ± pair is the right unit for this. It is specific enough that a coincidental match is
unlikely, it is the form every generator emits, and it is exactly what a reader will
spot-check. Bare numbers are reported separately and only when they carry three or more
decimals, because `0.5` matches everything and says nothing.

## What it deliberately does NOT do

It does not map prose tables to CSV columns. That mapping is what `PROVENANCE_MAP.md` is
for, it needs judgement, and a wrong automatic mapping would be worse than none -- it
would report a confident match against the wrong column. This tool answers the narrower
question completely instead of the broad question approximately: *is this pair of digits
anywhere in the run I am claiming to quote?*

It also cannot tell a legitimately-cited external measurement from a drifted cell. The
NAFIPS 4,096-point pair is `cited` by decision (`PROVENANCE_MAP.md` note 1) and will
always report untraceable. `--ignore-file` takes a list of pairs to treat as known-cited
so the untraceable list stays short enough to read; anything on that list should have a
note in the provenance map saying why.

## Reading the output

`untraceable` is the interesting bucket and it has three causes, in descending
likelihood: the prose is quoting an older archive, the number was hand-transcribed, or
the cell comes from a generator this archive does not run. Check which before editing --
the first two are prose bugs and the third is a harness gap.

`--against` adds a second archive and marks any untraceable pair that DOES appear there.
That is the single most useful mode: it turns "this number is from somewhere" into "this
number is from the run you superseded", which names the fix.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

# The document and the harness both write ± and ² and λ. Windows defaults stdout to
# cp1252, which encodes none of them, so a report would be assembled and then die in
# its own print -- the failure shape build_pdf.py, make_figures.py, the table emitters
# and the Chapter 5 driver each hit separately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(HERE, "outputs")
PROSE = os.path.join(ROOT, "research", "proposal-defense", "prose")

# `−` (MINUS SIGN) is what the prose uses; the generators emit ASCII `-`. Without it
# here, a negative prose cell is compared as if unsigned and reports untraceable against
# the very CSV that holds it -- `-2.106 ± 4.274` and `-12.562 ± 24.000` in Table 4.3
# both did. `norm()` folds it to ASCII so the two forms compare equal.
_NUM = r"[-+−]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# `±`, `+/-`, and LaTeX `\pm` all appear in the prose; the generators emit `±`.
_SEP = r"(?:±|\+/-|\\pm)"
_PAIR = re.compile(rf"({_NUM})\s*{_SEP}\s*({_NUM})")
# Three or more decimals: enough to be a table cell rather than a round number in a
# sentence. `0.9967` qualifies, `0.5` and `1.00` do not.
_PRECISE = re.compile(rf"(?<![\d.])({_NUM}\.\d{{3,}})(?![\d])")

# Prose that is deliberately not a harness cell. Kept as substrings of the source line
# rather than as numbers, so the reason travels with the exemption.
_SKIP_LINE = (
    "REPRO_NAIVE_CAP",  # the cited NAFIPS row explains itself on its own line
)


def norm(text):
    """A numeric literal as a canonical string, so 0.840 and 0.84 do not both pass.

    Compared as digits rather than as floats on purpose: a prose cell written to two
    decimals where the harness emits three is a real difference a reader can see, and
    float equality would hide it. The float value is carried alongside for the
    near-miss check.
    """
    return text.strip().lstrip("+").replace("−", "-")


def read_archive(label):
    """{(mean, std): [files]} and {mean: [files]} for one archive's CSVs."""
    path = os.path.join(OUTPUTS, label)
    if not os.path.isdir(path):
        raise SystemExit(
            f"error: no archive {label!r} under " f"{os.path.relpath(OUTPUTS, ROOT)}"
        )
    pairs, singles = defaultdict(list), defaultdict(list)
    for name in sorted(os.listdir(path)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(path, name), newline="", encoding="utf-8") as f:
            text = f.read()
        for mean, std in _PAIR.findall(text):
            pairs[(norm(mean), norm(std))].append(name)
        for cell in _PRECISE.findall(text):
            singles[norm(cell)].append(name)
        # Every mean of a ± pair is also a bare value someone may quote alone.
        for mean, _ in _PAIR.findall(text):
            singles[norm(mean)].append(name)
    return pairs, singles


def read_prose(chapter=None):
    """[(file, lineno, line)] for the prose files in scope."""
    if not os.path.isdir(PROSE):
        raise SystemExit(f"error: no prose directory at {os.path.relpath(PROSE, ROOT)}")
    out = []
    for name in sorted(os.listdir(PROSE)):
        if not name.endswith(".md"):
            continue
        if chapter and not name.startswith(chapter):
            continue
        with open(os.path.join(PROSE, name), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                out.append((name, i, line.rstrip("\n")))
    return out


# How far a candidate's spread may differ from the prose's before a matching mean stops
# being evidence of anything. Without this the check is unit-blind and suggests an R²
# as the rounding of a wall clock: Table 4.5's `0.84 ± 0.01 s` drew `0.836 ± 0.054`
# (a Concrete R²) as a near miss, which is two different quantities that happen to round
# to the same three digits. Units live in column headers rather than in cells, so they
# cannot be compared directly -- but a spread five times wider is a different
# measurement whatever the unit, and that is cheap and unit-free to test.
_SPREAD_RATIO = 5.0
_MAX_SUGGESTIONS = 4


def _spread_compatible(prose_std, cand_std):
    try:
        a, b = abs(float(prose_std)), abs(float(cand_std))
    except ValueError:
        return True
    if a == 0 or b == 0:
        return a == b
    return (a / b if a > b else b / a) <= _SPREAD_RATIO


def near_miss(mean, std, pairs):
    """A pair whose mean is within one ulp of the prose's last printed digit.

    This is the rounding case: the prose says 0.877 and the harness says 0.8774, or the
    prose rounds 0.8765 to 0.876. Reported separately from a real drift because the fix
    is different -- one is a transcription rounding, the other is a superseded run.

    Candidates whose spread is more than `_SPREAD_RATIO` from the prose's are dropped:
    a matching mean with a wildly different spread is a coincidence, not a rounding.
    """
    try:
        want = float(mean)
    except ValueError:
        return []
    decimals = len(mean.split(".")[1]) if "." in mean else 0
    tol = 0.5 * (10**-decimals) if decimals else 0.5
    hits = []
    for (m, s), files in pairs.items():
        try:
            got = float(m)
        except ValueError:
            continue
        if m != mean and abs(got - want) <= tol and _spread_compatible(std, s):
            hits.append((m, s, files))
    # Closest first, so a truncated list keeps the useful ones.
    hits.sort(key=lambda h: abs(float(h[0]) - want))
    return hits[:_MAX_SUGGESTIONS]


# Numeric literals, tracking the VALUE and ignoring whatever unit follows it. Two
# versions of this were wrong in opposite directions, and both failed the same way -- by
# reporting a formatting change as a lost fact, which is worse than reporting nothing.
#
#   `(?<![\w.])\d+...` with a trailing `(?![\w])`: rejected any number followed by a
#   letter, and `\w` is Unicode-aware, so `571σ` did not tokenise at all. Chapter 5's
#   sigma-multiples then read as deleted when they had only gained their unit.
#
#   Before that, no trailing guard at all: `1,` and `7.` became tokens, so a sentence
#   split or a serial comma read as a lost number in three files.
#
# The trailing guard is therefore `(?!\d)(?!\.\d)`: the token must not run into another
# digit, nor into a decimal point that is followed by one. Anything else may follow it.
# Both halves are needed. `(?![\d.])` alone -- the obvious form -- rejects a number at
# the end of a sentence, so `... rising to 7.` silently contributes nothing and a
# reflowed paragraph reads as a deletion. `0.62 s`, `571σ`, `48×`, `12-class` and a
# sentence-final `7.` all yield the number; `4.3e-14` and `1,024` stay whole. The
# leading guard still excludes identifiers, since `_` is a word character, so `fig_07`
# and `table_5_x` do not match at all.
_TOKEN = re.compile(
    r"(?<![\w.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\d)(?!\.\d)"
)


def numeric_tokens(text):
    """Multiset of numeric literals, with the sign normalised away.

    A leading `+` is dropped so `+0.914` in a table cell and `0.914` in the sentence
    describing it are the same fact. Without that, moving a number between a table and
    its prose reads as one deletion and one addition -- which is exactly what Chapter 6
    produced when a compression pass cut a restatement whose table row was still there.
    A leading `-` is kept, because a negative R2 is a different claim from a positive
    one and this document has cells of both signs.
    """
    from collections import Counter

    return Counter(t.lstrip("+").replace("−", "-") for t in _TOKEN.findall(text))


def since(ref, chapter=None):
    """Numbers present in the prose at `ref` and missing now, per file.

    The companion check to the archive comparison above, and the one that catches a
    different failure: an editing pass that drops a fact while reporting that it did
    not. A voice or compression pass is supposed to change how the document reads and
    nothing about what it claims, so every numeric literal in it should survive.

    That is a real thing that happened. A compression pass reported per file that every
    correction had survived; this check, run against the pre-compression commit, found
    Chapter 5 missing the sigma-multiples and inter-level ratios its own table caption
    depended on. (They came back -- the file was mid-edit -- but the self-report could
    not have told the difference, and neither could a reader.)

    Two things it is NOT. It is not a fact checker: a number can survive while the
    sentence around it turns false. And a missing token is not automatically a defect --
    collapsing a cross-reference mentioned twice into one mention legitimately drops a
    `6.1`. So the output is a worklist to read, not a verdict, and it prints the counts
    so a 2 -> 1 duplicate reads differently from a 1 -> 0 deletion.
    """
    import subprocess

    out = []
    for name in sorted(os.listdir(PROSE)):
        if not name.endswith(".md") or (chapter and not name.startswith(chapter)):
            continue
        rel = os.path.relpath(os.path.join(PROSE, name), ROOT).replace(os.sep, "/")
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            out.append((name, None, None))  # not present at that ref
            continue
        with open(os.path.join(PROSE, name), encoding="utf-8") as f:
            now = numeric_tokens(f.read())
        was = numeric_tokens(proc.stdout)
        out.append((name, was - now, now - was, was, now))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "archive",
        nargs="?",
        help="archive label the prose claims to quote; omit with --since",
    )
    ap.add_argument(
        "--against",
        metavar="LABEL",
        help="second archive; untraceable pairs found here are named as "
        "coming from a superseded run",
    )
    ap.add_argument(
        "--chapter",
        metavar="PREFIX",
        help="restrict to prose files starting with this, e.g. 06",
    )
    ap.add_argument(
        "--show-ok",
        action="store_true",
        help="list the pairs that match, not only the ones that do not",
    )
    ap.add_argument(
        "--singles",
        action="store_true",
        help="also check bare numbers carrying 3+ decimals (noisier)",
    )
    ap.add_argument(
        "--ignore-file",
        metavar="PATH",
        help="file of 'mean ± std' pairs that are cited rather than "
        "harness-produced; one per line, # comments allowed",
    )
    ap.add_argument(
        "--since",
        metavar="GIT_REF",
        help="instead of checking against an archive, report numeric "
        "literals present in the prose at GIT_REF and missing now -- "
        "the check that an editing pass changed voice and not facts",
    )
    args = ap.parse_args()

    if args.since:
        rc = 0
        # Document-wide totals as well as per file, because a token MOVED between two
        # files is lost from one and gained by the other. Per file that reads as a
        # deletion plus an unexplained addition; across the document it is neither. A
        # relocation pass -- pulling retraction history out of Chapter 7 into a new
        # appendix section -- is exactly the case the per-file view cannot judge, so it
        # is the aggregate that decides whether anything actually went missing.
        from collections import Counter

        doc_was, doc_now = Counter(), Counter()
        print(f"prose numeric literals: {args.since} -> working tree")
        print(f"{'':-<78}")
        for name, lost, gained, was, now in since(args.since, args.chapter):
            if was is not None:
                doc_was += was
                doc_now += now
            if lost is None:
                print(f"  {name:46s} not present at {args.since}")
                continue
            if not lost and not gained:
                print(f"  {name:46s} identical")
                continue
            rc = 1
            print(f"  {name:46s} DIFFERS")
            # before -> after, not just the deficit. Without it every legitimate
            # de-duplication reads as a deletion: collapsing a fact stated three
            # times into two shows as "lost 0.62 x1" and looks identical to losing
            # the only mention. GONE is the line to read.
            for tok, n in sorted(lost.items()):
                after = now.get(tok, 0)
                tag = "GONE  " if after == 0 else "de-dup"
                print(f"      {tag} {tok:>14s}   {was[tok]} -> {after}")
            for tok, n in sorted(gained.items()):
                print(f"      new    {tok:>14s}   {was.get(tok, 0)} -> {now[tok]}")
        doc_lost, doc_gained = doc_was - doc_now, doc_now - doc_was
        gone = {t: n for t, n in doc_lost.items() if doc_now.get(t, 0) == 0}
        print(f"{'':-<78}")
        if not doc_lost and not doc_gained:
            print("  WHOLE DOCUMENT: identical")
        else:
            print(
                f"  WHOLE DOCUMENT: {len(gone)} token(s) gone, "
                f"{len(doc_lost) - len(gone)} de-duplicated, "
                f"{len(doc_gained)} new"
            )
            for tok in sorted(gone):
                print(f"      GONE   {tok:>14s}   {doc_was[tok]} -> 0")
        print("")
        print("Read the WHOLE DOCUMENT line first. A token moved between two files is")
        print("lost from one and gained by the other, so only the aggregate can say")
        print("whether a fact left the document. Per file, `de-dup` is a repeated fact")
        print("stated fewer times, which is what a compression pass is for.")
        return 1 if gone else 0

    if not args.archive:
        ap.error("an archive label is required unless --since is given")

    pairs, singles = read_archive(args.archive)
    other_pairs, other_singles = ({}, {})
    if args.against:
        other_pairs, other_singles = read_archive(args.against)

    ignored = set()
    if args.ignore_file:
        with open(args.ignore_file, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                m = _PAIR.search(line)
                if m:
                    ignored.add((norm(m.group(1)), norm(m.group(2))))

    ok, drifted, untraceable, cited = [], [], [], []
    seen_pairs = set()

    for name, lineno, line in read_prose(args.chapter):
        if any(s in line for s in _SKIP_LINE):
            continue
        for mean, std in _PAIR.findall(line):
            key = (norm(mean), norm(std))
            where = (name, lineno, line.strip()[:110])
            if key in ignored:
                cited.append((key, where))
                continue
            seen_pairs.add(key)
            if key in pairs:
                ok.append((key, where, pairs[key]))
            else:
                nm = near_miss(key[0], key[1], pairs)
                same_mean = [
                    (m, s, f)
                    for (m, s), f in pairs.items()
                    if m == key[0] and _spread_compatible(key[1], s)
                ]
                if same_mean or nm:
                    drifted.append((key, where, same_mean, nm))
                else:
                    in_other = other_pairs.get(key)
                    untraceable.append((key, where, in_other))

    label = args.archive
    print(
        f"prose vs {label}"
        + (f"  (also checked against {args.against})" if args.against else "")
    )
    print(f"{'':-<78}")
    print(f"{len(ok):5d} pairs match a cell in {label}")
    print(
        f"{len(drifted):5d} pairs DRIFTED -- same mean, different spread, or a "
        f"rounding difference"
    )
    print(f"{len(untraceable):5d} pairs UNTRACEABLE -- in no CSV of {label}")
    if cited:
        print(f"{len(cited):5d} pairs skipped as cited (--ignore-file)")
    print()

    if drifted:
        print("DRIFTED")
        print(f"{'':-<78}")
        for (mean, std), (name, lineno, line), same, nm in drifted:
            print(f"  {name}:{lineno}  prose says {mean} ± {std}")
            # These candidates come from ANY CSV in the archive, not from the table the
            # prose cell belongs to -- this tool does not map cells to columns (see the
            # module docstring). Check the filename before believing a suggestion: a
            # mean matching across two unrelated experiments is a coincidence.
            for m, s, files in same:
                print(
                    f"      same mean in {m} ± {s}   "
                    f"({', '.join(sorted(set(files)))})"
                )
            for m, s, files in nm:
                print(
                    f"      rounds to?   {m} ± {s}   "
                    f"({', '.join(sorted(set(files)))})"
                )
            print(f"      | {line}")
        print()

    if untraceable:
        print("UNTRACEABLE")
        print(f"{'':-<78}")
        for (mean, std), (name, lineno, line), in_other in untraceable:
            tag = ""
            if in_other:
                tag = (
                    f"  <- IS in {args.against} "
                    f"({', '.join(sorted(set(in_other)))}): superseded run"
                )
            print(f"  {name}:{lineno}  {mean} ± {std}{tag}")
            print(f"      | {line}")
        print()

    if args.singles:
        miss = []
        for name, lineno, line in read_prose(args.chapter):
            for cell in _PRECISE.findall(line):
                if norm(cell) not in singles:
                    in_other = other_singles.get(norm(cell))
                    miss.append(
                        (norm(cell), name, lineno, line.strip()[:110], in_other)
                    )
        print(f"BARE NUMBERS (3+ decimals) not in {label}: {len(miss)}")
        print(f"{'':-<78}")
        for cell, name, lineno, line, in_other in miss:
            tag = f"  <- in {args.against}" if in_other else ""
            print(f"  {name}:{lineno}  {cell}{tag}")
            print(f"      | {line}")
        print()

    if args.show_ok:
        print("OK")
        print(f"{'':-<78}")
        for (mean, std), (name, lineno, _), files in ok:
            print(
                f"  {name}:{lineno}  {mean} ± {std}   "
                f"({', '.join(sorted(set(files)))})"
            )
        print()

    # Non-zero when the document quotes something this run does not contain, so this can
    # gate a build. Drift is a failure too: a spread that does not match is a different
    # measurement, not a formatting difference.
    return 1 if (drifted or untraceable) else 0


if __name__ == "__main__":
    sys.exit(main())
