#!/usr/bin/env python3
"""Diff two archived table runs, cell by cell.

    python reproduce/compare_runs.py baseline-d0d6714 postfix-5d97892

Reads the CSVs archived under ``reproduce/outputs/<label>/`` by
``run_all_tables.sh`` and reports, for every table, which cells moved and by how
much. Writes ``reproduce/outputs/FIX_IMPACT.md``.

The point of this script is to make "unchanged" a *reported result* rather than
an absence of one. A fix whose blast radius is supposed to be confined to one
table is only shown to be confined if the other tables are checked and stated to
be identical -- so every table appears in the output, including the ones with
nothing to say.

Cells are compared numerically where they parse as ``mean`` or ``mean ± std``,
and as strings otherwise. ``N/A`` on both sides is unchanged; ``N/A`` on exactly
one side is a change, and usually the interesting kind -- it means a method
started or stopped being runnable.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")

# "0.859", "0.859 ± 0.017", "+0.155", "1.2e-3 ± 4e-4"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_CELL = re.compile(rf"^\s*(?P<mean>{_NUM})\s*(?:(?:±|\+/-)\s*(?P<std>{_NUM}))?\s*$")

# Cells whose value is a wall clock: expected to wobble between runs, so a
# change there is reported but never called a regression.
_TIME_HINTS = ("time", "sec", "seconds", "ms", "wall", "fit", "train")


def parse_cell(text):
    """-> (mean, std) if numeric, else None."""
    if text is None:
        return None
    m = _CELL.match(str(text).replace("**", ""))
    if not m:
        return None
    return float(m.group("mean")), (float(m.group("std")) if m.group("std") else 0.0)


def read_table(path):
    """-> (header, rows) or None if the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    return rows[0], rows[1:]


def row_key(row):
    """Rows are matched by their leading label column(s), not by position, so a
    reordered or newly-inserted row does not cascade into a wall of false
    differences."""
    return tuple(c.replace("**", "").strip() for c in row[:1])


def is_time_column(name):
    low = name.lower()
    return any(h in low for h in _TIME_HINTS)


def compare_table(name, base_path, new_path):
    """-> dict describing one table's differences."""
    base = read_table(base_path)
    new = read_table(new_path)

    if base is None and new is None:
        return {"name": name, "state": "absent-both"}
    if base is None:
        return {"name": name, "state": "new-only"}
    if new is None:
        return {"name": name, "state": "baseline-only"}

    bh, brows = base
    nh, nrows = new
    if bh != nh:
        return {"name": name, "state": "header-changed", "base_header": bh, "new_header": nh}

    bmap = {row_key(r): r for r in brows}
    nmap = {row_key(r): r for r in nrows}

    diffs, added, removed = [], [], []
    for key in nmap.keys() - bmap.keys():
        added.append(" / ".join(key))
    for key in bmap.keys() - nmap.keys():
        removed.append(" / ".join(key))

    n_cells = 0
    for key in sorted(bmap.keys() & nmap.keys()):
        brow, nrow = bmap[key], nmap[key]
        for col_idx in range(1, min(len(bh), len(brow), len(nrow))):
            n_cells += 1
            b_txt, n_txt = brow[col_idx].strip(), nrow[col_idx].strip()
            if b_txt == n_txt:
                continue
            b_num, n_num = parse_cell(b_txt), parse_cell(n_txt)
            entry = {
                "row": " / ".join(key),
                "column": bh[col_idx],
                "before": b_txt or "(empty)",
                "after": n_txt or "(empty)",
                "timing": is_time_column(bh[col_idx]),
            }
            if b_num and n_num:
                delta = n_num[0] - b_num[0]
                entry["delta"] = delta
                # Judge the move against the run-to-run spread, not against zero:
                # a shift well inside one standard deviation is not evidence.
                spread = max(b_num[1], n_num[1])
                entry["significant"] = (not entry["timing"]) and (
                    abs(delta) > max(spread, 1e-12)
                    if spread > 0
                    else not math.isclose(delta, 0.0, abs_tol=1e-9)
                )
            else:
                entry["significant"] = not entry["timing"]
            diffs.append(entry)

    return {
        "name": name,
        "state": "compared",
        "n_cells": n_cells,
        "diffs": diffs,
        "added_rows": sorted(added),
        "removed_rows": sorted(removed),
    }


def fmt_delta(entry):
    if "delta" not in entry:
        return ""
    d = entry["delta"]
    return f"{d:+.4f}" if abs(d) < 1000 else f"{d:+.3e}"


def render(base_label, new_label, results, provenance):
    L = []
    L.append(f"# Fix impact — `{base_label}` → `{new_label}`\n")
    L.append("Cell-by-cell diff of the archived table runs, produced by "
             "`reproduce/compare_runs.py`. Every table is listed, including the "
             "unchanged ones: confining a fix's blast radius is a claim, and it is "
             "only supported by showing the tables that did *not* move.\n")

    for label, text in provenance.items():
        L.append(f"<details><summary>Provenance — <code>{label}</code></summary>\n")
        L.append("```\n" + text.rstrip() + "\n```\n")
        L.append("</details>\n")

    compared = [r for r in results if r["state"] == "compared"]
    changed = [r for r in compared if r["diffs"] or r["added_rows"] or r["removed_rows"]]
    identical = [r for r in compared if r not in changed]
    problem = [r for r in results if r["state"] != "compared"]

    L.append("## Summary\n")
    L.append("| Table | Cells | Verdict |")
    L.append("|---|---:|---|")
    for r in sorted(results, key=lambda x: x["name"]):
        if r["state"] != "compared":
            L.append(f"| `{r['name']}` | — | **{r['state']}** |")
            continue
        sig = sum(1 for d in r["diffs"] if d["significant"])
        timing = sum(1 for d in r["diffs"] if d["timing"])
        noise = len(r["diffs"]) - sig - timing
        if not r["diffs"] and not r["added_rows"] and not r["removed_rows"]:
            verdict = "identical"
        else:
            bits = []
            if sig:
                bits.append(f"**{sig} changed**")
            if noise:
                bits.append(f"{noise} within noise")
            if timing:
                bits.append(f"{timing} timing")
            if r["added_rows"]:
                bits.append(f"{len(r['added_rows'])} rows added")
            if r["removed_rows"]:
                bits.append(f"{len(r['removed_rows'])} rows removed")
            verdict = ", ".join(bits)
        L.append(f"| `{r['name']}` | {r['n_cells']} | {verdict} |")
    L.append("")

    if problem:
        L.append("## Tables that could not be compared\n")
        for r in sorted(problem, key=lambda x: x["name"]):
            L.append(f"- `{r['name']}` — **{r['state']}**"
                     + (f"\n  - baseline header: `{r.get('base_header')}`"
                        f"\n  - new header: `{r.get('new_header')}`"
                        if r["state"] == "header-changed" else ""))
        L.append("")

    if changed:
        L.append("## What moved\n")
        for r in sorted(changed, key=lambda x: x["name"]):
            L.append(f"### `{r['name']}`\n")
            if r["added_rows"]:
                L.append(f"Rows only in `{new_label}`: "
                         + ", ".join(f"`{x}`" for x in r["added_rows"]) + "\n")
            if r["removed_rows"]:
                L.append(f"Rows only in `{base_label}`: "
                         + ", ".join(f"`{x}`" for x in r["removed_rows"]) + "\n")
            if r["diffs"]:
                L.append("| Row | Column | Before | After | Δ | |")
                L.append("|---|---|---|---|---:|---|")
                for d in sorted(r["diffs"],
                                key=lambda x: (not x["significant"], x["row"])):
                    flag = ("**changed**" if d["significant"]
                            else ("timing" if d["timing"] else "within noise"))
                    L.append(f"| {d['row']} | {d['column']} | {d['before']} | "
                             f"{d['after']} | {fmt_delta(d)} | {flag} |")
                L.append("")

    if identical:
        L.append("## Bit-identical\n")
        L.append("These tables produced exactly the same numbers on both sides:\n")
        for r in sorted(identical, key=lambda x: x["name"]):
            L.append(f"- `{r['name']}` ({r['n_cells']} cells)")
        L.append("")

    L.append("---\n")
    L.append("> A cell counts as **changed** only if it moved by more than the "
             "larger of the two runs' reported standard deviations; smaller moves "
             "are labelled *within noise*. Wall-clock columns are always reported "
             "separately and never called a regression — this harness does not "
             "control clocks or thermals (see G4 in `NEXT_STEPS.md`).\n")
    return "\n".join(L)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    base_label, new_label = sys.argv[1], sys.argv[2]
    base_dir = os.path.join(OUTPUTS, base_label)
    new_dir = os.path.join(OUTPUTS, new_label)
    for d in (base_dir, new_dir):
        if not os.path.isdir(d):
            sys.exit(f"no such run directory: {d}\n"
                     f"produce one with: reproduce/run_all_tables.sh <label>")

    names = sorted({f[:-4] for d in (base_dir, new_dir)
                    for f in os.listdir(d) if f.endswith(".csv")})
    if not names:
        sys.exit(f"no CSVs found in {base_dir} or {new_dir}")

    results = [compare_table(n,
                             os.path.join(base_dir, f"{n}.csv"),
                             os.path.join(new_dir, f"{n}.csv"))
               for n in names]

    provenance = {}
    for label, d in ((base_label, base_dir), (new_label, new_dir)):
        p = os.path.join(d, "PROVENANCE.txt")
        if os.path.exists(p):
            with open(p) as f:
                provenance[label] = f.read()

    out = os.path.join(OUTPUTS, "FIX_IMPACT.md")
    with open(out, "w") as f:
        f.write(render(base_label, new_label, results, provenance))
    print(f"wrote {out}")

    for r in sorted(results, key=lambda x: x["name"]):
        if r["state"] != "compared":
            print(f"  {r['name']:<40} {r['state']}")
        elif not r["diffs"]:
            print(f"  {r['name']:<40} identical")
        else:
            sig = sum(1 for d in r["diffs"] if d["significant"])
            print(f"  {r['name']:<40} "
                  f"{len(r['diffs'])} cells differ ({sig} beyond noise)")


if __name__ == "__main__":
    main()
