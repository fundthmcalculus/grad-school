"""Read measured values out of an archived harness run.

A figure that quotes a number must quote the same number the corresponding
table does. The way to guarantee that is not care -- this project has four
rounds of retraction on record from numbers that were transcribed carefully --
it is to read the table's own CSV.

So a generator asks for `table("table_4_4b_theta_sweep")` and gets the rows the
harness wrote, together with the label of the archive they came from, which the
figure then prints. Nothing is typed in by hand, and a figure drawn against a
superseded archive says so on its face.

Archive selection, in order:
  1. `REPRO_ARCHIVE=<label>` in the environment -- pin a specific run;
  2. the newest archive by the `generated:` stamp in its `PROVENANCE.txt`;
  3. the loose files in `reproduce/outputs/`, which are whatever ran last.
"""

from __future__ import annotations

import csv
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
OUTPUTS = os.path.join(ROOT, "reproduce", "outputs")

_STAMP = re.compile(r"^generated:\s*(\S+)", re.M)
_LABEL = re.compile(r"^label:\s*(\S+)", re.M)


def archives():
    """Every labelled archive, newest first, as (label, path, timestamp)."""
    found = []
    for entry in sorted(os.listdir(OUTPUTS)):
        prov = os.path.join(OUTPUTS, entry, "PROVENANCE.txt")
        if not os.path.isfile(prov):
            continue
        with open(prov) as f:
            text = f.read()
        stamp = _STAMP.search(text)
        label = _LABEL.search(text)
        found.append(((label.group(1) if label else entry),
                      os.path.join(OUTPUTS, entry),
                      stamp.group(1) if stamp else ""))
    return sorted(found, key=lambda r: r[2], reverse=True)


def archive(label=None):
    """(label, path) of the archive to read. See the module docstring for order."""
    label = label or os.environ.get("REPRO_ARCHIVE")
    found = archives()
    if label:
        for name, path, _ in found:
            if name == label:
                return name, path
        raise FileNotFoundError(
            f"no archive labelled {label!r} under {os.path.relpath(OUTPUTS, ROOT)}; "
            f"have: {', '.join(n for n, _, _ in found) or '(none)'}")
    if found:
        return found[0][0], found[0][1]
    return "(unarchived)", OUTPUTS


def table(basename, label=None):
    """(rows, archive_label) for one table's CSV. Rows are dicts, values strings."""
    name, path = archive(label)
    csv_path = os.path.join(path, f"{basename}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{basename}.csv is not in archive {name!r}. Run the generator that "
            f"produces it (see reproduce/PROVENANCE_MAP.md) before drawing a "
            f"figure from it.")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f)), name


def number(text):
    """First numeric value in a harness cell: '0.92 ± 0.03 s' -> 0.92, 'N/A' -> None."""
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None


def spread(text):
    """The ± half-width in a harness cell, or None if the cell carries no spread."""
    m = re.search(r"±\s*([-+]?\d*\.?\d+)", text or "")
    return float(m.group(1)) if m else None


def provenance_note(label):
    """The one-line source stamp a data figure carries in its corner."""
    return f"source: reproduce/outputs/{label}"
