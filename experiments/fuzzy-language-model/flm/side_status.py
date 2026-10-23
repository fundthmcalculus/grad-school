"""Collect the state of the three CPU side explorations (DUST-style SGD, post-training quantisation, "surgery") into one JSON table for the status dashboard: ``python -m flm.side_status <out.json> [notes.json]``.

Every number comes from a file the explorations write (``outputs/dust/*.log``, ``outputs/ptq_probe.log``, ``outputs/surgery_*.log``); nothing is typed in.  ``notes.json`` (optional) holds a one-line ``{agent: headline}`` written by hand at each refresh.
Rows: ``panel, id, label, group, x, y, aux, note, updated`` (unused cells are null).  ``panel`` is ``agent`` (one per exploration), ``ptq`` (x bytes, y bits per character), ``dust`` (one row per evaluation: group the run, x training seconds, y bits per character, aux step), ``dust_run`` (one per run: y final bits per character, x seconds per step), ``surgery`` (progress and table lines) or ``meta``.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("outputs")


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def newest(paths):
    ts = [p.stat().st_mtime for p in paths if p.exists()]
    return max(ts) if ts else None


def row(
    panel,
    id_,
    label=None,
    group=None,
    x=None,
    y=None,
    aux=None,
    z=None,
    note=None,
    detail=None,
    updated=None,
):
    return dict(
        panel=panel,
        id=id_,
        label=label,
        group=group,
        x=x,
        y=y,
        aux=aux,
        z=z,
        detail=detail,
        note=note,
        updated=updated,
    )


def ptq_rows():
    rows, log = [], OUT / "ptq_probe.log"
    if not log.exists():
        return rows, None
    pat = re.compile(r"^(.+?)\s{2,}([0-9.]+)\s+(\d+) B\s+\(([0-9.]+)s\)")
    for i, line in enumerate(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        m = pat.match(line)
        if m:
            name, bpc, size = m.group(1).strip(), float(m.group(2)), int(m.group(3))
            kind = name.split()[1] if name.startswith("all ") else name.split()[0]
            rows.append(
                row(
                    "ptq",
                    f"ptq{i}",
                    name,
                    kind,
                    x=size,
                    y=bpc,
                    aux=float(m.group(4)),
                    updated=iso(log.stat().st_mtime),
                )
            )
        elif "FAILED" in line:
            rows.append(
                row(
                    "ptq",
                    f"ptq{i}",
                    line.split(" FAILED")[0].strip(),
                    "failed",
                    note=line.split("FAILED:")[-1].strip()[:140],
                    updated=iso(log.stat().st_mtime),
                )
            )
    return rows, log.stat().st_mtime


def dust_rows():
    rows, files = [], sorted((OUT / "dust").glob("*.log"))
    pat = re.compile(
        r"^step (\d+) train_loss ([0-9.]+) val_bpc ([0-9.]+) train_seconds ([0-9.]+)"
    )
    for f in files:
        pts = [
            pat.match(l)
            for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
        ]
        pts = [m for m in pts if m]
        if not pts:
            rows.append(
                row(
                    "dust_run",
                    f.stem,
                    f.stem,
                    "starting",
                    note="no evaluation yet",
                    updated=iso(f.stat().st_mtime),
                )
            )
            continue
        stem = f.stem.lower()
        method = (
            "backprop (Adam)"
            if "adam" in stem
            else (
                "backprop (SGD)"
                if "sgd" in stem
                else (
                    "backprop (Lion)"
                    if "lion" in stem
                    else "backprop (Muon)" if "muon" in stem else "DUST-like"
                )
            )
        )
        for m in pts:
            rows.append(
                row(
                    "dust",
                    f"{f.stem}-{m.group(1)}",
                    f.stem,
                    method,
                    x=float(m.group(4)),
                    y=float(m.group(3)),
                    aux=int(m.group(1)),
                    updated=iso(f.stat().st_mtime),
                )
            )
        last = pts[-1]
        sps = float(last.group(4)) / max(1, int(last.group(1)))
        done = (f.with_suffix(".json")).exists()
        rows.append(
            row(
                "dust_run",
                f.stem,
                f.stem,
                method,
                x=round(sps, 3),
                y=float(last.group(3)),
                aux=int(last.group(1)),
                note="finished" if done else "running",
                updated=iso(f.stat().st_mtime),
            )
        )
    return rows, newest(files)


def surgery_result_rows():
    """Median effect of each edit method on the bystander positions (report/surgery.json, written when the run ends)."""
    f = Path("report/surgery.json")
    if not f.exists():
        return [], None
    import numpy as np

    d = json.loads(f.read_text())
    rows = []
    for stratum in ("near", "far"):
        for method, rs in d["results"].get(stratum, {}).items():
            ok = [r for r in rs if r.get("success")]

            def med(k):
                v = [r[k] for r in ok if k in r and r[k] == r[k]]
                return round(float(np.median(v)), 5) if v else None

            rows.append(
                row(
                    "surgery_res",
                    f"{stratum}-{method}",
                    method,
                    stratum,
                    x=med("flips"),
                    y=med("dbpc"),
                    aux=len(ok),
                    z=med("rel_norm"),
                    note=f"{len(rs)} events",
                    updated=iso(f.stat().st_mtime),
                )
            )
    return rows, f.stat().st_mtime


def surgery_example_rows():
    """Readable before / after examples written by the surgery run (report/surgery_examples.json)."""
    f = Path("report/surgery_examples.json")
    if not f.exists():
        return [], None
    rows = []
    for i, e in enumerate(json.loads(f.read_text())):
        detail = {
            k: e.get(k)
            for k in (
                "chose",
                "wanted",
                "top3_before",
                "top3_after",
                "top3_after_unembed",
                "text_before",
                "text_after",
                "bystanders",
                "stratum",
                "event_number",
            )
        }
        rows.append(
            row(
                "surgery_ex",
                f"ex{i}",
                e.get("context"),
                e.get("kind"),
                x=e.get("rank_before"),
                y=e.get("edit_size"),
                aux=e.get("n_rules"),
                z=e.get("dx_over_x"),
                detail=json.dumps(detail, ensure_ascii=False),
                updated=iso(f.stat().st_mtime),
            )
        )
    return rows, f.stat().st_mtime


def surgery_rows():
    rows, logs = [], [OUT / "surgery_collect.log", OUT / "surgery_exp.log"]
    for log in logs:
        if not log.exists():
            continue
        for i, line in enumerate(
            log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        ):
            if line.strip():
                rows.append(
                    row(
                        "surgery",
                        f"{log.stem}-{i}",
                        line.strip()[:200],
                        log.stem,
                        updated=iso(log.stat().st_mtime),
                    )
                )
    res, t = surgery_result_rows()
    ex, t2 = surgery_example_rows()
    return rows + res + ex, max([x for x in (newest(logs), t, t2) if x] or [None])


def main(out: str, notes_path: str | None):
    notes = (
        json.loads(Path(notes_path).read_text())
        if notes_path and Path(notes_path).exists()
        else {}
    )
    now = time.time()
    rows = []
    for name, task, fn in (
        (
            "dust",
            "Can a fuzzy LM be trained by a DUST-style forward-only gradient, and is it faster?",
            dust_rows,
        ),
        (
            "ptq",
            "How much does simple post-training quantisation cost the best 100K model?",
            ptq_rows,
        ),
        (
            "surgery",
            "Can a wrong next-piece be corrected by editing rules, with little collateral damage?",
            surgery_rows,
        ),
    ):
        part, last = fn()
        state = (
            "idle" if last is None else "running" if now - last < 1800 else "waiting"
        )
        if str(notes.get(name) or "").startswith(("Final", "Run finished")):
            state = "done"
        rows.append(
            row(
                "agent",
                name,
                task,
                state,
                x=len(
                    [r for r in part if r["panel"] in ("ptq", "dust_run", "surgery")]
                ),
                note=notes.get(name),
                updated=iso(last) if last else None,
            )
        )
        rows += part
    rows.append(
        row("meta", "refreshed", "refreshed", None, note=iso(now), updated=iso(now))
    )
    Path(out).write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
