"""Progress of one training run as a JSON table for a status dashboard: ``python -m flm.run_status <log> <out.json> [--total-chars 300e6 --ref report/pythia_pile_folded.json --params 14067715 --lr 4e-2 --name ...]``.

Reads the run's training log (the ``step N/T train X val Y t=Zs`` lines `flm.train` prints at every evaluation) and, when given, the checkpoint directory next to it and a reference score; nothing is typed in.
Rows (columns ``panel, id, label, group, x, y, aux, z, note, updated``): ``eval`` (x step, y validation bits per character, aux training seconds so far, z training bits per token, smoothed), ``ref`` (a reference score, y), ``meta`` (``progress``: x step, y total steps, aux elapsed training seconds, z estimated seconds left; ``config``: note; ``checkpoint``: x step of the newest checkpoint, updated).
The estimate of the time left is the remaining steps at the seconds per step of the last four evaluations.
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

PAT = re.compile(r"^step (\d+)/(\d+) train ([0-9.]+) val ([0-9.]+) t=(\d+)s")


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        note=note,
        updated=updated,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-label", default="Pythia-14M, scored on the same text")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--config", default="", help="one-line description of the run")
    a = ap.parse_args()
    log = Path(a.log)
    rows, evals = [], []
    for line in (
        log.read_text(encoding="utf-8", errors="replace").splitlines()
        if log.exists()
        else []
    ):
        m = PAT.match(line)
        if m:
            evals.append(tuple(map(float, m.groups())))
    t_now = time.time()
    for step, total, train, val, secs in evals:
        rows.append(
            row(
                "eval",
                f"e{int(step)}",
                f"step {int(step)}",
                "validation",
                x=int(step),
                y=val,
                aux=int(secs),
                z=train,
                updated=iso(log.stat().st_mtime),
            )
        )
    if a.ref and Path(a.ref).exists():
        d = json.loads(Path(a.ref).read_text())
        c = next(iter(d["corpora"].values()))
        rows.append(
            row(
                "ref",
                "pythia",
                a.ref_label,
                "reference",
                y=round(c["bpc"], 4),
                note=f"{d['params']:,} parameters",
                updated=iso(Path(a.ref).stat().st_mtime),
            )
        )
    if evals:
        step, total, _, _, secs = evals[-1]
        recent = evals[-4:]
        sps = (
            (recent[-1][4] - recent[0][4]) / max(1.0, recent[-1][0] - recent[0][0])
            if len(recent) > 1
            else secs / step
        )
        left = (total - step) * sps
        rows.append(
            row(
                "meta",
                "progress",
                "progress",
                "run",
                x=int(step),
                y=int(total),
                aux=int(secs),
                z=int(left),
                note=f"{100 * step / total:.0f}% of the steps",
                updated=iso(log.stat().st_mtime),
            )
        )
    else:
        rows.append(
            row("meta", "progress", "progress", "run", note="no evaluation yet")
        )
    if a.config:
        rows.append(row("meta", "config", "config", "run", note=a.config))
    if a.ckpt_dir and Path(a.ckpt_dir).exists():
        ck = sorted(
            Path(a.ckpt_dir).glob("step_*.ckpt"),
            key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
        )
        if ck:
            rows.append(
                row(
                    "meta",
                    "checkpoint",
                    "newest checkpoint",
                    "run",
                    x=int(re.search(r"step_(\d+)", ck[-1].name).group(1)),
                    updated=iso(ck[-1].stat().st_mtime),
                )
            )
    rows.append(
        row("meta", "refreshed", "refreshed", None, note=iso(t_now), updated=iso(t_now))
    )
    Path(a.out).write_text(json.dumps(rows))
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
