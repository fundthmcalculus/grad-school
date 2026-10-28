"""Builds the in-progress page from the run logs: python -m flm.progress_page [out.html]"""

import glob
import html
import re
import sys
import time
from statistics import mean

FINAL = re.compile(r"val_bpc=([\d.]+)")
STEP = re.compile(r"step (\d+)/(\d+) train [\d.]+ val ([\d.]+) t=(\d+)s")


def read(path):
    """(step, total, val, final) of one run from its log."""
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    fin = FINAL.findall(txt)
    st = STEP.findall(txt)
    step, total, val, _ = st[-1] if st else (0, 0, "nan", 0)
    return int(step), int(total), float(val), float(fin[-1]) if fin else None


def runs(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        out[p.replace("\\", "/").split("/")[-2]] = read(p)
    return out


def num(r):
    return r[3] if r[3] is not None else (r[2] if r[0] else None)


def cell(r):
    if not r[0]:
        return "queued", ""
    if r[3] is not None:
        return f"{r[3]:.4f}", "done"
    return f"{r[2]:.4f}", f"{100 * r[0] // r[1]}%"


def table(head, rows):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<div class="scroll"><table><tr>{h}</tr>{b}</table></div>'


def main():
    c300 = runs("outputs/canon300/*/ci_*.log")
    wid = runs("outputs/width/*/ci_*.log")
    ref = runs("outputs/canon/*/ci_*.log")
    rows = []
    for s in (0, 1, 2):
        a, b = c300.get(f"learned_s{s}"), c300.get(f"canonInit_s{s}")
        if not a or not b:
            rows.append([f"seed {s}", "queued", "queued", "", ""])
            continue
        na, nb = num(a), num(b)
        gap = f"{na - nb:+.4f}" if na and nb else ""
        state = "done" if a[3] and b[3] else f"{100 * min(a[0], b[0]) // max(1, a[1])}% (interim)"
        rows.append([f"seed {s}", cell(a)[0], cell(b)[0], gap, state])
    t1 = table(["300M characters", "learned", "constructed start", "gain", "state"], rows)

    def avg(d, key, seeds=(0, 1)):
        v = [d[f"{key}_s{s}"][3] for s in seeds if f"{key}_s{s}" in d and d[f"{key}_s{s}"][3]]
        return (mean(v), len(v)) if v else (None, 0)

    l64, _ = avg(ref, "learned")
    c64, _ = avg(ref, "canonInit")
    rows = [["64 (earlier run)", f"{l64:.4f}" if l64 else "", f"{c64:.4f}" if c64 else "", "0 (reference)", "2/2"]]
    for k in (56, 32, 16, 8):
        l, nl = avg(wid, f"learned{k}")
        p, npr = avg(wid, f"pca{k}")
        part = [wid.get(f"{n}{k}_s{s}") for n in ("learned", "pca") for s in (0, 1)]
        live = sum(1 for r in part if r and r[0] and not r[3])
        rows.append(
            [
                str(k),
                f"{l:.4f}" if l else "",
                f"{p:.4f}" if p else "",
                f"{p - c64:+.4f}" if p and c64 else "",
                f"{nl + npr}/4" + (f", {live} running" if live else ""),
            ]
        )
    t2 = table(["width", "learned", "constructed start (projected)", "start vs width 64", "runs done"], rows)
    now = time.strftime("%Y-%m-%d %H:%M")
    page = f"""<title>Run progress</title>
<style>
:root{{--bg:#f6f4ef;--fg:#1d2420;--mut:#5d675f;--line:#d9d5ca;--acc:#2f6b57;--card:#fffdf8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151a17;--fg:#e6ebe6;--mut:#99a49b;--line:#2e3731;--acc:#7cc4a8;--card:#1c231f;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#151a17;--fg:#e6ebe6;--mut:#99a49b;--line:#2e3731;--acc:#7cc4a8;--card:#1c231f;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding-block:24px;padding-inline:16px;max-width:760px;margin:0 auto}}
h1{{font-size:1.4rem;margin:0 0 4px}} h2{{font-size:1.05rem;margin:28px 0 6px}}
p{{color:var(--mut);margin:4px 0 10px;max-width:65ch}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;background:var(--card)}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums;white-space:nowrap}}
th{{color:var(--mut);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}}
</style>
<h1>Run progress</h1>
<p>Updated {now}. Validation bits per character, lower is better; 100K-parameter invertible model.</p>
<h2>Does the constructed dictionary hold up with more training?</h2>
<p>Learned dictionary against the constructed one used as the starting value. A positive gain means the constructed start is better.</p>
{t1}
<h2>How few dimensions?</h2>
<p>30M characters, two seeds. The constructed dictionary is projected to the width and used as the starting value; the control learns its dictionary. Rule counts grow as the width shrinks to keep 100K parameters.</p>
{t2}
"""
    open(sys.argv[1] if len(sys.argv) > 1 else "outputs/progress/index.html", "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    main()
