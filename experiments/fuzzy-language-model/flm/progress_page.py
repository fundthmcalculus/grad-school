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
    grid = runs("outputs/grid/*/ci_*.log")
    old = {  # cells already measured (learned dictionary): (width, rules) -> mean of seeds 0 and 1
        (64, 124): l64,
        (32, 328): avg(wid, "learned32")[0],
        (16, 706): avg(wid, "learned16")[0],
    }

    def gcell(d, r):
        if (d, r) in old:
            return f"{old[(d, r)]:.4f}" if old[(d, r)] else ""
        v = [grid[f"d{d}_r{r}_s{s}"] for s in (0, 1) if f"d{d}_r{r}_s{s}" in grid]
        fin = [x[3] for x in v if x[3]]
        if len(fin) == 2:
            return f"{mean(fin):.4f}"
        if fin:
            return f"{fin[0]:.4f} (1 seed)"
        return "running" if any(x[0] for x in v) else "queued"

    params = {64: ("99.6K", "207K", "406K"), 32: ("46K", "99.9K", "200K"), 24: ("33.9K", "74.4K", "149K"), 16: ("22.3K", "49.5K", "99.6K")}
    rows = []
    for d in (64, 32, 24, 16):
        rows.append([str(d)] + [f"{gcell(d, r)} <small>{params[d][i]}</small>" for i, r in enumerate((124, 328, 706))])
    rows.append(["24 at its 100K shape", f"{gcell(24, 456)} <small>99.8K, 456 rules</small>", "", ""])
    t3 = table(["width", "124 rules", "328 rules", "706 rules"], rows)
    hand = runs("outputs/hand32/*/ci_*.log")
    ctrl = {328: avg(wid, "learned32")[0], 706: avg(grid, "d32_r706")[0]}

    def hcell(a, r):
        v = [hand[f"{a}_r{r}_s{s}"] for s in (0, 1) if f"{a}_r{r}_s{s}" in hand]
        fin = [x[3] for x in v if x[3]]
        if len(fin) == 2:
            m = mean(fin)
            return f"{m:.4f} <small>{m - ctrl[r]:+.4f} vs learned</small>"
        if fin:
            return f"{fin[0]:.4f} (1 seed)"
        return "running" if any(x[0] for x in v) else "queued"

    rows = [
        [f"{r} rules", f"{ctrl[r]:.4f}" if ctrl[r] else "", hcell("hand32Frozen", r), hcell("hand32Init", r)]
        for r in (328, 706)
    ]
    t4 = table(["width 32", "learned dictionary", "hand-built, frozen", "hand-built, as start"], rows)
    osq = runs("outputs/ohsq/*/ci_*.log")

    def ocell(tag, ref=None):
        v = [osq[f"{tag}_s{s}"] for s in (0, 1) if f"{tag}_s{s}" in osq]
        fin = [x[3] for x in v if x[3]]
        if len(fin) == 2:
            m = mean(fin)
            return f"{m:.4f}" + (f" <small>{m - ref:+.4f}</small>" if ref else "")
        if fin:
            return f"{fin[0]:.4f} (1 seed)"
        return "running" if any(x[0] for x in v) else "queued"

    l48 = avg(osq, "learned48")[0]
    rows = [
        ["width 48, 196 rules (100.1K)", ocell("learned48"), ocell("oh48Frozen", l48), ocell("oh48Init", l48)],
        ["width 32, 328 rules (99.9K)", f"{ctrl[328]:.4f}" if ctrl[328] else "", ocell("squeeze32Frozen", ctrl[328]), ocell("squeeze32Init", ctrl[328])],
    ]
    t5 = table(["one flag per letter", "learned dictionary", "frozen", "as starting value"], rows)
    qrows = []
    for grp, label in (
        ("oh48_300", "oh48, 300M TinyStories"),
        ("oh40", "oh40, 30M"),
        ("pull", "oh48 drift penalty, 30M"),
        ("oh48_pile", "oh48, 300M Pile"),
        ("final100k", "final model, 300M Cosmopedia"),
    ):
        for name, r in sorted(runs(f"outputs/{grp}/*/ci_*.log").items()):
            c = cell(r)
            qrows.append([label, name, c[0], c[1]])
    t6 = table(["study", "run", "validation", "state"], qrows or [["queued", "", "", ""]])
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
small{{color:var(--mut);margin-left:6px}}
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
<h2>Rules against width</h2>
<p>Learned dictionary, 30M characters, mean of two seeds. Rules are per feed-forward system; the small figure is the parameter count. This separates the effect of the dictionary width from the effect of the rule count.</p>
{t3}
<h2>Hand-built 32-wide dictionary</h2>
<p>Every coordinate named. Frozen, or used as the starting value; 30M characters, mean of two seeds. The small figure is the difference from the learned dictionary at the same shape (lower is better).</p>
{t4}
<h2>Equal-distance letters: how few dimensions?</h2>
<p>A frozen dictionary lost 0.26 to 0.5 bits when letters were a 5-bit code, because some letters looked alike for no reason. These give each letter its own flag. The 48-wide one has a named coordinate for everything; the 32-wide one squeezes the non-letters into a 6-bit code (the floor for one flag per letter). The small figure is the difference from the learned dictionary at that shape (lower is better).</p>
{t5}
<h2>On the GPU now and queued</h2>
<p>Every run in the current queue, in order of start. Validation is the latest evaluation until the run is done.</p>
{t6}
"""
    open(sys.argv[1] if len(sys.argv) > 1 else "outputs/progress/index.html", "w", encoding="utf-8").write(page)


if __name__ == "__main__":
    main()
