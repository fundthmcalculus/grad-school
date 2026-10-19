"""Summarize a screen: python -m flm.screen v3-attrib [--seeds]

Plain: every cell sorted by val BPC. --seeds: cells named <arm>_s<k> are grouped by arm
and reported as mean +/- std over seeds.
"""

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "outputs" / sys.argv[1]
recs = [(f.stem, json.loads(f.read_text())) for f in sorted(d.glob("*.json"))]
if "--seeds" in sys.argv:
    g = defaultdict(list)
    for n, r in recs:
        g[re.sub(r"_s\d+$", "", n)].append((r["val_bpc"], r["params_total"]))
    for n, v in sorted(g.items(), key=lambda kv: st.mean(x[0] for x in kv[1])):
        b = [x[0] for x in v]
        sd = st.pstdev(b) if len(b) > 1 else float("nan")
        print(
            f"{st.mean(b):.4f} ± {sd:.4f}  {v[0][1]:>8,}  n={len(b)}  {[round(x, 3) for x in b]}  {n}"
        )
else:
    rows = [(r["val_bpc"], n, r["params_total"], r["chars_per_sec"]) for n, r in recs]
    for b, n, p, c in sorted(rows, key=lambda x: (x[0] != x[0], x[0])):
        print(f"{b:8.4f}  {p:>8,}  {c:>9,}  {n}")
