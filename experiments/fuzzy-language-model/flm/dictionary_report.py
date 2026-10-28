"""Builds the dictionary-design report: data.json and index.html in one folder.  python -m flm.dictionary_report OUT_DIR

Run from experiments/fuzzy-language-model of the worktree; reads the run logs under outputs/ and writes the page and its data into OUT_DIR (a git-ignored folder in the current directory, so the artifact tool can upload both).
"""

import glob
import json
import os
import re
import sys
from statistics import mean

import numpy as np

from flm.canonical_dictionary_oh import oh48, squeeze32
from flm.data import decode

VAL = re.compile(r"val_bpc=([\d.]+)")


def label(i):
    return {0: "EOT", 1: "UNK", 2: "\\n"}.get(i) or decode([i])


def finals(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        m = VAL.findall(open(p, errors="ignore").read())
        if m:
            out.append(float(m[-1]))
    return out


def stat(vals):
    return dict(mean=mean(vals), seeds=vals) if vals else None


def rows(controls):
    R = []

    def add(name, group, width, rules, pattern, ctrl, note=""):
        s = stat(finals(pattern))
        R.append(dict(name=name, group=group, width=width, rules=rules, control=ctrl, result=s, note=note, vs=(s["mean"] - controls[ctrl]) if s and controls.get(ctrl) else None))

    add("Constructed, one flag per letter", "frozen", 64, 124, "outputs/canon/canonFrozen_s*/ci_*.log", "w64", "26 letter flags, every coordinate named")
    add("Random vectors", "frozen", 64, 124, "outputs/canon/randFrozen_s*/ci_*.log", "w64", "control: no structure")
    add("Constructed, letters as a 5-bit code", "frozen", 64, 124, "outputs/hand32/canon64HFrozen_s*/ci_*.log", "w64", "only the letter coding changed")
    add("Hand-built 32, letters as a 5-bit code", "frozen", 32, 328, "outputs/hand32/hand32Frozen_r328_s*/ci_*.log", "w32", "")
    add("Same, letter distances doubled", "frozen", 32, 328, "outputs/hand32/hand32LFrozen_r328_s*/ci_*.log", "w32", "scaling does not fix it")
    add("oh48: a flag per letter, 48 dimensions", "frozen", 48, 196, "outputs/ohsq/oh48Frozen_s*/ci_*.log", "w48", "every coordinate named")
    add("squeeze32: a flag per letter, 32 dimensions", "frozen", 32, 328, "outputs/ohsq/squeeze32Frozen_s*/ci_*.log", "w32", "26 flags plus a 6-bit code for the rest")
    add("Constructed (64), as starting value", "start", 64, 124, "outputs/canon/canonInit_s*/ci_*.log", "w64", "trains afterwards")
    add("Hand-built 32 (5-bit letters), as starting value", "start", 32, 328, "outputs/hand32/hand32Init_r328_s*/ci_*.log", "w32", "")
    add("oh48, as starting value", "start", 48, 196, "outputs/ohsq/oh48Init_s*/ci_*.log", "w48", "")
    add("squeeze32, as starting value", "start", 32, 328, "outputs/ohsq/squeeze32Init_s*/ci_*.log", "w32", "")
    return R


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    c = {
        "w64": mean(finals("outputs/canon/learned_s*/ci_*.log")),
        "w32": mean(finals("outputs/width/learned32_s*/ci_*.log")),
        "w48": mean(finals("outputs/ohsq/learned48_s*/ci_*.log")) if finals("outputs/ohsq/learned48_s*/ci_*.log") else None,
    }
    ladder = []
    for w, pat, r in (
        (64, "outputs/canon/learned_s*/ci_*.log", 124),
        (56, "outputs/width/learned56_s*/ci_*.log", 156),
        (32, "outputs/width/learned32_s*/ci_*.log", 328),
        (24, "outputs/grid/d24_r456_s*/ci_*.log", 456),
        (16, "outputs/width/learned16_s*/ci_*.log", 706),
        (8, "outputs/width/learned8_s*/ci_*.log", 1443),
    ):
        v = finals(pat)
        ladder.append(dict(width=w, rules=r, mean=mean(v) if v else None, seeds=v))
    diag = json.load(open("outputs/hand32/diagnose.json")) if os.path.exists("outputs/hand32/diagnose.json") else {}
    classes = ["lower", "upper", "mark", "space", "newline", "special"]
    dg = {k: {cl: v["next"].get(cl, {}).get("contrib") for cl in classes} for k, v in diag.items() if k in ("learned", "frozen")}

    layouts = {}
    for name, f in (("oh48", oh48), ("squeeze32", squeeze32)):
        C = f()
        d = np.linalg.norm(C[:, None] - C[None], axis=-1)
        nn = {}
        for ch in "ae3.(, ":
            i = ord(ch) - 29
            nn[ch] = [label(int(j)) for j in np.argsort(d[i])[1:5]]
        layouts[name] = nn
    data = dict(controls=c, rows=rows(c), ladder=ladder, diagnostic=dg, neighbours=layouts)
    js = json.dumps(data, ensure_ascii=False)
    open(os.path.join(out, "data.json"), "w", encoding="utf-8").write(js)
    tpl = open("flm/dictionary_report_template.html", encoding="utf-8").read()
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", js))
    print("wrote", out, "rows with results:", sum(1 for r in data["rows"] if r["result"]))


if __name__ == "__main__":
    main()
