"""Builds the final-model page: data.json and index.html in one folder.  python -m flm.final_report OUT_DIR

Run from experiments/fuzzy-language-model of the worktree; reads outputs/final100k, outputs/oh48_300, outputs/oh48_pile, outputs/pull300 and outputs/final_eval.
"""

import glob
import json
import os
import re
import sys
from statistics import mean

VAL = re.compile(r"val_bpc=([\d.]+)")


def finals(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        m = VAL.findall(open(p, errors="ignore").read())
        if m:
            out.append(float(m[-1]))
    return out


def arm(*patterns):
    v = [x for p in patterns for x in finals(p)]
    return dict(seeds=v, mean=mean(v)) if v else None


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cos = {
        "learned dictionary": arm("outputs/final100k/learned_s*/ci_*.log"),
        "frozen named dictionary": arm("outputs/final100k/frozen_s*/ci_*.log"),
        "named dictionary as starting value": arm("outputs/final100k/init_s*/ci_*.log"),
        "named dictionary, weak pull-back (0.0001 per step)": arm("outputs/pull_retune/k0.0001_s0/ci_*.log", "outputs/pull_confirm/cosmo_k0.0001_s1/ci_*.log"),
        "tied head (not invertible)": arm("outputs/final100k/tied_s*/tied_*.log", "outputs/gaps/tied_cosmo_s1/tied_*.log"),
    }
    s1 = all(finals(f"outputs/pile_s1/{a}_s1/ci_*.log") for a in ("learned", "frozen", "init"))
    ps1 = lambda a: [f"outputs/pile_s1/{a}_s1/ci_*.log"] if s1 else []
    freeze = []
    for name, ln, fr, st, pl in (
        ("TinyStories", ["outputs/oh48_300/learned_s*/ci_*.log"], ["outputs/oh48_300/frozen_s*/ci_*.log"], ["outputs/oh48_300/init_s*/ci_*.log"], []),
        ("Cosmopedia", ["outputs/final100k/learned_s*/ci_*.log"], ["outputs/final100k/frozen_s*/ci_*.log"], ["outputs/final100k/init_s*/ci_*.log"], ["outputs/pull_retune/k0.0001_s0/ci_*.log", "outputs/pull_confirm/cosmo_k0.0001_s1/ci_*.log"]),
        ("The Pile", ["outputs/oh48_pile/learned_s*/ci_*.log"] + ps1("learned"), ["outputs/oh48_pile/frozen_s*/ci_*.log"] + ps1("frozen"), ["outputs/oh48_pile/init_s*/ci_*.log"] + ps1("init"), ["outputs/pull_confirm/pile_k0.0001_s0/ci_*.log", "outputs/gaps/pile_k0.0001_s1/ci_*.log"]),
    ):
        L, F, S = arm(*ln), arm(*fr), arm(*st)
        pull = arm(*pl) if pl else None
        freeze.append(dict(corpus=name, learned=L["mean"], frozen=F["mean"], start=S["mean"], pull=pull["mean"] if pull else None, n=len(L["seeds"]), npull=len(pull["seeds"]) if pull else 0))
    ci, tied = jload("outputs/final_eval/ci.json"), jload("outputs/final_eval/tied.json")
    cross = []
    for key, label in (("learned", "learned dictionary"), ("frozen", "frozen named dictionary"), ("init", "named dictionary as starting value")):
        cross.append(dict(model=label, **ci[key]["bpc"]))
    cross.append(dict(model="tied head (not invertible)", **tied["tied"]["bpc"]))
    explain = {k: jload(f"outputs/final_eval/explain_{k}.json") for k in ("frozen", "init", "tied")}
    ex = {}
    for k, j in explain.items():
        ex[k] = dict(
            effective=[round(r["effective_rules"], 1) for r in j["rules"]],
            rules=[r["rules"] for r in j["rules"]],
            mi_rule=[round(r["mi_rule_next"], 2) for r in j["rules"]],
            mi_prev=round(j["rules"][0]["mi_prev_next"], 2),
            ablate=[dict(top=round(a["top_cost"], 2), rand=round(a["random_cost_max"], 2)) for a in j["ablation"]],
        )
    samples = {k: [dict(prompt=p, text=t) for p, t in list(ci[k]["greedy"].items())[:5]] for k in ("frozen", "init")}
    samples["tied"] = [dict(prompt=p, text=t) for p, t in list(tied["tied"]["greedy"].items())[:5]]
    data = dict(cosmo=cos, freeze=freeze, cross=cross, explain=ex, samples=samples)
    js = json.dumps(data, ensure_ascii=False)
    open(os.path.join(out, "data.json"), "w", encoding="utf-8").write(js)
    tpl = open("flm/final_report_template.html", encoding="utf-8").read()
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", js))
    print("wrote", out)


if __name__ == "__main__":
    main()
