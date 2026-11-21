"""Arms for the Pile sample (`PLAN_PILE.md`): the F7 allocation (head 640, tail 8, d32) with the feed-forward rules refitted to the parameter cap for the vocabulary of the Pile tokenizer, written in the format of ``report/shape_arms.json``.

``python -m flm.pile_arms [--out report/pile_arms.json] [name:V:NH:E:d ...]``  (default: trained BPEs of 8,192 and 6,144; Pythia's 50,277-token vocabulary cannot fit: its embedding alone is above 139K parameters, see PLAN_PILE.md).
"""

import argparse
import json
from pathlib import Path

from flm.shape_budget import fit

DEFAULT = ["P_bpe6144:6144:640:8:32", "P_bpe8192:8192:512:6:32"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default="report/pile_arms.json")
    a = ap.parse_args()
    table = {}
    for spec in a.arms:
        name, V, NH, E, d = spec.split(":")
        V, NH, E, d = int(V), int(NH), int(E), int(d)
        fitted = fit(V=V, NH=NH, E=E, d=d)
        if fitted is None:
            raise SystemExit(f"{name}: no rule count fits the cap at V={V} NH={NH}")
        r, r0, params = fitted
        table[name] = dict(V=V, NH=NH, E=E, d=d, rules=r, rules0=r0, params=params)
        print(name, table[name])
    Path(a.out).write_text(json.dumps(table, indent=1))
