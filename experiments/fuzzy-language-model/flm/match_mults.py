"""Size-match the neural reference to the fuzzy model: write flm/matched_mults.json.

    .venv/bin/python -m flm.match_mults

The goal-track fuzzy model D (flm.grids_v3.D, delta-form fuzzy mixer, 8 heads, first-order
raw-firing HTSK TSK FFN, short-conv 8) has a fixed parameter count at each width. A neural
reference with an MLP FFN is padded to the same count by scaling the MLP width (``--ffn-mult``),
searched in steps of 0.01 against the real parameter count of the instantiated model. Without
this the reference sits 4% below the fuzzy model at d = 72 (93.3K vs 97.5K), a mismatch that
favours the fuzzy model.
"""

from __future__ import annotations

import json
from pathlib import Path

from flm.models import TinyLM
from flm.ref_space import NEURAL, SHORTCONVS, WIDTHS
from flm.train import get_args, make_config

OUT = Path(__file__).resolve().parent / "matched_mults.json"


def n_params(argv: list[str]) -> int:
    return TinyLM(make_config(get_args(argv + ["--out", "unused.json"]))).n_params()


def match(argv: list[str], target: int) -> tuple[float, int]:
    best = None
    for i in range(100, 700):
        mult = i / 100
        p = n_params(argv + ["--ffn-mult", str(mult)])
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (mult, p)
    return best


def main() -> None:
    from flm.grids_v3 import D

    out = {"target": {}, "mult": {}, "params": {}}
    for d in WIDTHS:
        target = n_params(D + ["--d", str(d), "--layers", "2", "--shortconv", "8"])
        out["target"][f"d{d}|L2"] = target
        for name, argv in NEURAL.items():
            for sc in SHORTCONVS:
                base = argv + ["--d", str(d), "--layers", "2", "--shortconv", str(sc)]
                mult, p = match(base, target)
                out["mult"][f"{name}|sc{sc}|d{d}|L2"] = mult
                out["params"][f"{name}|sc{sc}|d{d}|L2"] = p
    # a 3-layer neural variant at d = 56, matched to D's 97.5K (depth was searched for the fuzzy model)
    target = out["target"]["d72|L2"]
    for sc in SHORTCONVS:
        base = NEURAL["delta-h8"] + [
            "--d",
            "56",
            "--layers",
            "3",
            "--shortconv",
            str(sc),
        ]
        mult, p = match(base, target)
        out["mult"][f"delta-h8|sc{sc}|d56|L3"] = mult
        out["params"][f"delta-h8|sc{sc}|d56|L3"] = p
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    worst = max(
        abs(out["params"][k] - out["target"][k.split("|")[2] + "|" + k.split("|")[3]])
        / out["target"][k.split("|")[2] + "|" + k.split("|")[3]]
        for k in out["params"]
        if k.endswith("L2")
    )
    print(
        f"wrote {len(out['mult'])} matched configs to {OUT}; worst L2 mismatch {worst:.2%}"
    )


if __name__ == "__main__":
    main()
