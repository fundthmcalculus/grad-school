"""Stage 9 -- audit the abstention regex.

The refusal/fabrication split is decided by a hand-written regex in `capture.py`.
It was already wrong once (it scored "as a text-based AI, I don't have the
ability to access…" as a hallucination), and errors here are not neutral: a
refusal mislabelled as a fabrication lets the detector learn "refusal template"
instead of "fabrication", which inflates every result downstream.

Rather than ask for 200 hand labels, this cross-checks the regex against an
independent structural heuristic and surfaces only the **disagreements**, which
is where the errors live and is a set small enough to read.

The second heuristic is deliberately built on different evidence: first-person
modal/negation constructions and apology openers, rather than the regex's
keyword list.

Outputs a sample for review and, if `--labels` is given, computes precision and
recall against it.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import DATA
from capture import _ABSTAIN

# Independent heuristic: structural, not keyword-based.
#  - an apology or hedge in the first clause
#  - a first-person negated capability/knowledge verb
#  - an explicit non-existence predicate applied to the subject
_H2 = [
    re.compile(r"^\s*(i'm sorry|i am sorry|sorry|unfortunately|i apologi)", re.I),
    re.compile(r"\bi\s+(?:do\s+not|don't|cannot|can't|am\s+not|'m\s+not)\s+"
               r"\w*\s*(know|have|find|recall|aware|sure|access|provide)", re.I),
    re.compile(r"\b(?:is|are|was|were)\s+not\s+(?:a\s+)?"
               r"(real|actual|recognized|recognised|known|existing)", re.I),
    re.compile(r"\b(?:doesn't|does not|didn't|did not)\s+(?:seem to\s+)?exist", re.I),
    # "X is a fictional character created by …" is a *fabrication* that happens
    # to use the word, not a refusal — so require hedged/epistemic framing.
    re.compile(r"\b(appears?\s+to\s+be|seems?\s+to\s+be|may\s+be|might\s+be|"
               r"is\s+likely|i\s+believe\s+\w+\s+is)\s+(?:a\s+)?"
               r"(fictional|fictitious|made[- ]up|invented|hypothetical)", re.I),
]


def h2(text: str) -> bool:
    return any(p.search(text) for p in _H2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="capture_v2")
    ap.add_argument("--n", type=int, default=40, help="disagreements to print")
    ap.add_argument("--labels", help="CSV with id,true_abstain for scoring")
    args = ap.parse_args()

    path = DATA / f"{args.prefix}_meta.parquet"
    if not path.exists():
        path = DATA / "capture_meta.parquet"
    meta = pd.read_parquet(path)
    print(f"auditing {path.name} ({len(meta):,} generations)")

    ans = meta.answer.fillna("").astype(str)
    r1 = ans.map(lambda t: bool(_ABSTAIN.search(t)))
    r2 = ans.map(h2)

    n = len(meta)
    print(f"\n{'='*78}\nAGREEMENT BETWEEN THE TWO HEURISTICS\n{'='*78}")
    ct = pd.crosstab(r1.rename("regex_abstain"), r2.rename("structural_abstain"))
    print(ct.to_string())
    agree = int((r1 == r2).sum())
    print(f"\nagreement: {agree:,}/{n:,} = {agree/n:.2%}")
    print(f"regex says abstain      : {int(r1.sum()):,} ({r1.mean():.2%})")
    print(f"structural says abstain : {int(r2.sum()):,} ({r2.mean():.2%})")

    # Cohen's kappa -- agreement above chance
    po = agree / n
    pe = (r1.mean() * r2.mean()) + ((1 - r1.mean()) * (1 - r2.mean()))
    print(f"Cohen's kappa           : {(po - pe) / (1 - pe):.3f}")

    dis = meta[r1 != r2].copy()
    dis["regex_abstain"] = r1[r1 != r2]
    dis["structural_abstain"] = r2[r1 != r2]
    print(f"\ndisagreements: {len(dis):,} ({len(dis)/n:.2%}) — "
          f"the error budget is bounded by this")
    print("by family:")
    print(pd.crosstab(dis.family, dis.regex_abstain).to_string())

    out = DATA / "abstain_disagreements.csv"
    dis[["id", "family", "label", "question", "answer",
         "regex_abstain", "structural_abstain"]].to_csv(out, index=False)
    print(f"\nwrote all {len(dis):,} disagreements -> {out.name}")

    rng = np.random.default_rng(0)
    for flag, title in ((True, "regex=ABSTAIN, structural=not"),
                        (False, "regex=not, structural=ABSTAIN")):
        sub = dis[dis.regex_abstain == flag]
        if not len(sub):
            continue
        print(f"\n{'='*78}\n{title}  ({len(sub):,} cases)\n{'='*78}")
        take = sub.iloc[rng.choice(len(sub), min(args.n // 2, len(sub)),
                                   replace=False)]
        for _, r in take.iterrows():
            print(f"[{r.family}/{r.label}] Q: {r.question[:72]}")
            print(f"    A: {r.answer[:150]}")

    if args.labels:
        lab = pd.read_csv(args.labels).set_index("id")["true_abstain"].astype(bool)
        j = meta.set_index("id").join(lab, how="inner")
        if len(j):
            pred = j.answer.fillna("").astype(str).map(
                lambda t: bool(_ABSTAIN.search(t)))
            tp = int((pred & j.true_abstain).sum())
            fp = int((pred & ~j.true_abstain).sum())
            fn = int((~pred & j.true_abstain).sum())
            print(f"\n{'='*78}\nSCORED AGAINST {len(j)} LABELS\n{'='*78}")
            print(f"precision {tp/max(tp+fp,1):.3f}  recall {tp/max(tp+fn,1):.3f}")


if __name__ == "__main__":
    main()
