"""Automatic text metrics for the blinded coherence sets, and how well they track the raters.

    .venv/bin/python -m flm.coherence_auto

Per text: repeat3 (share of word trigrams already seen earlier in the text), oov (share of words not among the
words seen at least 5 times in the TinyStories validation text), names (distinct capitalised names introduced after
"named"/"called" plus sentence-initial names), sent_rep (share of sentences that repeat an earlier one verbatim).
Then the correlation of each metric with the mean rater total and plot over every text of every set that has a key.
Registered before running (CPU session, 2026-10-12): repeat3 correlates with the rater total at |r| >= 0.4 (60%); no
metric reaches |r| >= 0.7 (80%); oov correlates more strongly with grammar than with plot (65%).
"""

from __future__ import annotations

import collections
import json
import re
import statistics as st
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "report"
DATA = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "tinystories"
    / "TinyStoriesV2-GPT4-valid.txt"
)
SETS = {
    "1": ["coherence_scores.json", "coherence_scores_rater2.json"],
    "2": ["coherence2_scores_rater1.json", "coherence2_scores_rater2.json"],
    "4": ["coherence4_scores_rater_gpu.json", "coherence4_scores_rater_cpu.json"],
    "5": ["coherence5_scores_rater1.json", "coherence5_scores_rater2.json"],
    "7": ["coherence7_scores_rater_gpu.json", "coherence7_scores_rater_cpu.json"],
    "8": ["coherence8_scores_rater_cpu.json"],
}
WORD = re.compile(r"[A-Za-z']+")


def vocab() -> set[str]:
    c = collections.Counter(
        w.lower() for w in WORD.findall(DATA.read_text(errors="ignore")[:5_000_000])
    )
    return {w for w, n in c.items() if n >= 5}


def metrics(text: str, V: set[str]) -> dict:
    text = text.split("<|eot|>")[0] if text.count("<|eot|>") else text
    text = text.replace("\x1f", " ")
    words = [w.lower() for w in WORD.findall(text)]
    tri = list(zip(words, words[1:], words[2:]))
    seen, rep = set(), 0
    for t in tri:
        rep += t in seen
        seen.add(t)
    sents = [s.strip().lower() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    ss, srep = set(), 0
    for s in sents:
        srep += s in ss
        ss.add(s)
    names = set(re.findall(r"(?:named|called) ([A-Z][a-z]+)", text)) | set(
        re.findall(r"(?:^|[.!?] )([A-Z][a-z]+) (?:was|said|went|saw)", text)
    )
    return {
        "repeat3": rep / max(1, len(tri)),
        "oov": sum(w not in V for w in words) / max(1, len(words)),
        "names": float(len(names)),
        "sent_rep": srep / max(1, len(sents)),
    }


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main():
    V = vocab()
    rows = []
    for s, files in SETS.items():
        blind = {
            r["id"]: r["text"]
            for r in json.loads(
                (
                    REPORT / ("coherence%s_blind.json" % ("" if s == "1" else s))
                ).read_text()
            )
        }
        scores = [json.loads((REPORT / f).read_text()) for f in files]
        for i, text in blind.items():
            m = metrics(text, V)
            m["total"] = st.mean(sum(sc[i].values()) for sc in scores)
            m["plot"] = st.mean(sc[i]["P"] for sc in scores)
            m["gram"] = st.mean(sc[i]["G"] for sc in scores)
            rows.append(m)
    print(f"{len(rows)} texts, {len(V)} vocabulary words")
    print("metric      r(total)  r(plot)  r(grammar)")
    for k in ("repeat3", "oov", "names", "sent_rep"):
        x = [r[k] for r in rows]
        print(
            f"{k:10s}  {corr(x, [r['total'] for r in rows]):+.2f}     {corr(x, [r['plot'] for r in rows]):+.2f}    {corr(x, [r['gram'] for r in rows]):+.2f}"
        )
    print(
        "means: "
        + ", ".join(
            f"{k} {st.mean(r[k] for r in rows):.3f}"
            for k in ("repeat3", "oov", "names", "sent_rep", "total", "plot")
        )
    )


if __name__ == "__main__":
    main()
