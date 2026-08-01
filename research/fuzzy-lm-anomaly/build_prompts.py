"""Stage 1 -- build the probe set.

Two families, both giving ground truth without a human in the loop:

* ``triviaqa``  -- closed-book factual QA. Grading is exact-match against the
  gold aliases, so a wrong-but-confident answer is a hallucination and a right
  answer is a sample of known-good behaviour.
* ``falsepremise`` -- questions whose subject does not exist. Any substantive
  answer is necessarily fabricated; the only correct behaviour is pushback.

Entities in the false-premise family are assembled from invented syllables so
that no accidental real-world referent can make the label wrong.

Writes ``prompts.jsonl``.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"

# --- invented entity vocabulary -------------------------------------------
# Deliberately synthetic: consonant clusters and endings that do not collide
# with real names, places, or terminology.
_FIRST = ["Varabel", "Quenteth", "Draviel", "Mornuth", "Kelverin", "Ysandor",
          "Tobrecht", "Halvenne", "Perimund", "Sarquist", "Volmerath", "Endriche"]
_LAST = ["Thrennick", "Oquendel", "Vastrimer", "Belhoun", "Karsivett", "Dunmorrow",
         "Ashkelvane", "Prendaux", "Wexlingham", "Ombrathe", "Silvenwood", "Corrathine"]
_FIELDS = ["Interpretive Dance", "Applied Cryptobotany", "Recursive Gastronomy",
           "Theoretical Lexicography", "Quantum Cartography", "Speculative Metallurgy",
           "Comparative Hydrolinguistics", "Ornamental Topology"]
_COUNTRIES = ["Brazendia", "Kelmoravia", "Ondasqua", "Vurenthal", "Tashmiria",
              "Grellandine", "Ospravia", "Mordaquesne"]
_ELEMENTS = ["Vandalium", "Threquine", "Oscadium", "Merluvium", "Kaspheron", "Ytterbane"]
_ALGOS = ["Drennick-Vasquez partition", "reverse Holbrand sieve", "Kessendorf projection",
          "adaptive Molverin descent", "Tanager-Bell decomposition"]
_TOPICS = ["marine biology", "medieval agriculture", "orbital mechanics", "phonology",
           "immunology", "civil engineering", "cognitive psychology"]


def _person(rng):
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def build_false_premise(n, seed=0):
    """Templated questions with fabricated subjects. Any answer is a hallucination."""
    rng = random.Random(seed)
    rows, seen = [], set()
    makers = [
        lambda: (f"Who won the {rng.randint(1950, 2015)} Nobel Prize in {rng.choice(_FIELDS)}?",
                 "fake_award"),
        lambda: (f"What is the researcher {_person(rng)} best known for?", "fake_person"),
        lambda: (f"Summarize the main finding of the {rng.randint(1985, 2020)} paper "
                 f"\"{rng.choice(['On the', 'Toward a', 'Revisiting the'])} "
                 f"{rng.choice(_ALGOS).title()}\" by {_person(rng)}.", "fake_paper"),
        lambda: (f"What is the capital city of {rng.choice(_COUNTRIES)}?", "fake_place"),
        lambda: (f"What are the chemical properties of the element {rng.choice(_ELEMENTS)}?",
                 "fake_element"),
        lambda: (f"Explain how the {rng.choice(_ALGOS)} algorithm works.", "fake_method"),
        lambda: (f"Why did {_person(rng)} abandon their research into {rng.choice(_TOPICS)}?",
                 "fake_event"),
        lambda: (f"Which treaty ended the {rng.randint(1600, 1900)} "
                 f"{rng.choice(_COUNTRIES)}n Succession War?", "fake_event"),
    ]
    for maker in itertools.cycle(makers):
        if len(rows) >= n:
            break
        q, sub = maker()
        if q in seen:
            continue
        seen.add(q)
        rows.append({"id": f"fp-{len(rows):05d}", "family": "falsepremise",
                     "subtype": sub, "question": q, "answers": []})
    return rows


def build_triviaqa(n, seed=0):
    """Closed-book QA. ``answers`` holds every accepted alias for exact match."""
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rows = []
    for ex in ds:
        if len(rows) >= n:
            break
        ans = ex["answer"]
        aliases = list({ans["value"], *ans.get("aliases", [])})
        if not ans["value"].strip():
            continue
        rows.append({"id": f"tq-{len(rows):05d}", "family": "triviaqa",
                     "subtype": "closedbook", "question": ex["question"],
                     "answers": aliases})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trivia", type=int, default=1200)
    ap.add_argument("--n-false", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_triviaqa(args.n_trivia, args.seed) + build_false_premise(args.n_false, args.seed)

    path = OUT / "prompts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_fp = sum(r["family"] == "falsepremise" for r in rows)
    print(f"wrote {len(rows)} prompts -> {path}")
    print(f"  triviaqa     : {len(rows) - n_fp}")
    print(f"  falsepremise : {n_fp}")


if __name__ == "__main__":
    main()
