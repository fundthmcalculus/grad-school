"""Stage 15 -- long-form template-matched probes for the false-premise niche.

§19 found the one class where the fuzzy rule beats entropy: long-form
false-premise fabrication (FIS 0.786 vs entropy 0.561). But that comparison used
TriviaQA-correct as the truthful set, so it carries exactly the confound that
falsified §9 -- long discursive fabrications against short factual answers, with
only length matched.

This builds the control. Every long-form template gets a **real-subject twin in
the identical surface form**, so both sides elicit fluent, discursive output:

    real : "Explain how the Dijkstra's shortest path algorithm works."
    fake : "Explain how the reverse Holbrand sieve algorithm works."

**Label semantics differ from v2 on purpose.** Grading a paragraph automatically
is unreliable, so the label here is *groundedness*, not correctness: the real
subject exists and the model's answer is anchored to something, the invented
subject does not exist so any answer is necessarily ungrounded. A real-subject
answer may still contain errors. The honest claim this supports is therefore
"detects ungrounded generation about a non-existent entity", which is precisely
the niche §19 identified -- not "detects factual error".

Seven templates x ~30 real subjects x invented twins.
"""

import argparse
import json
from pathlib import Path

OUT = Path(__file__).parent / "data"


def _split(s):
    return [x.strip() for x in s.strip().split(";") if x.strip()]


# --- real subjects, all long-form-eliciting ------------------------------
RESEARCHERS = _split("""
Albert Einstein; Marie Curie; Isaac Newton; Charles Darwin; Alan Turing;
Ada Lovelace; Niels Bohr; Werner Heisenberg; Rosalind Franklin;
Dmitri Mendeleev; Louis Pasteur; Gregor Mendel; Michael Faraday;
James Clerk Maxwell; Max Planck; Erwin Schrodinger; Paul Dirac; Enrico Fermi;
Richard Feynman; Linus Pauling; Barbara McClintock; Dorothy Hodgkin;
Emmy Noether; Srinivasa Ramanujan; Henri Poincare; David Hilbert; Kurt Godel;
John von Neumann; Claude Shannon; Norbert Wiener; Grace Hopper;
Edsger Dijkstra; Donald Knuth; Tim Berners-Lee; Katherine Johnson;
Chien-Shiung Wu; Lise Meitner; Subrahmanyan Chandrasekhar; Tu Youyou;
Jane Goodall
""")

ALGORITHMS = _split("""
quicksort; merge sort; heapsort; binary search; Dijkstra's shortest path;
A* search; Bellman-Ford; Floyd-Warshall; Kruskal's minimum spanning tree;
Prim's minimum spanning tree; breadth-first search; depth-first search;
Newton-Raphson; gradient descent; backpropagation; k-means clustering;
PageRank; RSA encryption; Huffman coding; Kalman filter;
fast Fourier transform; simplex; Metropolis-Hastings;
expectation-maximization; Viterbi; Knuth-Morris-Pratt; union-find;
Bloom filter; simulated annealing; Monte Carlo tree search
""")

ELEMENTS = _split("""
Hydrogen; Helium; Carbon; Nitrogen; Oxygen; Fluorine; Sodium; Magnesium;
Aluminium; Silicon; Phosphorus; Sulfur; Chlorine; Argon; Potassium; Calcium;
Titanium; Chromium; Iron; Cobalt; Nickel; Copper; Zinc; Silver; Tin; Iodine;
Tungsten; Platinum; Gold; Mercury; Lead; Uranium
""")

EFFECTS = _split("""
Doppler; Hall; Zeeman; Casimir; Compton; photoelectric; Meissner; Josephson;
Coriolis; Faraday; Seebeck; Peltier; Stark; Kerr; Raman; Mossbauer;
Aharonov-Bohm; Venturi; Magnus; piezoelectric; Barkhausen; Purkinje;
Tyndall; Zeno; Doppler-Fizeau; skin; Kondo; Hawking
""")

CONJECTURES = _split("""
Riemann; Goldbach; Collatz; Poincare; Twin Prime; Hodge; Continuum;
Four Colour; Erdos-Straus; Catalan; Legendre; Bunyakovsky; Jacobian;
Sendov; Schanuel; Beal; Firoozbakht; Cramer
""")

NOBEL_FIELDS = _split("Physics; Chemistry; Medicine; Literature; Peace")

PAPERS = [
    ("On Computable Numbers", "Alan Turing", 1936),
    ("A Mathematical Theory of Communication", "Claude Shannon", 1948),
    ("Molecular Structure of Nucleic Acids", "James Watson", 1953),
    ("On the Electrodynamics of Moving Bodies", "Albert Einstein", 1905),
    ("Computing Machinery and Intelligence", "Alan Turing", 1950),
    ("A Logical Calculus of the Ideas Immanent in Nervous Activity",
     "Warren McCulloch", 1943),
    ("Attention Is All You Need", "Ashish Vaswani", 2017),
    ("ImageNet Classification with Deep Convolutional Neural Networks",
     "Alex Krizhevsky", 2012),
    ("The Chemical Basis of Morphogenesis", "Alan Turing", 1952),
    ("Can Quantum-Mechanical Description of Physical Reality Be Considered "
     "Complete?", "Albert Einstein", 1935),
    ("A Method for the Construction of Minimum-Redundancy Codes",
     "David Huffman", 1952),
    ("Error Detecting and Error Correcting Codes", "Richard Hamming", 1950),
]

# --- invented twins, index-addressed (unique by construction) -------------
_N_A = _split("""Varabel; Quenteth; Draviel; Mornuth; Kelverin; Ysandor;
Tobrecht; Halvenne; Perimund; Sarquist; Volmerath; Endriche; Aldrimen;
Bracewold; Corvantis; Delphrane""")
_N_B = _split("""Thrennick; Oquendel; Vastrimer; Belhoun; Karsivett; Dunmorrow;
Ashkelvane; Prendaux; Wexlingham; Ombrathe; Silvenwood; Corrathine;
Marchevant; Hastrimond; Follenby; Quillemont""")
_M_A = _split("""Drennick; Holbrand; Kessendorf; Molverin; Tanager; Vasquine;
Oquell; Brathwen; Selvantis; Pertheny; Wexlar; Ombrast""")
_M_B = _split("""partition; sieve; projection; descent; decomposition; cascade;
refinement; contraction; expansion; folding; pivoting; relaxation""")
_E_A = _split("""Vandal; Threqu; Oscad; Merluv; Kaspher; Ytterb; Drennic;
Palvor; Zirbal; Norvex; Cadril; Wexlar; Ombrath; Quenth; Selvan; Turmac""")
_E_B = _split("ium; ine; on; ide; ate; ane")
_F_A = _split("""Interpretive Dance; Applied Cryptobotany; Recursive Gastronomy;
Theoretical Lexicography; Quantum Cartography; Speculative Metallurgy;
Comparative Hydrolinguistics; Ornamental Topology""")


def _p(i, a, b):
    return f"{a[i % len(a)]} {b[(i // len(a)) % len(b)]}"


def _person(i):
    return _p(i, _N_A, _N_B)


def _method(i):
    return f"{_M_A[i % len(_M_A)]}-{_M_B[(i // len(_M_A)) % len(_M_B)]}"


def _element(i):
    return _E_A[i % len(_E_A)] + _E_B[(i // len(_E_A)) % len(_E_B)]


def _hyphen(i):
    """Eponymous-looking invented name, e.g. 'Vastrimer-Belhoun' (256 distinct)."""
    return f"{_N_B[i % len(_N_B)]}-{_N_B[(i // len(_N_B)) % len(_N_B)]}"


def _surname(i):
    return _N_B[i % len(_N_B)]


# Three surface forms per template, applied IDENTICALLY to real and invented
# subjects, so phrasing is balanced by construction and multiplies the scarce
# real side 3x.
TEMPLATES = {
    "researcher": (["What is the researcher {} best known for?",
                    "Describe the main contributions of {}.",
                    "Why is {} historically significant?"],
                   RESEARCHERS, _person),
    "algorithm": (["Explain how the {} algorithm works.",
                   "Describe the steps of the {} algorithm.",
                   "How does the {} algorithm operate?"],
                  ALGORITHMS, _method),
    "element": (["What are the chemical properties of the element {}?",
                 "Describe the properties and uses of {}.",
                 "Summarize what is known about the element {}."],
                ELEMENTS, _element),
    "effect": (["What is the {} effect in physics?",
                "Explain the {} effect and its cause.",
                "Describe how the {} effect arises."],
               EFFECTS, _hyphen),
    "conjecture": (["Describe the {} conjecture in mathematics.",
                    "What does the {} conjecture state?",
                    "Explain the significance of the {} conjecture."],
                   CONJECTURES, _surname),
    "nobel": (["Who won the 1997 Nobel Prize in {}?",
               "Name the 1997 Nobel laureate in {}.",
               "Who received the 1997 Nobel Prize in {}?"],
              NOBEL_FIELDS, lambda i: _F_A[i % len(_F_A)]),
}


def build(n_fake_per, seed=0):
    rows = []
    for t, (pats, reals, fake_fn) in TEMPLATES.items():
        n = 0
        for r in reals:
            for pi, pat in enumerate(pats):
                rows.append({"id": f"lr-{t}-{n:04d}", "family": "longform_real",
                             "subtype": t, "template": t, "phrasing": pi,
                             "question": pat.format(r), "answers": []})
                n += 1
        seen = set()
        for i in range(n_fake_per * 6):
            if len(seen) >= n_fake_per:
                break
            pi = i % len(pats)
            q = pats[pi].format(fake_fn(i // len(pats)))
            if q in seen:
                continue
            seen.add(q)
            rows.append({"id": f"lf-{t}-{i:04d}", "family": "longform_fake",
                         "subtype": t, "template": t, "phrasing": pi,
                         "question": q, "answers": []})

    # papers get their own handling (three slots to fill)
    for i, (title, auth, yr) in enumerate(PAPERS):
        rows.append({"id": f"lr-paper-{i:04d}", "family": "longform_real",
                     "subtype": "paper", "template": "paper",
                     "question": f'Summarize the main finding of the {yr} paper '
                                 f'"{title}" by {auth}.', "answers": []})
    seen = set()
    for i in range(n_fake_per * 4):
        if len(seen) >= n_fake_per:
            break
        q = (f'Summarize the main finding of the {1960 + (i % 55)} paper '
             f'"The {_method(i).title()}" by {_person(i)}.')
        if q in seen:
            continue
        seen.add(q)
        rows.append({"id": f"lf-paper-{i:04d}", "family": "longform_fake",
                     "subtype": "paper", "template": "paper",
                     "question": q, "answers": []})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fake-per-template", type=int, default=400)
    ap.add_argument("--out", default="prompts_v3.jsonl")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = build(args.n_fake_per_template)
    with (OUT / args.out).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {len(rows)} prompts -> {OUT / args.out}")
    print(f"  {'template':<12} {'real':>6} {'fake':>6}")
    for t in list(TEMPLATES) + ["paper"]:
        r = sum(x["template"] == t and x["family"] == "longform_real" for x in rows)
        f = sum(x["template"] == t and x["family"] == "longform_fake" for x in rows)
        print(f"  {t:<12} {r:>6} {f:>6}")
    print(f"\ntotals: {Counter(r['family'] for r in rows)}")
    print("\nexamples:")
    for t in ("researcher", "algorithm", "conjecture"):
        for fam in ("longform_real", "longform_fake"):
            ex = next(x for x in rows if x["template"] == t and x["family"] == fam)
            print(f"  [{fam:<14}] {ex['question'][:88]}")


if __name__ == "__main__":
    main()
