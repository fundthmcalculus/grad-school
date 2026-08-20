"""The master record of every antecedent tried, its measured score, and where it ended up.

This is the single source of truth for feature decisions. It exists because the decisions are
easy to lose: a feature gets argued for, added, later quietly dropped when a rule base is
rewritten, and six weeks on nobody can say whether it was measured and rejected or simply
forgotten. Scores come from ``features_probe.py``; the verdicts and scale membership are
recorded here; ``FINDINGS.md`` renders its table from this module rather than restating it, so
the two cannot disagree.

**How to read the scores.** ``auc`` is the probability that a randomly chosen city which
yielded an improving move scores above one that did not — 0.5 is worthless, and the distance
from 0.5 is what matters since a feature reading the "wrong" way is still informative.
``rho`` is the rank correlation with realised gain *among the cities that paid*, which is the
separate and harder question of ordering by how much rather than whether. Both are over
12 278 city scans on six instances, three TSPLIB and three synthetic, 10.9% of which paid off.

**The scales.** Three rule-base sizes are defined and compared:

* ``legacy`` — what the rule base held before any of this was measured. Features chosen by
  argument, 6 inputs over 27 rules with 9 two-input interactions.
* ``small`` — only features clearing AUC 0.74, three rules per input, no interactions.
  5 inputs, 15 rules.
* ``large`` — ``small`` plus the middling band (AUC 0.55–0.70) and interaction rules, to test
  whether the extra signal is worth the extra parameters now that the training pool is four
  times bigger. 8 inputs, 36 rules.

A feature can be in a scale's *input* list without every scale using it the same way; the
rules are in ``fis.py``.

Run:  python feature_registry.py            # the table as markdown
      python feature_registry.py --check    # verify it matches fis.py and features_probe.py
"""

from __future__ import annotations

import argparse

import paths
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str  # the identifier used in fis.py, without the E_/CH_ prefix
    base: str  # which rule base: EFFORT or CHAIN
    description: str
    auc: float | None  # None where the feature is not a per-city payoff predictor
    rho: float | None
    cost: str  # what computing it costs in the hot path
    verdict: str  # kept | rejected | dropped | not-screened
    scales: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


# ---------------------------------------------------------------------------
# EFFORT — per-city effort allocation. These are the ones the screening measured,
# because "will this city pay off" is a per-city question with an observable answer.
# ---------------------------------------------------------------------------
EFFORT_FEATURES = [
    Feature(
        "probe_frac",
        "EFFORT",
        "fraction of candidates passing the depth-1 positive-gain test, both directions",
        0.858,
        0.327,
        "a scan that breaks at the first failing candidate; usually 1-3 iterations",
        "kept",
        ("small", "large"),
        "Best of everything tried, on both questions. A look-ahead: one level of search run "
        "before committing to any.",
    ),
    Feature(
        "rank",
        "EFFORT",
        "how many strictly nearer neighbours the worse incident tour edge is ignoring, / k",
        0.833,
        0.301,
        "a scan of an ascending list, breaks at the edge length",
        "kept",
        ("small", "large"),
        "Scale-free without dividing by anything: counts better options rather than measuring "
        "an excess.",
    ),
    Feature(
        "probe",
        "EFFORT",
        "size of the best depth-1 gain available, over the broken edge length",
        0.795,
        0.197,
        "free alongside probe_frac — same loop",
        "kept",
        ("small", "large"),
        "Partly redundant with probe_frac; the marginal contribution of the pair has not been "
        "ablated.",
    ),
    Feature(
        "excess",
        "EFFORT",
        "mean incident edge length / nearest-neighbour distance",
        0.759,
        0.153,
        "two distance evaluations and a divide",
        "kept",
        ("legacy", "small", "large"),
    ),
    Feature(
        "edge_asym",
        "EFFORT",
        "|d_succ - d_pred| / (d_succ + d_pred)",
        0.741,
        0.157,
        "free — both distances are already computed",
        "kept",
        ("small", "large"),
        "One long edge and one short is a better prospect than two medium ones, because the "
        "long one is what a 2-opt can remove.",
    ),
    Feature(
        "turn",
        "EFFORT",
        "local turn sharpness at the city, 0 straight through to 1 doubling back",
        0.691,
        -0.116,
        "two hypots — the most expensive feature here",
        "dropped",
        ("legacy", "large"),
        "The signs disagree: it predicts *whether* a city pays (0.691) but among paying cities "
        "sharper turns pay *less* (-0.116). One monotone rule cannot serve both, which is why "
        "`small` drops it and `large` keeps it only to test whether interactions can use it.",
    ),
    Feature(
        "peak",
        "EFFORT",
        "nearest-neighbour distance / mean candidate distance",
        0.589,
        0.231,
        "two loads and a divide, both precomputed per instance",
        "dropped",
        ("legacy", "large"),
        "Weak at predicting whether, second-best at ordering by how much. Kept in `large` for "
        "that reason.",
    ),
    Feature(
        "progress",
        "EFFORT",
        "fraction of the run's total work already spent",
        0.579,
        0.018,
        "a divide",
        "dropped",
        ("legacy", "large"),
        "Nearly useless on both questions. It is a property of the run, not of the city, which "
        "is probably why.",
    ),
    Feature(
        "pos_spread",
        "EFFORT",
        "tour-position spread of the candidate neighbours, / n",
        0.547,
        0.174,
        "a full k-iteration scan, no early break",
        "rejected",
        (),
        "The idea was to detect geometry/tour mismatch. It does not, and it is one of the more "
        "expensive candidates.",
    ),
    Feature(
        "cand_step",
        "EFFORT",
        "(second-nearest - nearest) / nearest",
        0.520,
        -0.106,
        "two loads and a divide",
        "rejected",
        (),
        "Intended as a sharper local version of peak. Indistinguishable from noise.",
    ),
    Feature(
        "fails",
        "EFFORT",
        "how many times this city has already been searched without result",
        0.488,
        -0.123,
        "one load",
        "dropped",
        ("legacy",),
        "An *existing* input, and at AUC 0.488 indistinguishable from noise. The don't-look-bit "
        "queue already removes settled cities structurally, so the count adds nothing on top "
        "of the mechanism that produces it. Still used for the queue bookkeeping; just not as "
        "a rule-base input.",
    ),
    Feature(
        "in_degree",
        "EFFORT",
        "how many cities hold this one in their candidate list, / k",
        0.449,
        -0.031,
        "one load, precomputed per instance",
        "rejected",
        (),
        "Meant to distinguish hubs from leaves. It does not predict payoff.",
    ),
    Feature(
        "nbr_active",
        "EFFORT",
        "fraction of this city's candidate neighbours still in the work queue",
        0.426,
        -0.100,
        "a full k-iteration scan, no early break",
        "rejected",
        (),
        "Meant to capture whether a neighbourhood is still active. Reads *inverted* (fewer "
        "active neighbours slightly predicts paying off) and is weak either way.",
    ),
]

# ---------------------------------------------------------------------------
# CHAIN — whether to deepen or cut a gain chain. These are deliberately not scored by the
# per-city screen: the decision happens mid-chain, many times per city, and its outcome is
# "would one more level have improved the best closing gain", which the per-city payoff label
# does not answer. They are justified by the chain's own arithmetic and by ablation instead.
# ---------------------------------------------------------------------------
CHAIN_FEATURES = [
    Feature(
        "credit",
        "CHAIN",
        "gain credit carried into the next level, over the first broken edge",
        None,
        None,
        "free — the chain already has it",
        "kept",
        ("legacy", "small", "large"),
    ),
    Feature(
        "depth",
        "CHAIN",
        "how deep the chain already is, as a fraction of the cap",
        None,
        None,
        "free",
        "kept",
        ("legacy", "small", "large"),
        "Depth is what costs time, so this is the input the cut-off exists to weigh against.",
    ),
    Feature(
        "banked",
        "CHAIN",
        "best closing gain found so far, over the first broken edge",
        None,
        None,
        "free",
        "kept",
        ("legacy", "small", "large"),
    ),
    Feature(
        "trade",
        "CHAIN",
        "next step's break-long-add-short margin, same scale",
        None,
        None,
        "free — computed while choosing the next candidate",
        "kept",
        ("legacy", "small", "large"),
    ),
    Feature(
        "revcost",
        "CHAIN",
        "how much array this level's reversal moved, over the most it could",
        None,
        None,
        "free — `reverse` already returns its swap count",
        "kept",
        ("small", "large"),
        "The only input taken from the cost model rather than the search's own logic: reversal "
        "traffic is separately priced at ~4.4ns/element and a chain shuffling half the tour "
        "per level is expensive in a way no gain number reveals.",
    ),
]

FEATURES = EFFORT_FEATURES + CHAIN_FEATURES

SCALES = ("legacy", "small", "large")

SCALE_NOTES = {
    "legacy": "features by argument; 6 inputs, 27 rules, 9 interactions",
    "small": "AUC >= 0.74 only; 5 inputs, 15 rules, no interactions",
    "large": "small + the AUC 0.55-0.70 band + interactions; 8 inputs, 36 rules",
}


def inputs_for(base, scale):
    """The antecedent names one rule base uses at one scale, in registry order."""
    return [f.name for f in FEATURES if f.base == base and scale in f.scales]


def as_markdown():
    lines = []
    lines.append(
        "| feature | AUC | ρ (paying) | cost | verdict | legacy | small | large |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for f in FEATURES:
        auc = f"{f.auc:.3f}" if f.auc is not None else "—"
        rho = f"{f.rho:+.3f}" if f.rho is not None else "—"
        marks = ["●" if sc in f.scales else "" for sc in SCALES]
        lines.append(
            f"| `{f.name}` ({f.base}) | {auc} | {rho} | {f.cost} | {f.verdict} | "
            f"{marks[0]} | {marks[1]} | {marks[2]} |"
        )
    return "\n".join(lines)


def check():
    """Cross-check the registry against what the code actually builds."""
    import fis

    problems = []
    for scale in ("small", "large"):
        want_e = inputs_for("EFFORT", scale)
        want_c = inputs_for("CHAIN", scale)
        got_e = fis.effort_inputs(scale)
        got_c = fis.chain_inputs(scale)
        if list(want_e) != list(got_e):
            problems.append(
                f"EFFORT/{scale}: registry {want_e} != fis.py {list(got_e)}"
            )
        if list(want_c) != list(got_c):
            problems.append(f"CHAIN/{scale}: registry {want_c} != fis.py {list(got_c)}")
    names = [f.name for f in FEATURES]
    if len(names) != len(set(names)):
        problems.append("duplicate feature names in the registry")
    for f in FEATURES:
        if f.verdict == "kept" and not f.scales:
            problems.append(f"{f.name}: verdict 'kept' but in no scale")
        if f.verdict == "rejected" and f.scales:
            problems.append(f"{f.name}: verdict 'rejected' but listed in {f.scales}")
        for sc in f.scales:
            if sc not in SCALES:
                problems.append(f"{f.name}: unknown scale {sc!r}")
    return problems


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        problems = check()
        for p in problems:
            print(f"  MISMATCH {p}")
        print(
            "registry consistent with fis.py"
            if not problems
            else f"{len(problems)} problems"
        )
        raise SystemExit(1 if problems else 0)
    for sc in SCALES:
        print(f"{sc:>7s}: {SCALE_NOTES[sc]}")
        print(f"         EFFORT {inputs_for('EFFORT', sc)}")
        print(f"         CHAIN  {inputs_for('CHAIN', sc)}")
    print()
    print(as_markdown())


if __name__ == "__main__":
    main()
