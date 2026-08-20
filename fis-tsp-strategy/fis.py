"""A small Takagi-Sugeno fuzzy inference engine, and the three rule bases that drive
the TSP strategy engine.

The engine is deliberately plain: three linguistic terms per input (LOW / MED / HIGH),
a product t-norm, singleton consequents, weighted-average defuzzification. All of it is
nopython-jitted, and — because the cost model measured a rule-base evaluation at
comparable cost to the whole city scan it was deciding about — allocation-free, with the
membership bank compiled to a lookup table and every scratch buffer owned by the caller.
A consequence worth having: the membership functions' *shape* is then just data, so
their centres, widths and functional form are all things the optimiser can move.

Three rule bases live here.

``CONSTRUCT`` ranks candidate next-cities during tour construction — the "which city is
next" question. Its antecedents are the four things a human looks at when tracing a tour
by hand: how much worse than the nearest option the candidate is, whether it is about to
be stranded, what coming back for it later would cost, and whether it continues the
direction of travel.

``EFFORT`` allocates Lin-Kernighan search effort per city — the "which parameters"
question. Its consequents are the LK parameters themselves: how far to back-track at the
first level, how many candidates to weigh at deeper ones, how deep to push the gain
chain, and how much Or-opt to attempt. A conventional LK uses one setting everywhere.

``CHAIN`` decides, at every level of every gain chain, whether to deepen or cut, from the
chain's own gain trajectory rather than at a fixed depth. Depth is where this solver's
time actually goes, so this is the rule base that buys the most.

Every antecedent is a scale-free ratio, so one fitted rule base transfers between
instances whose coordinates differ by orders of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# linguistic term indices
LOW, MED, HIGH = 0, 1, 2
ANY = -1  # "don't care" antecedent

N_TERMS = 3


def default_mf(n_in, sigma=0.30):
    """Membership terms centred at 0.0 / 0.5 / 1.0 on every input, with width ``sigma``.

    0.30 makes adjacent terms overlap enough that the rule base interpolates smoothly
    instead of switching. Inputs are all mapped to [0, 1] by their feature extractors,
    which is what lets one bank serve every input.
    """
    c = np.tile(np.array([0.0, 0.5, 1.0]), (n_in, 1))
    s = np.full((n_in, N_TERMS), sigma, dtype=np.float64)
    return np.ascontiguousarray(c), np.ascontiguousarray(s)


MF_RES = 64  # lookup-table resolution per input; (MF_RES + 1) samples over [0, 1]


def _mf_gaussian(xs, c, w):
    z = (xs - c) / max(w, 1e-6)
    return np.exp(-0.5 * z * z)


def _mf_triangular(xs, c, w):
    return np.maximum(0.0, 1.0 - np.abs(xs - c) / max(w, 1e-6))


MF_KINDS = {"triangular": _mf_triangular, "gaussian": _mf_gaussian}

# Triangular is the default. Fitted against gaussian terms on the same instances, the
# same optimiser and the same budget, triangular came out ahead every time it was tried
# (validation frontier ratio 1.017 against 1.069 on the last full run), and it is cheaper
# to tabulate besides. Gaussian is kept selectable because the invariant tests check the
# lookup table against both closed forms.
DEFAULT_MF_KIND = "triangular"

# Which rule-base size the module-level defaults use. `small` is the reported default;
# `large` is selectable and compared in the findings.
DEFAULT_SCALE = "small"


def mf_table(mf_c, mf_w, kind=DEFAULT_MF_KIND):
    """Compile a membership-function bank into a lookup table over [0, 1].

    Every fuzzy input here is already normalised to [0, 1] by its feature extractor,
    which means the membership functions can be tabulated once and evaluated in the
    hot path by a table lookup and a lerp — no exponentials, and no dependence on the
    functional form. Two things fall out of that:

    * it is much cheaper. The cost model measured a rule-base evaluation at ~580ns
      against ~870ns for the whole city scan it was deciding about, and the membership
      functions were most of it;
    * the *shape* of the membership functions stops being hard-coded. Centres, widths
      and the functional form all become parameters the optimiser can move, because
      the hot path never knows which it got.
    """
    fn = MF_KINDS[kind]
    n_in, n_terms = mf_c.shape
    xs = np.linspace(0.0, 1.0, MF_RES + 1)
    tab = np.empty((n_in, n_terms, MF_RES + 1), dtype=np.float64)
    for i in range(n_in):
        for t in range(n_terms):
            tab[i, t] = fn(xs, mf_c[i, t], mf_w[i, t])
    return np.ascontiguousarray(tab)


@njit(cache=True, inline="always")
def _memberships(x, tab, mu):
    """Fill ``mu[i, t]`` with input i's membership in term t, by table lerp.

    Every rule that constrains input i needs the same terms, so they are computed
    once per evaluation rather than once per rule — 12 lookups instead of up to 72.
    """
    n_in = mu.shape[0]
    n_terms = mu.shape[1]
    for i in range(n_in):
        xi = x[i]
        if xi < 0.0:
            xi = 0.0
        elif xi > 1.0:
            xi = 1.0
        f = xi * MF_RES
        j = int(f)
        if j >= MF_RES:
            j = MF_RES - 1
        a = f - j
        for t in range(n_terms):
            v0 = tab[i, t, j]
            mu[i, t] = v0 + a * (tab[i, t, j + 1] - v0)


@njit(cache=True)
def fis_eval(x, mu, tab, ant, cons, out):
    """Evaluate a TSK-0 rule base. ``out`` is filled with the defuzzified outputs.

    ``ant[r, i]`` is the term index required of input i by rule r, or ANY (-1).
    ``cons[r, j]`` is rule r's singleton for output j. Firing strength is the
    product of the memberships the rule actually constrains.

    ``mu`` is a caller-owned (n_in, N_TERMS) scratch buffer. It is an argument rather
    than a local because a local would be a heap allocation on every call, and at
    this call frequency that allocation costs about as much as the exponentials it
    was introduced to save — measured, not assumed.
    """
    n_rules = ant.shape[0]
    n_in = ant.shape[1]
    n_out = cons.shape[1]
    _memberships(x, tab, mu)
    for j in range(n_out):
        out[j] = 0.0
    den = 0.0
    for r in range(n_rules):
        w = 1.0
        for i in range(n_in):
            a = ant[r, i]
            if a >= 0:
                w *= mu[i, a]
                if w < 1e-12:
                    break
        if w < 1e-12:
            continue
        den += w
        for j in range(n_out):
            out[j] += w * cons[r, j]
    if den <= 1e-12:  # no rule fired: sit in the middle of every output
        for j in range(n_out):
            out[j] = 0.5
        return
    for j in range(n_out):
        out[j] /= den


@njit(cache=True)
def fis_eval1(x, mu, tab, ant, cons):
    """Single-output fast path — used by the construction ranker and the chain
    cut-off, the two hottest calls in the system. ``mu`` is caller-owned scratch."""
    n_rules = ant.shape[0]
    n_in = ant.shape[1]
    _memberships(x, tab, mu)
    num = 0.0
    den = 0.0
    for r in range(n_rules):
        w = 1.0
        for i in range(n_in):
            a = ant[r, i]
            if a >= 0:
                w *= mu[i, a]
                if w < 1e-12:
                    break
        if w < 1e-12:
            continue
        den += w
        num += w * cons[r, 0]
    if den <= 1e-12:
        return 0.5
    return num / den


def _pack(rules, n_in, n_out):
    """(antecedent int8 array, consequent float64 array) from a rule list."""
    ant = np.full((len(rules), n_in), ANY, dtype=np.int8)
    cons = np.zeros((len(rules), n_out), dtype=np.float64)
    for r, (a, c) in enumerate(rules):
        for i, term in a.items():
            ant[r, i] = term
        cons[r] = c
    return np.ascontiguousarray(ant), np.ascontiguousarray(cons)


# ---------------------------------------------------------------------------
# CONSTRUCT: rank the candidate next-cities
# ---------------------------------------------------------------------------
# inputs
#
# C_EXCESS is deliberately *relative to the best candidate available right now*
# rather than to the city's neighbourhood scale. Measured against a neighbourhood
# scale, the nearest and second-nearest candidate both score "very near" and both
# fire the same rules, so the other three cues end up deciding every step and the
# construction wanders — measurably worse than plain nearest-neighbour. Measured
# as excess over the best option, the nearest candidate sits exactly at 0.0 and a
# candidate 30% further sits at 0.3, which is the resolution the ranking needs.
C_EXCESS = 0  # (d - d_best) / d_best, clipped         (LOW = this is the NN choice)
C_STRAND = 1  # fraction of cand's own neighbours already visited (HIGH = stranding)
C_RETREAT = 2  # cost of coming back for it later      (HIGH = expensive to defer)
C_STRAIGHT = 3  # turn alignment with the current heading        (HIGH = straight on)
C_N_IN = 4

# Each rule is ({input: term}, [consequent]) with the consequent a desirability in
# [0, 1]; the city with the highest defuzzified score is taken next. The starting
# consequents below are the hand-written strategy — `tune_opt.py` then fits them.
#
# The shape of the rule base matters: it has to reduce to nearest-neighbour when
# the other cues are neutral, and only override that when deferring a city is
# demonstrably expensive. So C_EXCESS carries the widest consequent spread, and the
# cues that argue for leaving the greedy choice have to agree with each other to
# outvote it.
CONSTRUCT_RULES = [
    # the nearest-neighbour instinct, and its opposite
    ({C_EXCESS: LOW}, [0.90]),
    ({C_EXCESS: MED}, [0.35]),
    ({C_EXCESS: HIGH}, [0.05]),
    # rescue cities that are about to lose their last unvisited neighbour
    ({C_STRAND: HIGH}, [0.70]),
    ({C_STRAND: LOW}, [0.45]),
    # a city whose own neighbours are all gone is expensive to come back for
    ({C_RETREAT: HIGH}, [0.65]),
    ({C_RETREAT: LOW}, [0.45]),
    # a hand-drawn tour goes straight until it has to turn
    ({C_STRAIGHT: HIGH}, [0.60]),
    ({C_STRAIGHT: LOW}, [0.40]),
    # interactions: the cues that argue against the greedy choice only win when
    # they agree, and the ones that agree with it reinforce
    ({C_EXCESS: LOW, C_STRAND: HIGH}, [0.95]),
    ({C_EXCESS: LOW, C_STRAIGHT: HIGH}, [0.95]),
    ({C_EXCESS: MED, C_STRAND: HIGH, C_RETREAT: HIGH}, [0.80]),
    ({C_EXCESS: HIGH, C_STRAND: HIGH, C_RETREAT: HIGH}, [0.55]),
    ({C_EXCESS: MED, C_STRAND: LOW}, [0.20]),
    ({C_EXCESS: HIGH, C_STRAND: LOW}, [0.02]),
    ({C_EXCESS: MED, C_STRAIGHT: HIGH}, [0.45]),
    ({C_EXCESS: MED, C_RETREAT: LOW}, [0.20]),
    ({C_EXCESS: HIGH, C_RETREAT: LOW}, [0.02]),
    ({C_STRAND: HIGH, C_RETREAT: HIGH}, [0.75]),
    ({C_STRAND: LOW, C_RETREAT: LOW}, [0.35]),
]

CONSTRUCT_ANT, CONSTRUCT_CONS = _pack(CONSTRUCT_RULES, C_N_IN, 1)
CONSTRUCT_MF_C, CONSTRUCT_MF_S = default_mf(C_N_IN)
CONSTRUCT_TAB = mf_table(CONSTRUCT_MF_C, CONSTRUCT_MF_S)


# ---------------------------------------------------------------------------
# EFFORT: allocate LK search effort to the city about to be searched
# ---------------------------------------------------------------------------
# inputs
# Inputs, chosen by measured predictive power rather than by argument. `experiments/features_probe.py`
# scores each candidate on how well it predicts whether a city search will yield an improving
# move (AUC) and, among those that do, how large the gain is. `feature_registry.py` is the
# master record of what was tried, what was rejected, and which scale each survivor is in.
#
# The first five clear AUC 0.74 and are in every scale. The last three sit in the 0.55-0.70
# band: dropped from `small`, kept in `large` to test whether interaction rules and a four
# times larger training pool can extract something the single-antecedent base could not.
#
# Index order is load-bearing: the feature extractor in `fis_lk.py` fills x[0:5]
# unconditionally and x[5:8] only when the rule base is wide enough to reference them, so the
# small scale does not pay for what it does not read.
E_PROBEF = 0  # candidates passing the depth-1 gain test, / 2k        (AUC 0.858)
E_RANK = 1  # nearer neighbours the worse tour edge ignores, / k      (AUC 0.833)
E_PROBE = 2  # best depth-1 gain available, over the broken edge      (AUC 0.795)
E_EXCESS = 3  # mean incident edge / nearest-neighbour distance       (AUC 0.759)
E_ASYM = 4  # |d_succ - d_pred| / (d_succ + d_pred)                   (AUC 0.741)
E_TURN = (
    5  # local turn sharpness                                   (AUC 0.691, large only)
)
E_PEAK = (
    6  # nearest-neighbour / mean candidate distance            (AUC 0.589, large only)
)
E_PROGRESS = (
    7  # fraction of the run's work already spent           (AUC 0.579, large only)
)
E_N_IN_SMALL = 5
E_N_IN_LARGE = 8

# outputs, each in [0, 1] and rescaled to a real LK parameter by the caller
E_BREADTH = 0  # how far into the candidate list to back-track at the first level
E_DEEP = 1  # how many candidates to weigh at each deeper level
E_DEPTH = 2  # how deep to push the gain chain — the parameter that costs real time
E_ORSEG = 3  # longest Or-opt segment to try
E_N_OUT = 4

# Consequents are [breadth, deep_breadth, depth, or_seg]. Depth carries the widest spread
# because depth is the parameter that costs time; breadth barely moves the clock, so it stays
# generous throughout.
#
# `small`: three rules per input, no interactions. A purely additive base cannot express "long
# edge but the probe sees nothing", but it has 60 consequents and every rule reads as a
# sentence.
_EFFORT_CORE = [
    # the probe is the strongest signal there is: if one level of search can already see gain
    # here, commit; if it can see none, there is little point going deeper
    ({E_PROBEF: LOW}, [0.30, 0.20, 0.05, 0.25]),
    ({E_PROBEF: MED}, [0.65, 0.50, 0.45, 0.65]),
    ({E_PROBEF: HIGH}, [1.00, 0.90, 0.95, 1.00]),
    # how many strictly better neighbours the current edge is ignoring
    ({E_RANK: LOW}, [0.25, 0.15, 0.05, 0.25]),
    ({E_RANK: MED}, [0.65, 0.50, 0.45, 0.65]),
    ({E_RANK: HIGH}, [0.95, 0.85, 0.90, 0.95]),
    # the size of the best gain a single level can see, not just whether one exists
    ({E_PROBE: LOW}, [0.35, 0.25, 0.10, 0.35]),
    ({E_PROBE: MED}, [0.65, 0.50, 0.50, 0.65]),
    ({E_PROBE: HIGH}, [0.95, 0.85, 0.95, 0.95]),
    # a city whose edges are already near-minimal is not worth a deep chain
    ({E_EXCESS: LOW}, [0.30, 0.20, 0.05, 0.30]),
    ({E_EXCESS: MED}, [0.60, 0.45, 0.35, 0.65]),
    ({E_EXCESS: HIGH}, [1.00, 0.90, 0.90, 1.00]),
    # one long edge and one short is a stronger signal than two medium ones
    ({E_ASYM: LOW}, [0.45, 0.30, 0.20, 0.45]),
    ({E_ASYM: MED}, [0.65, 0.50, 0.50, 0.65]),
    ({E_ASYM: HIGH}, [0.90, 0.80, 0.85, 0.90]),
]

# `large`: the middling-AUC inputs, plus interactions. The interactions are chosen where the
# screening gives a reason to expect one, not exhaustively — an exhaustive set would be 3^2
# rules per pair and put the parameter count back where the overfitting was.
_EFFORT_EXTRA = [
    # turn sharpness predicts *whether* a city pays (AUC 0.691) but among paying cities
    # sharper turns pay *less* (rho -0.116). A single monotone rule cannot serve both; these
    # two say "a kink is worth looking at, but not worth a deep chain on its own"
    ({E_TURN: HIGH}, [0.85, 0.65, 0.45, 0.80]),
    ({E_TURN: LOW}, [0.45, 0.35, 0.30, 0.45]),
    # peakedness is weak on whether (0.589) and second-best on how much (rho 0.231), so it is
    # given influence over depth, which is what pays off on the large gains
    ({E_PEAK: LOW}, [0.40, 0.35, 0.55, 0.55]),
    ({E_PEAK: HIGH}, [0.70, 0.55, 0.45, 0.65]),
    # late in the run the queue holds mostly stragglers
    ({E_PROGRESS: HIGH}, [0.50, 0.40, 0.30, 0.50]),
    ({E_PROGRESS: LOW}, [0.65, 0.50, 0.50, 0.65]),
    # interactions: the probe agreeing or disagreeing with the static geometry is the case a
    # purely additive base cannot express, and it is the reason to try `large` at all
    ({E_PROBEF: HIGH, E_EXCESS: HIGH}, [1.00, 0.95, 1.00, 1.00]),
    ({E_PROBEF: LOW, E_EXCESS: HIGH}, [0.40, 0.30, 0.15, 0.40]),
    ({E_PROBEF: HIGH, E_EXCESS: LOW}, [0.85, 0.70, 0.70, 0.85]),
    ({E_PROBEF: LOW, E_RANK: LOW}, [0.15, 0.10, 0.02, 0.12]),
    ({E_PROBE: HIGH, E_PEAK: LOW}, [0.90, 0.80, 1.00, 0.95]),
    ({E_TURN: HIGH, E_PROBEF: LOW}, [0.55, 0.40, 0.15, 0.50]),
    ({E_RANK: HIGH, E_PROGRESS: HIGH}, [0.85, 0.70, 0.75, 0.85]),
    ({E_ASYM: HIGH, E_PROBEF: HIGH}, [1.00, 0.90, 0.95, 1.00]),
    ({E_EXCESS: LOW, E_PROBE: LOW}, [0.20, 0.12, 0.02, 0.18]),
]

EFFORT_RULES_BY_SCALE = {
    "small": (_EFFORT_CORE, E_N_IN_SMALL),
    "large": (_EFFORT_CORE + _EFFORT_EXTRA, E_N_IN_LARGE),
}

_E_NAMES = (
    "probe_frac",
    "rank",
    "probe",
    "excess",
    "edge_asym",
    "turn",
    "peak",
    "progress",
)


def effort_inputs(scale):
    """The antecedent names this scale's EFFORT base reads, in index order."""
    return list(_E_NAMES[: EFFORT_RULES_BY_SCALE[scale][1]])


def effort_base(scale=DEFAULT_SCALE):
    """(ant, cons, mf_c, mf_w, tab) for one EFFORT scale."""
    rules, n_in = EFFORT_RULES_BY_SCALE[scale]
    ant, cons = _pack(rules, n_in, E_N_OUT)
    c, w = default_mf(n_in)
    return ant, cons, c, w, mf_table(c, w)


# ---------------------------------------------------------------------------
# CHAIN: keep deepening this gain chain, or cut it here?
# ---------------------------------------------------------------------------
# A conventional LK deepens every chain to the same fixed cut-off. That is the one
# decision in the whole solver where a fixed rule is most obviously wrong: by the
# time a chain is three levels down it has *told* you how it is going — how much
# gain credit it has banked, whether it is still finding long edges to trade away,
# and whether it has already found a move worth taking. A rule base reading that
# trajectory can cut a hopeless chain at level 2 and let a promising one run to 15,
# which is exactly where the time in this solver is spent.
#
# This is the ranking question the LK step actually asks — which city to move to
# next, and whether to keep going — answered with a fuzzy system instead of a
# constant.
CH_CREDIT = 0  # gain credit carried into the next level, over the first broken edge
CH_DEPTH = 1  # how deep the chain already is, as a fraction of the cap
CH_BANKED = 2  # best closing gain found so far, over the first broken edge
CH_TRADE = 3  # next step's "break long, add short" margin, same scale
# How much array the reversal at this level actually moved, as a fraction of the most it
# could (n/2). This is the one input taken directly from the cost model rather than from
# the search's own logic: reversal element traffic is a real and separately-priced part of
# runtime (~5ns per element against ~120ns per chain level), and a chain working in a
# region where every level shuffles half the tour is expensive in a way that no gain
# number reveals. It costs nothing to read, because `reverse` already returns the count.
CH_REVCOST = 4  # (HIGH = each level here is shifting a lot of array)
CH_N_IN = 5

# One output: a continuation score. The chain deepens while it is above 0.5.
# Three rules per input, no interactions, for the same reason as EFFORT: the previous base
# carried 24 rules of which ten were interactions, and they were where the overfitting lived.
CHAIN_RULES = [
    # gain credit is the chain's own evidence that continuing is worth it
    ({CH_CREDIT: LOW}, [0.12]),
    ({CH_CREDIT: MED}, [0.55]),
    ({CH_CREDIT: HIGH}, [0.92]),
    # depth is the cost, so the deeper we already are the better the news has to be
    ({CH_DEPTH: LOW}, [0.82]),
    ({CH_DEPTH: MED}, [0.50]),
    ({CH_DEPTH: HIGH}, [0.18]),
    # a move already banked is a reason to stop and take it
    ({CH_BANKED: LOW}, [0.62]),
    ({CH_BANKED: MED}, [0.45]),
    ({CH_BANKED: HIGH}, [0.28]),
    # still trading long edges for short ones? then there is more to find
    ({CH_TRADE: LOW}, [0.22]),
    ({CH_TRADE: MED}, [0.55]),
    ({CH_TRADE: HIGH}, [0.88]),
    # reversal traffic is real time that the gain trajectory says nothing about
    ({CH_REVCOST: LOW}, [0.72]),
    ({CH_REVCOST: MED}, [0.50]),
    ({CH_REVCOST: HIGH}, [0.30]),
]

_CHAIN_EXTRA = [
    # `large` adds interactions only; the chain's five inputs are all free to compute, so
    # there is nothing to add on that side. These say what the additive base cannot: that the
    # same gain credit means different things at different depths and reversal costs.
    ({CH_CREDIT: HIGH, CH_DEPTH: HIGH}, [0.55]),
    ({CH_CREDIT: LOW, CH_DEPTH: HIGH}, [0.02]),
    ({CH_CREDIT: LOW, CH_DEPTH: LOW}, [0.38]),
    ({CH_CREDIT: HIGH, CH_TRADE: HIGH}, [0.95]),
    ({CH_CREDIT: LOW, CH_TRADE: LOW}, [0.04]),
    ({CH_BANKED: HIGH, CH_DEPTH: HIGH}, [0.08]),
    ({CH_BANKED: LOW, CH_TRADE: HIGH}, [0.90]),
    ({CH_REVCOST: HIGH, CH_CREDIT: LOW}, [0.04]),
    ({CH_REVCOST: HIGH, CH_CREDIT: HIGH}, [0.68]),
    ({CH_REVCOST: LOW, CH_CREDIT: MED}, [0.66]),
]

CHAIN_RULES_BY_SCALE = {
    "small": CHAIN_RULES,
    "large": CHAIN_RULES + _CHAIN_EXTRA,
}

_CH_NAMES = ("credit", "depth", "banked", "trade", "revcost")


def chain_inputs(scale):
    """The antecedent names this scale's CHAIN base reads, in index order."""
    return list(_CH_NAMES[:CH_N_IN])


def chain_base(scale=DEFAULT_SCALE):
    """(ant, cons, mf_c, mf_w, tab) for one CHAIN scale."""
    ant, cons = _pack(CHAIN_RULES_BY_SCALE[scale], CH_N_IN, 1)
    c, w = default_mf(CH_N_IN)
    return ant, cons, c, w, mf_table(c, w)


# Module-level defaults, built at the default scale. Everything that does not care about
# scale — the benchmark's hand-written arms, the invariant tests — reads these; anything that
# does calls effort_base()/chain_base() with the scale it wants.
EFFORT_ANT, EFFORT_CONS, EFFORT_MF_C, EFFORT_MF_S, EFFORT_TAB = effort_base(
    DEFAULT_SCALE
)
CHAIN_ANT, CHAIN_CONS, CHAIN_MF_C, CHAIN_MF_S, CHAIN_TAB = chain_base(DEFAULT_SCALE)

# Passed to the shared LK chain by the baseline arm, which does not consult a rule
# base at all; never read, because the ``use_chain`` flag is false there.
NO_CHAIN_ANT = np.full((1, CH_N_IN), ANY, dtype=np.int8)
NO_CHAIN_CONS = np.full((1, 1), 0.5, dtype=np.float64)
NO_CHAIN_TAB = CHAIN_TAB


@dataclass(frozen=True)
class Tuned:
    """A fitted rule base, with the antecedent arrays its consequents are indexed against.

    The pairing is the whole point of the type. A fitted consequent table is meaningless on
    its own: rule *i* of the ``small`` EFFORT base and rule *i* of the ``large`` one read
    different inputs, so loading one scale's consequents against the other's antecedents
    silently produces a rule base that runs, terminates, and means nothing. The scale is
    recorded in the ``.npz`` at fitting time and the antecedents are rebuilt from it here,
    so the two cannot be paired up wrongly by a caller.
    """

    scale: str
    effort_ant: np.ndarray
    effort_cons: np.ndarray
    effort_tab: np.ndarray
    chain_ant: np.ndarray
    chain_cons: np.ndarray
    chain_tab: np.ndarray


def load_tuned(path):
    """Read a ``tuned_<scale>.npz`` written by ``tune_opt.py``.

    ``CONSTRUCT`` is deliberately absent: fuzzy construction is a measured failure (FINDINGS
    §8) and appears in no reported arm, so it is no longer fitted and the ranker keeps its
    hand-written rules.
    """
    z = np.load(path)
    scale = str(z["scale"]) if "scale" in z else DEFAULT_SCALE
    e_ant = effort_base(scale)[0]
    h_ant = chain_base(scale)[0]
    return Tuned(
        scale=scale,
        effort_ant=e_ant,
        effort_cons=np.ascontiguousarray(z["effort_cons"]),
        effort_tab=np.ascontiguousarray(z["effort_tab"]),
        chain_ant=h_ant,
        chain_cons=np.ascontiguousarray(z["chain_cons"]),
        chain_tab=np.ascontiguousarray(z["chain_tab"]),
    )


def hand_written(scale=DEFAULT_SCALE):
    """The same record, holding the hand-written rules — the control for any fitted one."""
    e_ant, e_cons, _, _, e_tab = effort_base(scale)
    h_ant, h_cons, _, _, h_tab = chain_base(scale)
    return Tuned(scale, e_ant, e_cons, e_tab, h_ant, h_cons, h_tab)


# ---------------------------------------------------------------------------
# packing the tunable parameters into one flat vector
# ---------------------------------------------------------------------------
def theta_from_rules(construct_cons, effort_cons, chain_cons):
    """Flatten all three consequent tables into the vector the tuner searches."""
    return np.concatenate(
        [construct_cons.ravel(), effort_cons.ravel(), chain_cons.ravel()]
    )


N_CONSTRUCT = CONSTRUCT_CONS.size
N_EFFORT = EFFORT_CONS.size
N_CHAIN = CHAIN_CONS.size


def rules_from_theta(theta):
    """Inverse of :func:`theta_from_rules`, clipped back into [0, 1]."""
    a = N_CONSTRUCT
    b = a + N_EFFORT
    c = np.ascontiguousarray(theta[:a].reshape(CONSTRUCT_CONS.shape).clip(0.0, 1.0))
    e = np.ascontiguousarray(theta[a:b].reshape(EFFORT_CONS.shape).clip(0.0, 1.0))
    h = np.ascontiguousarray(theta[b:].reshape(CHAIN_CONS.shape).clip(0.0, 1.0))
    return c, e, h


DEFAULT_THETA = theta_from_rules(CONSTRUCT_CONS, EFFORT_CONS, CHAIN_CONS)
