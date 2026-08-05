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

import numpy as np
from numba import njit

# linguistic term indices
LOW, MED, HIGH = 0, 1, 2
ANY = -1  # "don't care" antecedent

N_TERMS = 3


def default_mf(n_in, sigma=0.30):
    """Gaussian terms centred at 0.0 / 0.5 / 1.0 on every input.

    ``sigma`` 0.30 makes adjacent terms cross at roughly half membership, so the
    rule base interpolates smoothly instead of switching. Inputs are all mapped to
    [0, 1] by their feature extractors, which is what lets one MF bank serve every
    input.
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


MF_KINDS = {"gaussian": _mf_gaussian, "triangular": _mf_triangular}


def mf_table(mf_c, mf_w, kind="gaussian"):
    """Compile a membership-function bank into a lookup table over [0, 1].

    Every fuzzy input here is already normalised to [0, 1] by its feature extractor,
    which means the membership functions can be tabulated once and evaluated in the
    hot path by a table lookup and a lerp — no exponentials, and no dependence on the
    functional form. Two things fall out of that:

    * it is much cheaper. The cost model measured a rule-base evaluation at ~580ns
      against ~870ns for the whole city scan it was deciding about, and the twelve
      exponentials were most of it;
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
# consequents below are the hand-written strategy — `tune.py` then fits them.
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
E_EXCESS = 0  # mean incident edge / nearest-neighbour distance  (HIGH = bad edges)
E_FAIL = 1  # how often this city has already come up empty     (HIGH = give up)
E_TURN = 2  # local turn sharpness at the city                  (HIGH = kinked)
E_PROGRESS = 3  # fraction of the search's work already spent    (HIGH = late)
E_N_IN = 4

# outputs, each in [0, 1] and rescaled to a real LK parameter by the caller
E_BREADTH = 0  # how far into the candidate list to back-track at the first level
E_DEEP = 1  # how many candidates to weigh at each deeper level
E_DEPTH = 2  # how deep to push the gain chain — the parameter that costs real time
E_ORSEG = 3  # longest Or-opt segment to try
E_N_OUT = 4

# Consequents are [breadth, deep_breadth, depth, or_seg].
#
# The ordering of those four matters, and it is not the one intuition suggests.
# Measured on the baseline, sweeping the first-level breadth from 2 to 32 barely
# moves the clock — the sequential gain criterion truncates most scans long before
# the cap bites. Sweeping the chain *depth* from 4 to 10 costs 2.6x. Depth is
# therefore the parameter worth being clever about, and these rules exist mainly to
# decide which cities deserve a deep chain: a city already sitting on
# near-minimal edges gets 2 or 3 levels, one carrying a long or sharply kinked edge
# gets the full chain.
EFFORT_RULES = [
    # a city whose edges are already about as short as they can be is not worth
    # a deep search; one carrying long edges is worth everything
    ({E_EXCESS: LOW}, [0.30, 0.20, 0.05, 0.30]),
    ({E_EXCESS: MED}, [0.60, 0.45, 0.35, 0.70]),
    ({E_EXCESS: HIGH}, [1.00, 0.90, 0.95, 1.00]),
    # repeated failure is the cheapest evidence there is that a city is done
    ({E_FAIL: HIGH}, [0.35, 0.20, 0.05, 0.20]),
    ({E_FAIL: LOW}, [0.70, 0.55, 0.45, 0.70]),
    # a sharp kink is a strong local signal that a move exists nearby
    ({E_TURN: HIGH}, [0.85, 0.70, 0.75, 0.90]),
    ({E_TURN: LOW}, [0.45, 0.30, 0.15, 0.40]),
    # late in the run the queue holds mostly stragglers; stay cheap unless the
    # city itself looks bad
    ({E_PROGRESS: HIGH}, [0.50, 0.35, 0.20, 0.50]),
    ({E_PROGRESS: LOW}, [0.65, 0.50, 0.40, 0.65]),
    # interactions
    ({E_EXCESS: HIGH, E_FAIL: LOW}, [1.00, 0.95, 1.00, 1.00]),
    ({E_EXCESS: HIGH, E_FAIL: HIGH}, [0.55, 0.50, 0.30, 0.60]),
    ({E_EXCESS: LOW, E_FAIL: HIGH}, [0.25, 0.10, 0.02, 0.10]),
    ({E_EXCESS: LOW, E_TURN: HIGH}, [0.65, 0.45, 0.40, 0.70]),
    ({E_EXCESS: HIGH, E_TURN: HIGH}, [1.00, 0.90, 1.00, 1.00]),
    ({E_EXCESS: MED, E_PROGRESS: HIGH}, [0.50, 0.35, 0.20, 0.50]),
    ({E_EXCESS: HIGH, E_PROGRESS: HIGH}, [0.90, 0.75, 0.85, 0.90]),
    ({E_TURN: LOW, E_FAIL: HIGH}, [0.25, 0.10, 0.02, 0.10]),
    ({E_TURN: LOW, E_PROGRESS: HIGH}, [0.35, 0.20, 0.08, 0.30]),
]

EFFORT_ANT, EFFORT_CONS = _pack(EFFORT_RULES, E_N_IN, E_N_OUT)
EFFORT_MF_C, EFFORT_MF_S = default_mf(E_N_IN)
EFFORT_TAB = mf_table(EFFORT_MF_C, EFFORT_MF_S)


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
CH_N_IN = 4

# One output: a continuation score. The chain deepens while it is above 0.5.
CHAIN_RULES = [
    # gain credit is the chain's own evidence that it is worth continuing
    ({CH_CREDIT: HIGH}, [0.90]),
    ({CH_CREDIT: MED}, [0.55]),
    ({CH_CREDIT: LOW}, [0.15]),
    # depth is the cost; the deeper we are, the better the news has to be
    ({CH_DEPTH: LOW}, [0.80]),
    ({CH_DEPTH: MED}, [0.50]),
    ({CH_DEPTH: HIGH}, [0.20]),
    # a move already banked is a reason to stop and take it
    ({CH_BANKED: HIGH}, [0.30]),
    ({CH_BANKED: LOW}, [0.60]),
    # still trading long edges for short ones? then there is more to find
    ({CH_TRADE: HIGH}, [0.85]),
    ({CH_TRADE: LOW}, [0.25]),
    # interactions
    ({CH_CREDIT: HIGH, CH_DEPTH: HIGH}, [0.60]),
    ({CH_CREDIT: LOW, CH_DEPTH: HIGH}, [0.02]),
    ({CH_CREDIT: LOW, CH_DEPTH: LOW}, [0.40]),
    ({CH_CREDIT: HIGH, CH_TRADE: HIGH}, [0.95]),
    ({CH_CREDIT: LOW, CH_TRADE: LOW}, [0.05]),
    ({CH_BANKED: HIGH, CH_DEPTH: HIGH}, [0.10]),
    ({CH_BANKED: LOW, CH_TRADE: HIGH}, [0.90]),
    ({CH_BANKED: HIGH, CH_CREDIT: LOW}, [0.08]),
]

CHAIN_ANT, CHAIN_CONS = _pack(CHAIN_RULES, CH_N_IN, 1)
CHAIN_MF_C, CHAIN_MF_S = default_mf(CH_N_IN)
CHAIN_TAB = mf_table(CHAIN_MF_C, CHAIN_MF_S)

# Passed to the shared LK chain by the baseline arm, which does not consult a rule
# base at all; never read, because the ``use_chain`` flag is false there.
NO_CHAIN_ANT = np.full((1, CH_N_IN), ANY, dtype=np.int8)
NO_CHAIN_CONS = np.full((1, 1), 0.5, dtype=np.float64)
NO_CHAIN_TAB = CHAIN_TAB


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
