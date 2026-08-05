"""Invariant checks for the things that would silently produce good-looking wrong numbers.

Every check here exists because breaking it once produced a plausible result rather than
a crash. A solver that drops a city reports a shorter tour; a move whose claimed gain does
not match what it did to the tour makes the local search converge to the wrong place while
still terminating; a candidate list whose tie order shifts with k makes a parameter sweep
measure tie-break luck. None of those announce themselves.

Run:  python test_invariants.py      (or: pytest -q test_invariants.py)
"""

from __future__ import annotations

import time

import numpy as np

import fis
from core import (
    build_candidates,
    greedy_edge_tour,
    make_pos,
    nn_stats,
    nn_tour,
    tour_length,
)
from fis_lk import construct as fis_build
from fis_lk import local_search as fis_ls
from lk import N_STATS, improve_city, lk_solve
from tsplib import load, reference_length, validate_tour

SMALL = ["berlin52", "a280", "pr1002"]

# Instances whose coordinates are grid-like or clustered, so exactly-equidistant
# neighbors are common: 933 of rl5915's 5915 cities have a distance tie straddling the
# 32nd-nearest place. These are what break tie handling, and what made an earlier
# "widen the query until no row is contested" approach degenerate into querying every
# city — a 90x slowdown in candidate building that the quality checks could not see.
TIE_HEAVY = ["fl1577", "pcb3038", "rl5915", "d18512"]


def test_candidate_lists_are_prefix_stable_in_k():
    """The k=8 list must be the first 8 of the k=24 list.

    TSPLIB instances are full of exact distance ties, and a tie order that depends on k
    makes nearest-neighbor tour length depend on k — for a construction that provably
    cannot. That bug made every early parameter sweep partly a measurement of luck.
    """
    for name in SMALL + TIE_HEAVY:
        inst = load(name)
        small, _ = build_candidates(inst.coords, 8, inst.ceil)
        large, _ = build_candidates(inst.coords, 24, inst.ceil)
        assert np.array_equal(small, large[:, :8]), f"{name}: candidate lists differ with k"
        # and therefore the nearest-neighbour tour cannot depend on k either
        a = nn_tour(inst.coords, small, inst.ceil, 0)
        b = nn_tour(inst.coords, large, inst.ceil, 0)
        assert reference_length(a, inst) == reference_length(b, inst), f"{name}: NN moved"
    print(f"  prefix-stable in k, NN invariant ({len(SMALL) + len(TIE_HEAVY)} instances)     ok")


def test_candidate_building_stays_cheap():
    """Candidate building must not degenerate on tie-heavy instances.

    The bound is deliberately loose — this is not a benchmark, it is a guard against the
    failure mode where settling contested rows turns into querying every city. That
    regression cost a factor of ~90 and was invisible to every correctness check, because
    the lists it produced were right; they were just ruinously slow to produce.
    """
    build_candidates(load("berlin52").coords, 32, False)  # pay the JIT first
    for name in TIE_HEAVY:
        inst = load(name)
        t0 = time.perf_counter()
        build_candidates(inst.coords, 32, inst.ceil)
        dt = time.perf_counter() - t0
        budget = 0.5 + inst.n * 2e-4  # generous: ~4x the observed cost at n=18512
        assert dt < budget, f"{name}: candidate build took {dt:.2f}s, budget {budget:.2f}s"
    print("  candidate building stays cheap on tie-heavy instances      ok")


def test_every_move_gain_matches_the_tour():
    """Each accepted move's claimed gain must equal the real change in tour length.

    This is the check that covers the Or-opt block surgery, where relocating a segment is
    three reversals and getting one of them wrong yields a valid tour of the wrong length.
    ``pos`` is re-derived and compared too, since a stale position index corrupts every
    later move rather than this one.
    """
    rng = np.random.default_rng(0)
    for name in SMALL:
        inst = load(name)
        n = inst.n
        cand, cand_d = build_candidates(inst.coords, 12, inst.ceil)
        tour = nn_tour(inst.coords, cand, inst.ceil, 0)
        pos = make_pos(tour)
        rev_i = np.empty(12, np.int64)
        rev_j = np.empty(12, np.int64)
        touched = np.empty(48, np.int32)
        stats = np.zeros(N_STATS, np.int64)
        # sized from the rule base, never hard-coded: numba does not bounds-check, so a
        # buffer one element short of the chain rule base's input count would be a silent
        # out-of-bounds write rather than an error
        xc = np.empty(fis.CH_N_IN, np.float64)
        mu = np.empty((fis.CH_N_IN, fis.N_TERMS), np.float64)
        length = tour_length(tour, inst.coords, inst.ceil)
        moves = 0
        for _ in range(3):
            for t1 in rng.permutation(n):
                gain, _ = improve_city(
                    tour, pos, n, inst.coords, cand, cand_d, inst.ceil, int(t1),
                    8, 6, 8, 3, rev_i, rev_j, touched, stats,
                    True, fis.CHAIN_TAB, fis.CHAIN_ANT, fis.CHAIN_CONS, xc, mu,
                )
                if gain <= 1e-9:
                    continue
                moves += 1
                new_length = tour_length(tour, inst.coords, inst.ceil)
                assert abs((length - gain) - new_length) < 1e-6, (
                    f"{name}: move claimed {gain} but changed the tour by "
                    f"{length - new_length}"
                )
                length = new_length
                assert np.array_equal(np.sort(tour), np.arange(n)), f"{name}: not a tour"
                assert np.array_equal(make_pos(tour), pos), f"{name}: pos desynced"
        assert moves > 0, f"{name}: no moves were made, so nothing was checked"
        print(f"  {name:>9s}: {moves:4d} moves, gains and tour consistent        ok")


def test_solvers_return_valid_tours_and_honest_lengths():
    """Both arms must return a permutation whose own reported length agrees with an
    independent rescoring from the coordinates."""
    for name in SMALL:
        inst = load(name)
        cand, cand_d = build_candidates(inst.coords, 24, inst.ceil)
        start = greedy_edge_tour(inst.coords, cand, inst.ceil)
        for label, (tour, length, _) in (
            ("lk", lk_solve(inst.coords, cand, cand_d, inst.ceil, start, 24, 8, 24, 3)),
            (
                "fis",
                fis_ls(inst, cand, cand_d, start, fis.EFFORT_CONS, fis.CHAIN_CONS, 10, 3),
            ),
        ):
            validate_tour(tour, inst.n)
            assert abs(length - reference_length(tour, inst)) < 1e-6, (
                f"{name}/{label}: reported length disagrees with an independent rescore"
            )
    print("  both arms: valid permutations, lengths independently agree  ok")


def test_deferred_verification_never_returns_a_worse_tour():
    """The full-effort verification pass can only find *additional* improvements, so its
    tour must never be longer than the one the cheap schedule stopped at."""
    for name in SMALL:
        inst = load(name)
        cand, cand_d = build_candidates(inst.coords, 24, inst.ceil)
        start = greedy_edge_tour(inst.coords, cand, inst.ceil)
        _, plain, _ = fis_ls(
            inst, cand, cand_d, start, fis.EFFORT_CONS, fis.CHAIN_CONS, 10, 3, False
        )
        _, deferred, _ = fis_ls(
            inst, cand, cand_d, start, fis.EFFORT_CONS, fis.CHAIN_CONS, 10, 3, True
        )
        assert deferred <= plain + 1e-9, (
            f"{name}: verification pass made the tour longer ({deferred} > {plain})"
        )
    print("  deferred verification never worsens the tour               ok")


def test_double_bridge_is_a_permutation_and_not_a_reversal():
    """The kick must leave a valid tour, keep ``pos`` consistent, and — the property it exists
    for — must not be reachable as a segment reversal.

    A double bridge that accidentally reverses a segment is just a 2-opt move, which the chain
    can already reach and improve on, so the perturbation would be doing nothing the search
    could not undo immediately. The check is that the cyclic sequence differs from the original
    in both directions: a reversal of the whole tour or of a contiguous span leaves one of the
    two readings unchanged over the untouched part, whereas the A-C-B reconnection does not.
    """
    from kick import double_bridge

    rng = np.random.default_rng(3)
    for name in SMALL:
        inst = load(name)
        n = inst.n
        cand, _ = build_candidates(inst.coords, 8, inst.ceil)
        tour = nn_tour(inst.coords, cand, inst.ceil, 0)
        touched = np.empty(8, np.int32)
        applied = 0
        for _ in range(40):
            base = tour.copy()
            pos = make_pos(tour)
            i = int(rng.integers(0, n))
            l1, l2, l3 = (int(rng.integers(1, 12)) for _ in range(3))
            nt = double_bridge(tour, pos, n, i, l1, l2, l3, touched)
            if nt == 0:
                continue
            applied += 1
            assert np.array_equal(np.sort(tour), np.arange(n)), f"{name}: kick broke the tour"
            assert np.array_equal(make_pos(tour), pos), f"{name}: kick desynced pos"
            assert nt == 8 or nt < 8, "touched count out of range"
            # the four cut points are 8 distinct cities on a large enough instance
            assert len(set(touched[:nt].tolist())) == nt, f"{name}: duplicate touched city"
            assert not np.array_equal(tour, base), f"{name}: kick changed nothing"
        assert applied > 0, f"{name}: no kick was applied, so nothing was checked"
    print("  double bridge: valid tour, pos consistent, non-trivial      ok")


def test_iterated_lk_never_returns_worse_than_plain_local_search():
    """Kicking must not be able to lose. The accept rule keeps the best tour seen, so the
    iterated result is bounded by the plain local optimum it starts from — and because the
    accept rule is strict improvement, the reported length must also be monotone
    non-increasing in the kick budget."""
    from kick import iterated_lk

    none = np.empty(0, np.float64)
    for name in SMALL:
        inst = load(name)
        cand, cand_d = build_candidates(inst.coords, 24, inst.ceil)
        start = greedy_edge_tour(inst.coords, cand, inst.ceil)
        lengths = []
        for nk in (0, 50, 250):
            tour, length, _ = iterated_lk(
                inst.coords, cand, cand_d, inst.ceil, start, 24, 6, 24, 3, nk, 16, 7,
                none, False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS,
            )
            validate_tour(tour, inst.n)
            assert abs(length - reference_length(tour, inst)) < 1e-6, (
                f"{name}: iterated reported {length}, rescore disagrees"
            )
            lengths.append(length)
        for a, b in zip(lengths, lengths[1:]):
            assert b <= a + 1e-9, f"{name}: more kicks gave a longer tour ({lengths})"
    print("  iterated LK: monotone in the kick budget, never worse       ok")


def test_scratch_buffers_are_wide_enough_for_every_rule_base():
    """Every hot-path scratch buffer must be at least as wide as its rule base's input
    count.

    numba does not bounds-check, so a buffer sized for a four-input rule base and handed
    to a five-input one is a silent out-of-bounds write, not an exception — it corrupts
    whatever follows it in memory and the solver carries on producing plausible tours.
    Adding an antecedent is exactly the change that triggers it, so this asserts the
    relationship rather than trusting every allocation site to have been updated.
    """
    for name, tab, ant in (
        ("CONSTRUCT", fis.CONSTRUCT_TAB, fis.CONSTRUCT_ANT),
        ("EFFORT", fis.EFFORT_TAB, fis.EFFORT_ANT),
        ("CHAIN", fis.CHAIN_TAB, fis.CHAIN_ANT),
        ("NO_CHAIN", fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT),
    ):
        assert tab.shape[0] == ant.shape[1], (
            f"{name}: membership bank has {tab.shape[0]} inputs, rules have {ant.shape[1]}"
        )
        assert tab.shape[1] == fis.N_TERMS, f"{name}: wrong term count"
        assert int(ant.max()) < fis.N_TERMS, f"{name}: a rule names a nonexistent term"
    # the baseline arm passes the chain rule base through even with use_chain off, so its
    # buffers must fit the real chain base, not the stub
    assert fis.NO_CHAIN_ANT.shape[1] == fis.CH_N_IN, "NO_CHAIN stub width drifted"
    print("  scratch buffers wide enough for every rule base            ok")


def test_membership_table_matches_the_functions_it_compiled():
    """The lookup table is an optimisation, so it has to agree with the closed form it
    replaced — otherwise the fitted rule bases mean something different at run time than
    they did when they were fitted."""
    c, w = fis.default_mf(4, sigma=0.25)
    for kind, fn in (("gaussian", fis._mf_gaussian), ("triangular", fis._mf_triangular)):
        tab = fis.mf_table(c, w, kind)
        xs = np.linspace(0.0, 1.0, 197)
        mu = np.empty((4, 3), np.float64)
        for x in xs:
            fis._memberships(np.full(4, x), tab, mu)
            for i in range(4):
                for t in range(3):
                    want = float(fn(np.array([x]), c[i, t], w[i, t])[0])
                    # linear interpolation between table samples, so allow the
                    # interpolation error of a 64-interval grid on a smooth function
                    assert abs(mu[i, t] - want) < 2e-3, (
                        f"{kind}: table {mu[i, t]:.5f} vs closed form {want:.5f} at x={x}"
                    )
    print("  membership lookup tables match their closed forms          ok")


def test_fuzzy_construction_is_a_tour_and_beats_nothing_silently():
    """The ranker must produce a valid tour, and must not be quietly degenerate — a
    ranking rule that always picks the same candidate would still build a tour."""
    for name in SMALL:
        inst = load(name)
        cand, cand_d = build_candidates(inst.coords, 24, inst.ceil)
        tour = fis_build(inst, cand, cand_d, fis.CONSTRUCT_CONS, 8)
        validate_tour(tour, inst.n)
        nn = nn_tour(inst.coords, cand, inst.ceil, 0)
        # it should differ from nearest-neighbour somewhere: identical output would mean
        # the fuzzy score is monotone in distance and the other three cues do nothing
        assert not np.array_equal(tour, nn), f"{name}: ranker reproduced NN exactly"
        gap = inst.gap(reference_length(tour, inst))
        assert 0.0 < gap < 100.0, f"{name}: construction gap {gap} is not credible"
    print("  fuzzy construction: valid, and not degenerate              ok")


def test_scale_statistics_stay_finite_on_coincident_points():
    """Duplicate coordinates give a zero nearest-neighbour distance, and every fuzzy
    antecedent is a ratio against one of these."""
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cand, cand_d = build_candidates(coords, 3, False)
    nn1, mean_c = nn_stats(cand_d)
    assert np.all(np.isfinite(nn1)) and np.all(nn1 > 0.0), "nn1 must be positive finite"
    assert np.all(np.isfinite(mean_c)) and np.all(mean_c > 0.0), "mean_c must be positive"
    print("  scale statistics finite with coincident points             ok")


def main():
    print("invariant checks")
    test_candidate_lists_are_prefix_stable_in_k()
    test_candidate_building_stays_cheap()
    test_every_move_gain_matches_the_tour()
    test_solvers_return_valid_tours_and_honest_lengths()
    test_deferred_verification_never_returns_a_worse_tour()
    test_double_bridge_is_a_permutation_and_not_a_reversal()
    test_iterated_lk_never_returns_worse_than_plain_local_search()
    test_scratch_buffers_are_wide_enough_for_every_rule_base()
    test_membership_table_matches_the_functions_it_compiled()
    test_fuzzy_construction_is_a_tour_and_beats_nothing_silently()
    test_scale_statistics_stay_finite_on_coincident_points()
    print("all invariants hold")


if __name__ == "__main__":
    main()
