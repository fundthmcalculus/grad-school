"""Invariant checks for the things that would silently produce good-looking wrong numbers.

Every check here exists because breaking it once produced a plausible result rather than
a crash. A solver that drops a city reports a shorter tour; a move whose claimed gain does
not match what it did to the tour makes the local search converge to the wrong place while
still terminating; a candidate list whose tie order shifts with k makes a parameter sweep
measure tie-break luck. None of those announce themselves.

Run:  python test_invariants.py      (or: pytest -q test_invariants.py)
"""

from __future__ import annotations

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
from lk import improve_city, lk_solve
from tsplib import load, reference_length, validate_tour

SMALL = ["berlin52", "a280", "pr1002"]


def test_candidate_lists_are_prefix_stable_in_k():
    """The k=8 list must be the first 8 of the k=24 list.

    TSPLIB instances are full of exact distance ties, and a tie order that depends on k
    makes nearest-neighbour tour length depend on k — for a construction that provably
    cannot. That bug made every early parameter sweep partly a measurement of luck.
    """
    for name in SMALL:
        inst = load(name)
        small, _ = build_candidates(inst.coords, 8, inst.ceil)
        large, _ = build_candidates(inst.coords, 24, inst.ceil)
        assert np.array_equal(small, large[:, :8]), f"{name}: candidate lists differ with k"
        # and therefore the nearest-neighbour tour cannot depend on k either
        a = nn_tour(inst.coords, small, inst.ceil, 0)
        b = nn_tour(inst.coords, large, inst.ceil, 0)
        assert reference_length(a, inst) == reference_length(b, inst), f"{name}: NN moved"
    print("  candidate lists prefix-stable in k, NN invariant           ok")


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
        stats = np.zeros(fis.N_TERMS * 0 + 6, np.int64)
        xc = np.empty(4, np.float64)
        mu = np.empty((4, 3), np.float64)
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
    test_every_move_gain_matches_the_tour()
    test_solvers_return_valid_tours_and_honest_lengths()
    test_deferred_verification_never_returns_a_worse_tour()
    test_membership_table_matches_the_functions_it_compiled()
    test_fuzzy_construction_is_a_tour_and_beats_nothing_silently()
    test_scale_statistics_stay_finite_on_coincident_points()
    print("all invariants hold")


if __name__ == "__main__":
    main()
