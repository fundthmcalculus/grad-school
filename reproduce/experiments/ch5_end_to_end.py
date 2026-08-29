#!/usr/bin/env python3
"""Fit a model on Chapter 5's antecedents and score it held out.

Checklist **C3**: the one missing arrow. Chapter 5 builds a graded membership
matrix and throws it away. `reproduce/tables/table_5_4_ch5_g1_scaling.py:91`
already stacks

    U = np.vstack([MS.block_membership(b, Dstar) for b in blocks])

normalizes it to a partition of unity (`:92`), and then uses it for exactly one
thing: its own residual `np.abs(s[covered] - 1.0)`. That residual is a tautology
-- `MS.normalize_partition` forces the column sums to 1 and the script then
measures how far they are from 1, so the number is machine epsilon by
construction. Every other Chapter 5 path calls `MS.assign` (a hard argmin over
minimax distance) and scores it with `adjusted_rand_score`. So the chapter
manufactures FIS antecedents, never fits a model on them, and reports a
clustering proxy instead. This script draws the arrow.

The bridge already exists and needs no new machinery:
`tribblefis.regression.solve_tsk_consequents_from_firing` takes a PRECOMPUTED
firing matrix, which is what `U.T` is, and
`tribblefis.regression.apply_tsk_consequents` evaluates the fitted consequents
against a held-out firing matrix.

WHAT THE MEMBERSHIP FUNCTION CAN AND CANNOT DO, BEFORE ANY SCORE
----------------------------------------------------------------
One structural fact governs every number below, and it is a theorem about
`block_membership`, not a property of any dataset. The blocks are single-linkage
dendrogram nodes, so a point outside a block can only reach it across the MST
edge that dissolves the block into its parent: d_B(x) >= death_B for EVERY
non-member, and d_B(x) = 0 for every member (the block's own row is in the min).
With the default Gaussian kernel `mu = 2**(-(d/death)**2)`
(`gated-minimax-selection/multiscale_persistence.py:421-426`), which puts
half-max exactly at `death`, that pins

    mu_B(x) = 1                      for members of B
    mu_B(x) <= 0.5                   for every non-member of B, on any data.

So U is saturated-or-skirt by construction, with nothing in (0.5, 1). The run
emits both halves of that (`max mu over non-members`, `min mu over members`) as
diagnostics so the claim is measured here rather than inherited. This is the
supervised-objective face of what
`gated-minimax-selection/notes/phase6_soft_validation.py` found against the
analytic posterior, and it is why the near-uniform normalized firing reported in
the diagnostics is expected rather than surprising.

WHAT SEES WHAT, AND WHY THAT IS SOUND
-------------------------------------
The split is by point index: one `KFold(n_splits=5, shuffle=True,
random_state=seed)` object per seed, and every arm consumes the identical
`(train_idx, test_idx)` pairs from it. `KFold` on n=252 depends only on the seed,
so the splits are structurally identical rather than identical by coincidence.
(The emitted `folds` column reports how many folds each arm actually recorded,
because an arm is scorable only on a fold where a cover was selected.)

  * The feature scaler is fitted on TRAIN ROWS ONLY and then applied to all 252
    rows (`FuzzySystemsExperiments/bodyfat.py:227-232` fits inside the fold; we
    match it). No test row contributes a mean, a range or a log bound.
  * `y` is touched ONLY on train rows, in every arm, at the consequent solve.
    No held-out target reaches any fit.
  * The antecedent geometry -- the minimax D*, the block cover, the
    memberships -- is UNSUPERVISED and, in the transductive arms, sees the
    feature matrix of all 252 rows. That is transduction, not leakage, and it is
    the standing convention of every Chapter 5 table
    (`table_3_7_g2_downstream.py` builds one 5000x5000 D* and selects blocks from
    it before any evaluation). It is a real caveat and it is stated in the note.
    It is NOT a comparability requirement: a first pass of this file claimed
    transduction "is what makes the crisp control and the graded arm comparable,
    since they then differ ONLY in gradation", which is a non-sequitur -- an
    inductive crisp arm and an inductive graded arm also differ only in
    gradation. Both pairs are therefore run, and the INDUCTIVE pair is the one
    that answers falsifier #1.
  * The inductive arms see no held-out row at any stage. D* is built on train
    rows only, the cover is selected from train rows only, and each held-out
    point's minimax distance to each block comes from exact single-point
    bottleneck insertion,
        D*(x, j) = min_k max( d(x, x_k), D*_train(k, j) ),
    which is exact because every path out of a newly inserted point must leave
    it on a direct edge. Points are inserted one at a time and cannot route
    through each other, so no held-out point influences another's memberships
    and none influences the cover.

OUTCOMES, REGISTERED BEFORE THE RUN
-----------------------------------
  PASS      the Chapter 5 graded arm's held-out R2 lands within 0.05 of the
            Chapter 4 arm (`FuzzySystemsExperiments/bodyfat.py`, TribbleRegressor
            + TRIO_LOG_TUNED, reported R2 = 0.647). The minimax membership
            carries usable supervised information and Chapter 5's antecedents
            are a real front end for a TSK model.
  FAIL      it loses, materially. THIS IS THE PRIOR, NOT THE FEAR.
            `gated-minimax-selection/notes/phase6_soft_validation.py` already
            found the normalized minimax membership is "a constant step per
            cluster, with no boundary resolution", scoring WORSE than crisp 0/1
            labels against the analytic posterior (Brier 0.136/0.200/0.208/0.122
            fuzzy vs 0.096/0.042/0.016/0.000 hard,
            `notes/MF_PROGRESS_LOG.md:259-264`). A FAIL still discharges C3: the
            arrow is drawn and the number is a supervised held-out score rather
            than an ARI.
  FAIL-FLOOR the graded arm does not beat the trivial floor (predict the
            training mean, R2 = 0 by construction) or the single-rule global
            polynomial. That is a DIFFERENT and more important finding than
            losing to Chapter 4: it would mean the antecedents carry no
            supervised information at all, not merely less than Chapter 4's.
            The run computes which outcome fired and writes it into the emitted
            note, so the table declares its own verdict instead of leaving a
            reader to compare two cells and guess.

THE FALSIFICATION CONDITIONS, ALSO REGISTERED BEFORE THE RUN
------------------------------------------------------------
Arms that exist purely to refute a comfortable reading of the graded arm's
number, whichever way it goes.

  1. **Crisp controls** (hard labels one-hot into the SAME solver, same
     features, same order, same ridge). If graded >= crisp, phase6's "constant
     step per cluster" finding does NOT transfer to a supervised objective and
     that note needs amending. If graded == crisp to three decimals, `U` is
     one-hot in disguise and the graded/crisp distinction is cosmetic. TWO crisp
     controls run, because there are two defensible hardenings and they are not
     the same partition (their disagreement is emitted): `MS.assign`, the
     argmin-over-minimax-distance labels every other Chapter 5 path uses, and
     `argmax(U)`, the true hardening of this U. `block_membership` divides each
     block's distance by ITS OWN death height, so a high-death block can win the
     argmax at a larger minimax distance; the two disagree on most bodyfat
     points, and `multiscale_persistence.py:426`'s claim that "argmax still
     reproduces the crisp labels" is false off the equal-death case. Reporting
     both brackets the question rather than picking the flattering one.
  2. **Single-rule global polynomial** (firing = a column of ones, identical
     features / order / ridge). This is the same TSK machinery with the
     antecedents deleted. If the Chapter 5 arm ties it, the block memberships
     are contributing nothing and the score belongs to the consequent
     polynomial, not to Chapter 5. This is the arm that would make a
     good-looking R2 meaningless.
  3. **Trivial floors.** Train-mean (R2 = 0 by construction) anchors the sign of
     every other cell. Ordinary least squares on the same columns anchors its
     usefulness: `FuzzySystemsExperiments/bodyfat.py:312-314` already scores OLS
     on this dataset and `bodyfat_report.md:29-30,41` already reports that OLS
     on all 13 features "is the best model here". A Chapter 5 arm below plain
     linear regression on the same three columns has not earned its machinery,
     whatever it does against Chapter 4. (This is the repo's standing lesson
     that a trivial baseline gets scored before any win is read.)

WHAT THIS DOES NOT DO, STATED LOUDLY
------------------------------------
  * NO TUNING WAS DONE HERE, AND ONE PIECE OF TUNING IS INHERITED. Every TSK arm
    is pinned to the Chapter 4 arm's own consequent configuration --
    `order="2nd"`, `l2_reg=1e-2`, `pin_extremes=False`, the `Abdomen/Hip/Chest`
    trio, and the same `tribblefis.scaling.MinMaxScaler(log_features=TRIO)` -- so
    the ONLY thing that differs between the Chapter 4 and Chapter 5 arms is where
    the antecedents came from. Nothing is swept in this file. But that config is
    not neutral: `bodyfat.py:159-160` says `TRIO_LOG_TUNED` was "selected as the
    best of 64 configs scored on all the partitions, so it is mildly optimistic"
    -- the same 252 rows and the same fold partitions this script scores on. The
    asymmetry runs AGAINST Chapter 5: the search maximized the Chapter 4 arm, so
    the Chapter 4 row is the flattered one, and a negative Chapter 5 result is
    robust to it. `firing_exponent` in particular stays at 1.0: raising it
    converges the fuzzy arm onto the crisp one, which is how a "win" could be
    faked by silently becoming Chapter 4. The all-13 arms additionally
    transplant that ridge onto a 13-column z-scored design it was never swept
    for, and `l2_reg` is not scale-free (`bodyfat_report.md:51`); those rows are
    reported as a conditioning artifact, not as a modelling result.
  * NO CLUSTERING SCORE. No ARI, NMI or silhouette appears in the output. A
    clustering score is precisely the proxy C3 exists to replace, and reporting
    one would re-close the loop this script opens.
  * NOT PhiUSIIL. Chapter 5 needs a dense NxN dissimilarity; 235,795 rows is
    0.4 TB at float64. It is not a possible venue and is not attempted.
  * `predict_tsk` IS NOT USED, despite being the obvious name.
    `tribblefis.regression.predict_tsk:1042` takes a `GaussianMixtureModel` and
    recomputes the firing internally via `tsk_firing_strengths`; it cannot
    accept a precomputed U. Its split-out half `apply_tsk_consequents:1120` is
    the correct held-out call, and is what the repo's own precomputed-firing
    call sites use (`tribblefis/it2_refine.py:480`,
    `tribble-tree/fuzzytree/deconstruct.py:159`).
  * NOT THE MULTI-SCALE HIERARCHY. Chapter 5 headlines a two-stage construction
    (`MS.select_multiscale` -> `FuzzyHierarchy`), and this script uses the flat
    `S.select_coverage_cover` instead. That is not a choice between two
    available generators: `select_multiscale` discovers ZERO bands on the
    bodyfat trio geometry, so `band_memberships` has nothing to build there. The
    run measures and emits the band count rather than asserting it.

VENUES
------
bodyfat (252 rows) is primary: both arms already exist on it
(`FuzzySystemsExperiments/bodyfat.py` for Chapter 4 at R2 = 0.647,
`FuzzySystemsExperiments/bodyfat_ivat.py` for the iVAT side, though that file
never calls `block_membership` -- C3 is the first membership call site on this
dataset). ECG5000 runs as a second venue when `aeon` is importable, reusing the
DTW + low-memory minimax path `reproduce/tables/table_3_7_g2_downstream.py`
already builds; without `aeon` its table emits N/A rows rather than crashing.

WHAT CHANGED AFTER SCORES EXISTED
---------------------------------
Registration is only worth something if the departures from it are listed. This
file was written, then run, then reviewed adversarially, then extended. Nothing
below is a hyperparameter; every item was added because a reviewer showed the
first pass could not support a sentence it had written, and every item that moved
the conclusion moved it AGAINST Chapter 5.

  a. The three OLS floors and the inductive crisp arm did not exist in the first
     pass. OLS bounds the whole table from above; the inductive crisp arm is
     what actually resolves falsifier #1 (the transductive pair answers a
     leakier question, and the two pairs disagree in sign).
  b. The `argmax(U)` crisp control, and the emitted `MS.assign` vs `argmax(U)`
     disagreement, were added after that disagreement was measured to be large.
  c. The membership-shape diagnostics (`max mu over non-members`, `min mu over
     members`, `core-member fraction`) replace a `distinct membership rows`
     count that the first pass over-read as evidence of boundary resolution. The
     distinct-column count is retained but relabelled as what it is: a count of
     bottleneck-equivalence classes.
  d. Zero-firing counters now cover every graded arm and both sides of the fit,
     not just the inductive held-out side.
  e. `design_health` now runs on the crisp arms too, because the conditioning
     story the first pass told had no control of its own.
  f. The inductive-vs-joint D* discrepancy and the multiscale band count are
     computed instead of asserted; the first pass asserted "~7e-1" for the
     former and it is roughly 3x smaller than that.
  g. The two prediction-agreement rows (mean absolute gap and Pearson r against
     the single-rule global polynomial) were added after a three-seed pass
     showed the graded arm and that antecedent-free control agreeing to within
     one seed-sigma. They compare PREDICTIONS, not scores, and change no cell in
     `ch5_end_to_end`.
  h. `TribbleRegressor`'s fits are wrapped in `redirect_stdout` (it prints a
     feature-ranking banner on every fit) and in `catch_warnings(record=True)`,
     the latter so its under-filled-bucket warnings are COUNTED the way
     `bodyfat.py:250-256` counts them rather than silenced by a blanket filter.
  i. The ECG5000 DTW cache is keyed on a hash of the fetched series and stores
     the labels beside the matrix, so a stale cache is rebuilt instead of being
     paired with the wrong labels; and the ECG5000 venue gained inductive arms
     and the antecedent-free control it was missing.

Run (from repo root):

    source reproduce/hostenv.sh >/dev/null 2>&1
    uv run --project tribble-fis --with aeon python \
        reproduce/experiments/ch5_end_to_end.py

(`--with aeon` is only needed for the ECG5000 venue; bodyfat runs without it.)

Knobs:
    REPRO_SEEDS="0,1,2"         quick smoke run (ten is the protocol)
    REPRO_OUTPUT_DIR=...        redirect emit() (see common.py:31-37)
    REPRO_C3_SKIP_ECG=1         bodyfat only, even if aeon is present
    REPRO_C3_DTW_CACHE=...      directory for the cached ECG5000 DTW matrix
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# NOT a blanket `filterwarnings("ignore")`. `bodyfat.py:250-256` -- the function
# whose scoring definition this script matches -- deliberately CAPTURES the
# library's under-filled-bucket RuntimeWarning and reports the count, because a
# rule fitted on almost no data is invisible in an aggregate RMSE. Silencing
# everything would discard exactly that signal in a run that fits 6-16 minimax
# rules on 201 rows. Only the two categories that would otherwise flood the log
# are suppressed, and both are named.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_REPRO)
sys.path.insert(0, _REPRO)
sys.path.insert(0, os.path.join(_REPRO, "tables"))
sys.path.insert(0, os.path.join(_ROOT, "gated-minimax-selection"))
sys.path.insert(0, _ROOT)  # repo root -> `import repro_data`

import common as C  # noqa: E402
from repro_data import load_bodyfat  # noqa: E402
import ivat_mf as im  # noqa: E402
import multiscale_persistence as MS  # noqa: E402
import selection as S  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.regression import (  # noqa: E402
    apply_tsk_consequents,
    solve_tsk_consequents_from_firing,
)
from tribblefis.scaling import MinMaxScaler as FuzzyMinMaxScaler  # noqa: E402

# ---------------------------------------------------------------------------
# bodyfat: the Chapter 4 arm's own constants. The dataset load and the leak
# drop now live in the shared repro_data.load_bodyfat, so this arm and the
# Chapter 4 arm cannot drift apart on either. TARGET/LEAKY are retained here for
# the log line and as documentation of the protocol.
# ---------------------------------------------------------------------------
BODYFAT_CSV = os.path.join(_ROOT, "data", "bodyfat.csv")
TARGET = "BodyFat"  # bodyfat.py:84
LEAKY = "Density"  # bodyfat.py:85 -- the target in another coordinate:
# Siri's equation 495/Density - 450 reproduces BodyFat at R2 0.977. Dropped by
# repro_data.load_bodyfat for every arm; it is not a choice, it is the protocol.
TRIO = ["Abdomen", "Hip", "Chest"]  # bodyfat.py:121
TRIO_LOG_TUNED = {  # bodyfat.py:163-168 -- the config behind the reported 0.647
    "n_gaussians": 1,
    "n_output_buckets": 3,
    "tsk_order": "2nd",
    "l2_reg": 1e-2,
}
# The consequent solve, held FIXED across every TSK arm below. Taken from
# TRIO_LOG_TUNED (`tsk_order`, `l2_reg`) plus TribbleRegressor's own default
# `pin_extremes=False`, so the Chapter 5 arms and the Chapter 4 arm fit the same
# consequents and differ only in their antecedents. Inherited, not chosen here;
# the NO TUNING paragraph says what that inheritance costs and which side it
# flatters.
ORDER = TRIO_LOG_TUNED["tsk_order"]
L2 = TRIO_LOG_TUNED["l2_reg"]

N_SPLITS = 5  # bodyfat.py:627 default --folds 5
ZERO_FIRING = 1e-6  # tribblefis.gauss_data.ZERO_FIRING_THRESHOLD
RANK_TOL = 1e-8  # see design_health


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# Chapter 5 antecedents.
# ---------------------------------------------------------------------------
def block_distances(blocks, Dstar):
    """(k, n) minimax distance from each block core to every column of `Dstar`.

    This is the one quantity both the graded and the crisp side are built from:
    `block_membership` maps it through the Gaussian kernel, `MS.assign` takes its
    argmin. Computing it once means the two controls cannot silently diverge in
    anything but the map applied to it. Accepts a RECTANGULAR (n_train, n_all)
    matrix as well as a square one.
    """
    return np.vstack(
        [Dstar[np.fromiter(b["members"], dtype=int), :].min(axis=0) for b in blocks]
    )


def memberships_from_distances(blocks, dist):
    """Graded U (k, n) from a precomputed (k, n) block-distance matrix.

    The KERNEL is the library's, not a copy of it: each row is handed to
    `MS.block_membership` as a one-row distance matrix with a single member, so
    the block's own birth/death heights and the exact `2**(-(d/death)**2)` map
    are the shipped ones. The run emits `kernel reuse identity` -- the max
    absolute difference between this path and a direct `block_membership(b,
    Dstar)` call on the transductive fold -- so the equivalence is measured every
    run rather than argued once.
    """
    return np.vstack(
        [
            MS.block_membership(
                {"members": {0}, "birth": b["birth"], "death": b["death"]},
                dist[i][None, :],
            )
            for i, b in enumerate(blocks)
        ]
    )


def onehot(labels, k):
    """(k, n) one-hot firing from hard labels."""
    labels = np.asarray(labels, dtype=int)
    H = np.zeros((k, len(labels)))
    H[labels, np.arange(len(labels))] = 1.0
    return H


def ch5_transductive(Z_all):
    """Primary Chapter 5 arm. D* and the cover see all rows' FEATURES (never y).

    Returns (blocks, U, U_normalized, block_distances, Dstar); the (k, n_all)
    matrices are sliced by column into train and test parts.
    """
    Dstar = im.minimax_transform_fast(im.dissimilarity(Z_all))
    blocks = S.select_coverage_cover(Dstar)
    if not blocks:
        return None, None, None, None, Dstar
    dist = block_distances(blocks, Dstar)
    U = memberships_from_distances(blocks, dist)
    Un = MS.normalize_partition(U)  # the exact object table_5_4:92 discards
    return blocks, U, Un, dist, Dstar


def extend_block_distances(g, d_new_tr):
    """Block distances for points outside the training graph. (k, n_new).

        d_B(x) = min_j max( d(x, j), g_B[j] ),  g_B[j] = min_{m in B} D*_tr(m, j)

    which is the single-point bottleneck insertion
    `D*(x, j) = min_k max(d(x, x_k), D*_tr(k, j))` with the min over j and the
    min over k commuted -- exact, and the reason this arm is affordable at all.
    Materializing the (n_new, n_tr, n_tr) insertion tensor and reducing it
    afterwards is 16 MB on bodyfat but 137 GiB on ECG5000, which is why a first
    pass reported the inductive ECG arm as "derivable, not done". Per block this
    form is (n_new, n_tr).
    """
    return np.vstack(
        [np.min(np.maximum(d_new_tr, g[i][None, :]), axis=1) for i in range(len(g))]
    )


def ch5_inductive_from_D(D_tr, d_new_tr):
    """Chapter 5 arm that sees no held-out row at any stage, from distances.

    `D_tr` is the raw (not minimax) train-by-train dissimilarity; `d_new_tr` is
    the raw (n_new, n_tr) dissimilarity from each held-out point to each train
    point. D*, the cover and every membership are built from train rows only.
    Returns (blocks, U, U_normalized, dist) with columns ordered
    [train rows in `tr` order | held-out rows in `te` order].
    """
    Dstar_tr = im.minimax_transform_fast(D_tr)
    blocks = S.select_coverage_cover(Dstar_tr)
    if not blocks:
        return None, None, None, None
    g = block_distances(blocks, Dstar_tr)
    dist = np.hstack([g, extend_block_distances(g, d_new_tr)])
    U = memberships_from_distances(blocks, dist)
    return blocks, U, MS.normalize_partition(U), dist


# ---------------------------------------------------------------------------
# The bridge: precomputed firing -> TSK consequents -> held-out prediction.
# ---------------------------------------------------------------------------
def tsk_from_firing(F_tr, F_te, X_tr, X_te, cols, y_tr, order=ORDER, l2=L2):
    """Fit consequents on train firing, evaluate on test firing.

    `y_bucket_mean=None` / `pin_extremes=False` are not defaults being accepted;
    they are required. The pin reads `y_bucket_mean[labels_train[0]]`, which is
    only meaningful when rule index == output-bucket index. That holds for
    `TribbleRegressor` (one rule per output bucket) and emphatically does not
    hold for minimax blocks -- block 0 has no relationship to the lowest target
    bucket -- so pinning would nail two arbitrary blocks' constants to the target
    extremes. Both existing precomputed-firing call sites make the same choice
    (`it2_refine.py:480-484`, `tribble-tree/fuzzytree/deconstruct.py:159-163`).
    """
    n_rules = F_tr.shape[1]
    labels = list(range(n_rules))
    corr, y_rule_const = solve_tsk_consequents_from_firing(
        F_tr,
        labels,
        X_tr,
        cols,
        None,
        pd.DataFrame({"y_value": np.asarray(y_tr, float)}),
        order=order,
        l2_reg=l2,
        basis="raw",
        pin_extremes=False,
        verbose=False,
    )
    return apply_tsk_consequents(
        X_te, cols, F_te, labels, y_rule_const, corr, order=order, basis="raw"
    )


def design_health(F_tr, Z_tr, order=ORDER):
    """(condition number, numerical rank, n_columns) of the stacked rule design.

    Reported because a near-uniform firing matrix makes every rule's design block
    a near-copy of every other's. The rank tolerance is `RANK_TOL = 1e-8`
    RELATIVE to the leading singular value, and it is stated because the choice
    matters: at 1e-12 a design at condition number 6e5 still counts as full rank,
    so the rank row could only ever confirm itself, and even at 1e-8 a design is
    full rank by arithmetic whenever its condition number is below 1e8. Neither
    quantity explains anything on its own: the crisp all-13 arm is BOTH the more
    rank-deficient and the worse-conditioned of the two all-13 arms, and it is
    the one that does not blow up. So the crisp arms are measured too, and the
    note reads neither number as a cause.
    """
    if order == "0th":
        phi = np.ones((len(Z_tr), 1))
    elif order == "1st":
        phi = np.hstack([np.ones((len(Z_tr), 1)), Z_tr])
    else:  # "2nd" -- [1 | x | x^2], matching build_consequent_features
        phi = np.hstack([np.ones((len(Z_tr), 1)), Z_tr, Z_tr**2])
    design = (F_tr[:, :, None] * phi[:, None, :]).reshape(len(Z_tr), -1)
    sv = np.linalg.svd(design, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))
    rank = int((sv > sv[0] * RANK_TOL).sum())
    return cond, rank, design.shape[1]


def zero_fire(F):
    """Rows of a firing matrix that underflow the library's zero-firing rule.

    On the TRAIN side such a row is dropped from the ridge normal equations
    ("effectively ignored by the fit", `_normalize_firing_strengths`' own
    docstring), so a graded arm can be fitted on fewer rows than the crisp arm
    beside it. On the TEST side it is predicted as exactly 0.0. Both are silent,
    both are asymmetric between the graded and crisp arms, and both are therefore
    counted on every arm rather than on one.
    """
    return int((np.asarray(F).sum(axis=1) <= ZERO_FIRING).sum())


def empty_rules(F_tr):
    """Rules receiving no training mass at all -- the crisp arm's analogue of a
    zero-firing row. Their consequents come back as the min-norm 0 and predict
    exactly 0.0 for anything assigned to them."""
    return int((np.asarray(F_tr).sum(axis=0) <= ZERO_FIRING).sum())


# ---------------------------------------------------------------------------
# bodyfat venue
# ---------------------------------------------------------------------------
A_GRADED = "Ch5 graded U -> TSK (trio)"
A_ASSIGN = "Ch5 crisp assign -> TSK (trio)"
A_ARGMAX = "Ch5 crisp argmax U -> TSK (trio)"
A_GRADED_I = "Ch5 graded U inductive -> TSK (trio)"
A_ASSIGN_I = "Ch5 crisp assign inductive -> TSK (trio)"
A_GRADED13 = "Ch5 graded U -> TSK (all 13)"
A_ASSIGN13 = "Ch5 crisp assign -> TSK (all 13)"
A_CH4 = "Ch4 TribbleRegressor (trio, log-tuned)"
A_CH4_13 = "Ch4 TribbleRegressor (all 13, default)"
A_GLOBAL = "global 1-rule TSK (trio)"
A_GLOBAL13 = "global 1-rule TSK (all 13)"
A_OLS = "floor: OLS (trio)"
A_OLS_LOG = "floor: OLS (trio, log1p-scaled)"
A_OLS13 = "floor: OLS (all 13)"
A_MEAN = "floor: predict train mean"


def run_bodyfat():
    if not os.path.exists(BODYFAT_CSV):
        print(f"  [skip] {BODYFAT_CSV} not found")
        return None

    # Shared loader: drops `Density` (the LEAK) and `BodyFat` (the target),
    # returning the 13-feature frame this arm fits. `df` here is that frame --
    # it no longer carries TARGET/LEAKY, and every use below indexes only TRIO
    # and feats13, so the substitution is exact (verified frame-identical to the
    # old inline read). See repro_data.load_bodyfat and dataset_specs.yaml.
    df, y_series = load_bodyfat()
    y = y_series.to_numpy(float)
    feats13 = list(df.columns)
    print(f"  bodyfat: {len(df)} rows, target {TARGET}, {LEAKY} dropped as a leak")
    print(f"  trio    = {TRIO}")
    print(f"  all-13  = {feats13}")

    arms = [
        A_GRADED,
        A_ASSIGN,
        A_ARGMAX,
        A_GRADED_I,
        A_ASSIGN_I,
        A_GRADED13,
        A_ASSIGN13,
        A_CH4,
        A_CH4_13,
        A_GLOBAL,
        A_GLOBAL13,
        A_OLS,
        A_OLS_LOG,
        A_OLS13,
        A_MEAN,
    ]
    per_seed = {a: [] for a in arms}
    worst_fold = {a: 1e9 for a in arms}
    n_folds = {a: 0 for a in arms}
    raw_rows = []
    diag = defaultdict(list)

    for seed in C.SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_scores = {a: [] for a in arms}
        # One split object per seed; EVERY arm below consumes these same indices.
        for fold_idx, (tr, te) in enumerate(kf.split(df)):
            # ---- scalers: fitted on TRAIN ROWS ONLY, applied to all rows ----
            sc_trio = FuzzyMinMaxScaler(log_features=list(TRIO)).fit(df[TRIO].iloc[tr])
            Z_trio = np.asarray(sc_trio.transform(df[TRIO]), float)
            sc_13 = StandardScaler().fit(df[feats13].iloc[tr])
            Z_13 = np.asarray(sc_13.transform(df[feats13]), float)

            X_trio = pd.DataFrame(Z_trio, columns=TRIO)
            X_13 = pd.DataFrame(Z_13, columns=feats13)
            Xtr_trio = X_trio.iloc[tr].reset_index(drop=True)
            Xte_trio = X_trio.iloc[te].reset_index(drop=True)
            Xtr_13 = X_13.iloc[tr].reset_index(drop=True)
            Xte_13 = X_13.iloc[te].reset_index(drop=True)
            kept: dict[str, np.ndarray] = {}

            def record(arm, pred, idx=te, fold=fold_idx):
                pred = np.asarray(pred, float).ravel()
                kept[arm] = pred
                r2 = r2_score(y[idx], pred)
                fold_scores[arm].append((_rmse(y[idx], pred), r2))
                worst_fold[arm] = min(worst_fold[arm], r2)
                n_folds[arm] += 1
                # The SPLIT's fold ordinal, not a per-arm counter: a per-arm
                # counter renumbers every later fold of a seed if one arm skips
                # one, silently breaking the (seed, fold) join this CSV exists
                # to support.
                raw_rows.append(
                    [
                        "bodyfat",
                        arm,
                        seed,
                        fold,
                        f"{_rmse(y[idx], pred):.6f}",
                        f"{r2:.6f}",
                    ]
                )

            # ---------------- Chapter 5, transductive, trio ----------------
            blocks_t, U_t, Un_t, dist_t, Dstar_t = ch5_transductive(Z_trio)
            if blocks_t is not None:
                k = len(blocks_t)
                mem_mask = np.zeros(U_t.shape, dtype=bool)
                for i, b in enumerate(blocks_t):
                    mem_mask[i, np.fromiter(b["members"], dtype=int)] = True
                diag["k_trans"].append(k)
                diag["cov_trans"].append(S.coverage_of(blocks_t, len(df)))
                diag["maxmem_trans"].append(float(Un_t.max(axis=0).mean()))
                diag["unif_trans"].append(1.0 / k)
                diag["mu_nonmember_max"].append(float(U_t[~mem_mask].max()))
                diag["mu_member_min"].append(float(U_t[mem_mask].min()))
                diag["core_frac"].append(float(mem_mask.any(axis=0).mean()))
                diag["distinct_cols"].append(
                    len(np.unique(np.round(Un_t.T, 9), axis=0))
                )
                # The kernel-reuse check for `memberships_from_distances`.
                U_ref = np.vstack([MS.block_membership(b, Dstar_t) for b in blocks_t])
                diag["kernel_identity"].append(float(np.abs(U_ref - U_t).max()))
                lab_assign = dist_t.argmin(axis=0)
                lab_argmax = U_t.argmax(axis=0)
                diag["assign_vs_argmax"].append(
                    float((lab_assign != lab_argmax).mean())
                )
                # `select_multiscale`'s band count on the same geometry: 0 means
                # the flat cover is the only Chapter 5 generator available here.
                diag["ms_bands"].append(len(MS.select_multiscale(Dstar_t).bands))

                F_tr, F_te = Un_t[:, tr].T, Un_t[:, te].T
                cond, rank, ncol = design_health(F_tr, Z_trio[tr])
                diag["cond_trans"].append(cond)
                diag["rank_trans"].append(rank)
                diag["ncol_trans"].append(ncol)
                diag["zf_te_trio"].append(zero_fire(F_te))
                diag["zf_tr_trio"].append(zero_fire(F_tr))
                record(
                    A_GRADED,
                    tsk_from_firing(F_tr, F_te, Xtr_trio, Xte_trio, TRIO, y[tr]),
                )
                H = onehot(lab_assign, k)
                cond_c, rank_c, _ = design_health(H[:, tr].T, Z_trio[tr])
                diag["cond_crisp_trio"].append(cond_c)
                diag["rank_crisp_trio"].append(rank_c)
                diag["empty_crisp_trio"].append(empty_rules(H[:, tr].T))
                record(
                    A_ASSIGN,
                    tsk_from_firing(
                        H[:, tr].T, H[:, te].T, Xtr_trio, Xte_trio, TRIO, y[tr]
                    ),
                )
                Hm = onehot(lab_argmax, k)
                record(
                    A_ARGMAX,
                    tsk_from_firing(
                        Hm[:, tr].T, Hm[:, te].T, Xtr_trio, Xte_trio, TRIO, y[tr]
                    ),
                )

            # ---------------- Chapter 5, inductive, trio ----------------
            D_tr = im.dissimilarity(Z_trio[tr])
            d_te_tr = cdist(Z_trio[te], Z_trio[tr])
            blocks_i, U_i, Un_i, dist_i = ch5_inductive_from_D(D_tr, d_te_tr)
            if blocks_i is not None:
                n_tr, ki = len(tr), len(blocks_i)
                F_tr_i, F_te_i = Un_i[:, :n_tr].T, Un_i[:, n_tr:].T
                diag["k_ind"].append(ki)
                diag["cov_ind"].append(S.coverage_of(blocks_i, n_tr))
                diag["zf_te_ind"].append(zero_fire(F_te_i))
                diag["zf_tr_ind"].append(zero_fire(F_tr_i))
                if Dstar_t is not None:
                    # How far single-point insertion is from the joint graph, for
                    # the blocks actually in play. Measured, not asserted: adding
                    # several points at once opens new MST routes between train
                    # points, and a first pass guessed this gap at ~7e-1.
                    cols = np.concatenate([tr, te])
                    joint = np.vstack(
                        [
                            Dstar_t[
                                np.ix_(tr[np.fromiter(b["members"], dtype=int)], cols)
                            ].min(axis=0)
                            for b in blocks_i
                        ]
                    )
                    diag["ind_vs_joint"].append(float(np.abs(joint - dist_i).max()))
                    diag["dstar_max"].append(float(Dstar_t.max()))
                record(
                    A_GRADED_I,
                    tsk_from_firing(F_tr_i, F_te_i, Xtr_trio, Xte_trio, TRIO, y[tr]),
                )
                Hi = onehot(dist_i.argmin(axis=0), ki)
                record(
                    A_ASSIGN_I,
                    tsk_from_firing(
                        Hi[:, :n_tr].T,
                        Hi[:, n_tr:].T,
                        Xtr_trio,
                        Xte_trio,
                        TRIO,
                        y[tr],
                    ),
                )

            # ---------------- Chapter 5, transductive, all 13 ----------------
            blocks_a, U_a, Un_a, dist_a, _ = ch5_transductive(Z_13)
            if blocks_a is not None:
                ka = len(blocks_a)
                Fa_tr, Fa_te = Un_a[:, tr].T, Un_a[:, te].T
                cond, rank, ncol = design_health(Fa_tr, Z_13[tr])
                diag["cond13"].append(cond)
                diag["rank13"].append(rank)
                diag["ncol13"].append(ncol)
                diag["zf_te_13"].append(zero_fire(Fa_te))
                diag["zf_tr_13"].append(zero_fire(Fa_tr))
                record(
                    A_GRADED13,
                    tsk_from_firing(Fa_tr, Fa_te, Xtr_13, Xte_13, feats13, y[tr]),
                )
                Ha = onehot(dist_a.argmin(axis=0), ka)
                cond_c, rank_c, _ = design_health(Ha[:, tr].T, Z_13[tr])
                diag["cond_crisp13"].append(cond_c)
                diag["rank_crisp13"].append(rank_c)
                diag["empty_crisp13"].append(empty_rules(Ha[:, tr].T))
                record(
                    A_ASSIGN13,
                    tsk_from_firing(
                        Ha[:, tr].T, Ha[:, te].T, Xtr_13, Xte_13, feats13, y[tr]
                    ),
                )

            # ---------------- Chapter 4 baselines ----------------
            # stdout is swallowed only to keep the run log readable -- the
            # estimator prints a feature-ranking banner per fit, 100 times over
            # (`FuzzySystemsExperiments/bodyfat.py:252` does the same). The
            # warning capture is bodyfat.py:250-256's, and its count is emitted.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with contextlib.redirect_stdout(io.StringIO()):
                    m = TribbleRegressor(random_state=0, **TRIO_LOG_TUNED)
                    m.fit(Xtr_trio, y[tr])
                    p_trio = m.predict(Xte_trio)
                    m13 = TribbleRegressor(random_state=0)
                    m13.fit(Xtr_13, y[tr])
                    p_13 = m13.predict(Xte_13)
            diag["starved_ch4"].append(
                sum("under-filled" in str(w.message) for w in caught)
            )
            record(A_CH4, p_trio)
            record(A_CH4_13, p_13)

            # ---------------- The floors ----------------
            ones_tr = np.ones((len(tr), 1))
            ones_te = np.ones((len(te), 1))
            record(
                A_GLOBAL,
                tsk_from_firing(ones_tr, ones_te, Xtr_trio, Xte_trio, TRIO, y[tr]),
            )
            record(
                A_GLOBAL13,
                tsk_from_firing(ones_tr, ones_te, Xtr_13, Xte_13, feats13, y[tr]),
            )
            # A least-squares fit is invariant to an affine reparameterization of
            # its columns, so the raw-column fit IS the z-scored row
            # `bodyfat.py:313-314` reports. The log1p row is a DIFFERENT model --
            # the log is nonlinear, and `bodyfat.py:156-157` records that it
            # hurts OLS (4.71 -> 5.36 RMSE) while helping the fuzzy arm -- so it
            # runs separately, on exactly the design the Chapter 5 trio arms see,
            # minus the rule structure.
            raw_trio = df[TRIO].to_numpy(float)
            raw_13 = df[feats13].to_numpy(float)
            record(
                A_OLS,
                LinearRegression().fit(raw_trio[tr], y[tr]).predict(raw_trio[te]),
            )
            record(
                A_OLS_LOG,
                LinearRegression().fit(Z_trio[tr], y[tr]).predict(Z_trio[te]),
            )
            record(
                A_OLS13,
                LinearRegression().fit(raw_13[tr], y[tr]).predict(raw_13[te]),
            )
            record(A_MEAN, np.full(len(te), y[tr].mean()))

            # POST-HOC, and said plainly: this pair of numbers was added AFTER a
            # first three-seed pass showed the Chapter 5 trio arm and the
            # antecedent-free single-rule polynomial agreeing to within one
            # seed-sigma. It measures no new model and changes no score -- it
            # only asks whether the two arms are producing the SAME predictions,
            # which is the difference between "Chapter 5 ties the baseline" and
            # "Chapter 5 IS the baseline". Falsifier #2 in the docstring is what
            # made it worth asking; the answer is a property of the run, so it
            # belongs in the diagnostics rather than in a paragraph afterwards.
            a_pred = kept.get(A_GRADED)
            b_pred = kept.get(A_GLOBAL)
            if a_pred is not None and b_pred is not None:
                diag["gap_vs_global"].append(float(np.abs(a_pred - b_pred).mean()))
                if a_pred.std() > 0 and b_pred.std() > 0:
                    diag["corr_vs_global"].append(
                        float(np.corrcoef(a_pred, b_pred)[0, 1])
                    )

        for a in arms:
            if fold_scores[a]:
                per_seed[a].append(np.array(fold_scores[a]).mean(axis=0))
            if len(fold_scores[a]) != N_SPLITS:
                # Loud rather than fatal: an arm is scorable only on a fold where
                # a cover was selected (`selection.py:150-152` returns [] when no
                # block clears the persistence gap), and losing seed 9 to an
                # assertion at seed 3 is worse than reporting a short arm. The
                # `folds` column carries this into the table itself.
                print(
                    f"  !! seed {seed}: {a} recorded "
                    f"{len(fold_scores[a])}/{N_SPLITS} folds"
                )
        print(f"  seed {seed}: {len(fold_scores[arms[0]])} folds done")

    rows = []
    r2_mean = {}
    for a in arms:
        if not per_seed[a]:
            rows.append([a, C.NA, C.NA, C.NA, C.NA, "0"])
            print(f"  {a:42s} SKIP (no cover selected)")
            continue
        arr = np.array(per_seed[a])
        rmse_vals, r2_vals = list(arr[:, 0]), list(arr[:, 1])
        r2_mean[a] = float(np.mean(r2_vals))
        rows.append(
            [
                a,
                C.cell(r2_vals),
                C.cell(rmse_vals),
                f"{worst_fold[a]:.3f}",
                f"{len(r2_vals)}",
                f"{n_folds[a]}",
            ]
        )
        print(f"  {a:42s} R2 {C.cell(r2_vals)}   RMSE {C.cell(rmse_vals)}")

    # ---- paired differences on the shared splits -------------------------
    # The arms are paired by construction, so the per-seed difference is the
    # right statistic: marginal error bars can overlap where the paired
    # difference is consistent. Positive means the Chapter 5 arm is ahead.
    def paired(a, b):
        if not per_seed[a] or not per_seed[b]:
            return C.NA
        da = np.array(per_seed[a])[:, 1] - np.array(per_seed[b])[:, 1]
        sd = float(np.std(da, ddof=1)) if len(da) > 1 else 0.0
        return (
            f"{np.mean(da):+.4f} ± {sd:.4f} "
            f"({int((da > 0).sum())}/{len(da)} seeds Ch5 ahead)"
        )

    diag_rows = []

    def drow(label, key, fmt="{:.3f}"):
        v = diag.get(key) or []
        diag_rows.append([label, C.cell([float(x) for x in v], fmt=fmt) if v else C.NA])

    tol = f"{RANK_TOL:g}"
    drow("blocks selected k (transductive, trio)", "k_trans", "{:.1f}")
    drow("fraction of points in some block core (trio)", "core_frac")
    drow("max mu over NON-members, raw U (trio)", "mu_nonmember_max", "{:.4f}")
    drow("min mu over members, raw U (trio)", "mu_member_min", "{:.4f}")
    drow("mean max normalized membership (trio)", "maxmem_trans")
    drow("uniform reference 1/k (trio)", "unif_trans")
    drow("bottleneck-equivalence classes of 252 columns", "distinct_cols", "{:.1f}")
    drow("label disagreement, MS.assign vs argmax U (trio)", "assign_vs_argmax")
    drow("kernel reuse identity, max abs diff (trio)", "kernel_identity", "{:.3g}")
    drow("select_multiscale bands discovered (trio)", "ms_bands", "{:.1f}")
    drow("design condition number, graded (trio)", "cond_trans", "{:.3g}")
    drow("design condition number, crisp (trio)", "cond_crisp_trio", "{:.3g}")
    drow(f"design rank at tol {tol}, graded (trio)", "rank_trans", "{:.1f}")
    drow(f"design rank at tol {tol}, crisp (trio)", "rank_crisp_trio", "{:.1f}")
    drow("design columns (trio)", "ncol_trans", "{:.1f}")
    drow("design condition number, graded (all 13)", "cond13", "{:.3g}")
    drow("design condition number, crisp (all 13)", "cond_crisp13", "{:.3g}")
    drow(f"design rank at tol {tol}, graded (all 13)", "rank13", "{:.1f}")
    drow(f"design rank at tol {tol}, crisp (all 13)", "rank_crisp13", "{:.1f}")
    drow("design columns (all 13)", "ncol13", "{:.1f}")
    drow("blocks selected k (inductive, trio)", "k_ind", "{:.1f}")
    drow("train coverage (inductive, trio)", "cov_ind")
    drow("max abs D* gap, point insertion vs joint graph", "ind_vs_joint", "{:.4f}")
    drow("max D* for scale (trio)", "dstar_max", "{:.4f}")
    drow("zero-firing HELD-OUT rows, graded (trio)", "zf_te_trio", "{:.2f}")
    drow("zero-firing TRAIN rows lost from fit, graded (trio)", "zf_tr_trio", "{:.2f}")
    drow("zero-firing HELD-OUT rows, graded inductive (trio)", "zf_te_ind", "{:.2f}")
    drow("zero-firing TRAIN rows lost, graded inductive (trio)", "zf_tr_ind", "{:.2f}")
    drow("zero-firing HELD-OUT rows, graded (all 13)", "zf_te_13", "{:.2f}")
    drow("zero-firing TRAIN rows lost from fit, graded (all 13)", "zf_tr_13", "{:.2f}")
    drow("rules with no training mass, crisp (trio)", "empty_crisp_trio", "{:.2f}")
    drow("rules with no training mass, crisp (all 13)", "empty_crisp13", "{:.2f}")
    drow("under-filled-bucket warnings, Chapter 4 arms", "starved_ch4", "{:.2f}")
    drow("mean abs prediction gap, Ch5 graded vs global 1-rule (pp)", "gap_vs_global")
    drow("Pearson r, Ch5 graded vs global 1-rule", "corr_vs_global", "{:.4f}")
    for label, a, b in [
        ("paired R2, graded minus global 1-rule (trio)", A_GRADED, A_GLOBAL),
        ("paired R2, graded minus crisp assign (trio)", A_GRADED, A_ASSIGN),
        ("paired R2, graded minus crisp argmax U (trio)", A_GRADED, A_ARGMAX),
        ("paired R2, graded minus crisp assign, INDUCTIVE", A_GRADED_I, A_ASSIGN_I),
        ("paired R2, graded minus Chapter 4 (trio)", A_GRADED, A_CH4),
        ("paired R2, graded minus OLS (trio)", A_GRADED, A_OLS),
    ]:
        diag_rows.append([label, paired(a, b)])

    return rows, diag_rows, raw_rows, r2_mean


def verdict_line(r2_mean):
    """Which registered outcome fired, computed from the run's own cells.

    The docstring registers FAIL-FLOOR as "does not beat the trivial floor OR the
    single-rule global polynomial", and it is checked FIRST because it dominates:
    an arm that ties the antecedent-free control has not shown its antecedents do
    anything, whatever it does against Chapter 4. The OLS floors are named beside
    it rather than folded into it, because they were added after the first run
    (see WHAT CHANGED) and were therefore not part of the registered condition,
    even though they bound the whole table.
    """
    need = [A_GRADED, A_CH4, A_GLOBAL, A_MEAN]
    if any(a not in r2_mean for a in need):
        return "REGISTERED OUTCOME: not computable (an arm produced no cell). "
    g, c4 = r2_mean[A_GRADED], r2_mean[A_CH4]
    gl, fl = r2_mean[A_GLOBAL], r2_mean[A_MEAN]
    best_ols = max(
        (r2_mean[a] for a in (A_OLS, A_OLS_LOG, A_OLS13) if a in r2_mean), default=None
    )
    ols = "" if best_ols is None else f", and below the best OLS floor ({best_ols:.3f})"
    if g <= max(gl, fl):
        return (
            f"REGISTERED OUTCOME: **FAIL-FLOOR** — the graded Chapter 5 arm "
            f"({g:.3f}) does not beat the single-rule global polynomial "
            f"({gl:.3f}), which is this same TSK machinery with the antecedents "
            f"deleted{ols}. On the registered reading that means the antecedents "
            f"carry no usable supervised information at this venue and the arm's "
            f"score belongs to its consequent polynomial. It fires even though "
            f"the PASS band also holds (graded is within 0.05 of Chapter 4's "
            f"{c4:.3f}), and it dominates: beating Chapter 4 while tying an "
            f"antecedent-free control is not evidence for the antecedents. "
        )
    if g >= c4 - 0.05:
        return (
            f"REGISTERED OUTCOME: **PASS** — the graded Chapter 5 arm ({g:.3f}) "
            f"is within 0.05 of Chapter 4 ({c4:.3f}) AND clears the "
            f"antecedent-free single-rule control ({gl:.3f}) and the train-mean "
            f"floor ({fl:.3f}){ols}. "
        )
    return (
        f"REGISTERED OUTCOME: **FAIL** — the graded Chapter 5 arm ({g:.3f}) "
        f"loses to Chapter 4 ({c4:.3f}) by more than 0.05, while clearing the "
        f"antecedent-free single-rule control ({gl:.3f}){ols}. "
    )


# ---------------------------------------------------------------------------
# ECG5000 venue (classification). Optional: needs aeon for the DTW matrix.
# ---------------------------------------------------------------------------
E_GRADED = "Ch5 graded U -> TSK (0th)"
E_ASSIGN = "Ch5 crisp assign -> TSK (0th)"
E_GRADED_I = "Ch5 graded U inductive -> TSK (0th)"
E_ASSIGN_I = "Ch5 crisp assign inductive -> TSK (0th)"
E_GLOBAL = "global 1-rule TSK (0th)"
E_MAJ = "floor: majority class"
ECG_ARMS = [E_GRADED, E_ASSIGN, E_GRADED_I, E_ASSIGN_I, E_GLOBAL, E_MAJ]


def run_ecg5000(raw_rows):
    if os.environ.get("REPRO_C3_SKIP_ECG"):
        print("  [skip] ECG5000 disabled by REPRO_C3_SKIP_ECG")
        return None, ""
    try:
        from aeon.datasets import load_classification
        from aeon.distances import dtw_pairwise_distance
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] aeon unavailable ({exc.__class__.__name__}); ECG5000 -> N/A")
        return None, ""

    cache_dir = os.environ.get("REPRO_C3_DTW_CACHE") or os.path.join(
        C.OUTPUT_DIR, "dtw-cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "dtw_ECG5000_N5000.npz")

    X, y_raw = load_classification("ECG5000")
    if X.ndim == 3:
        X = X[:, 0, :]
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray([int(v) for v in y_raw])
    xhash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    print(f"  ECG5000: {X.shape}, classes {np.unique(y)}, X sha256[:16] {xhash}")

    # The cache is keyed on the CONTENT it was built from, not on the dataset
    # name and a row count. A matrix keyed on the name alone silently survives a
    # reordered or partially written fetch, and every membership column would
    # then be attached to the wrong label with no error anywhere. `y` is stored
    # beside the matrix and checked too.
    D = None
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["xhash"].item()) == xhash and np.array_equal(z["y"], y):
            D = z["D"]
            print(f"  DTW loaded from {cache} (hash + labels verified)")
        else:
            print(f"  !! {cache} does not match the fetched X/y; rebuilding")
    if D is None:
        t0 = time.perf_counter()
        D = dtw_pairwise_distance(X, n_jobs=-1).astype(np.float64)
        np.savez(cache, D=D, y=y, xhash=xhash)
        print(f"  DTW built in {time.perf_counter() - t0:.1f}s -> {cache}")

    Ds = im.minimax_transform_fast(D)
    blocks = S.select_coverage_cover(Ds)
    if not blocks:
        print("  [skip] no cover selected on ECG5000")
        return None, ""
    k = len(blocks)
    dist = block_distances(blocks, Ds)
    Un = MS.normalize_partition(memberships_from_distances(blocks, dist))
    lab = dist.argmin(axis=0)
    H = onehot(lab, k)
    disagree = float((lab != Un.argmax(axis=0)).mean())
    cov = S.coverage_of(blocks, Ds.shape[0])
    print(f"  cover: k={k}, coverage {cov:.3f}, assign vs argmax {disagree:.3f}")
    del Ds

    classes = np.unique(y)
    # order="0th" -- 140 raw timepoints at 1st order would be k*(1+140) free
    # parameters against 3500 training rows. At 0th order the consequent is one
    # constant per rule and `rule_consequent_values` never touches X, so an empty
    # column list and a bare index frame are legal (regression.py:1106-1107). The
    # ridge stays pinned at L2 for consistency with the bodyfat venue and is
    # INERT here whatever it is set to: `regression.py:969-970` builds the
    # penalty vector and then zeroes every `n_coeffs_per_rule`-th entry, which at
    # 0th order (one coefficient per rule) is all of them.
    empty_cols: list = []

    def ovr(F_tr, F_te, y_tr):
        Xtr = pd.DataFrame(index=range(F_tr.shape[0]))
        Xte = pd.DataFrame(index=range(F_te.shape[0]))
        scores = np.zeros((F_te.shape[0], len(classes)))
        for j, c in enumerate(classes):
            scores[:, j] = tsk_from_firing(
                F_tr, F_te, Xtr, Xte, empty_cols, (y_tr == c).astype(float), order="0th"
            )
        return classes[scores.argmax(axis=1)]

    acc = {a: [] for a in ECG_ARMS}
    mf1 = {a: [] for a in ECG_ARMS}
    k_ind = []
    for seed in C.SEEDS:
        t0 = time.perf_counter()
        tr, te = next(
            StratifiedShuffleSplit(1, test_size=0.3, random_state=seed).split(
                np.zeros(len(y)), y
            )
        )
        maj = classes[np.argmax([(y[tr] == c).sum() for c in classes])]
        preds = {
            E_GRADED: ovr(Un[:, tr].T, Un[:, te].T, y[tr]),
            E_ASSIGN: ovr(H[:, tr].T, H[:, te].T, y[tr]),
            E_GLOBAL: ovr(np.ones((len(tr), 1)), np.ones((len(te), 1)), y[tr]),
            E_MAJ: np.full(len(te), maj),
        }
        # Inductive: D* and the cover from the training block of the DTW matrix
        # only, held-out series inserted by the same exact bottleneck extension
        # as bodyfat. Pairwise DTW distances do not depend on the split, so
        # slicing the cached matrix is not leakage -- nothing on the training
        # side is computed from a held-out row.
        bi, _Ui, Uni, disti = ch5_inductive_from_D(D[np.ix_(tr, tr)], D[np.ix_(te, tr)])
        if bi is not None:
            k_ind.append(len(bi))
            n_tr = len(tr)
            Hi = onehot(disti.argmin(axis=0), len(bi))
            preds[E_GRADED_I] = ovr(Uni[:, :n_tr].T, Uni[:, n_tr:].T, y[tr])
            preds[E_ASSIGN_I] = ovr(Hi[:, :n_tr].T, Hi[:, n_tr:].T, y[tr])
        for a, p in preds.items():
            a_ = accuracy_score(y[te], p)
            f_ = f1_score(y[te], p, average="macro", zero_division=0)
            acc[a].append(a_)
            mf1[a].append(f_)
            raw_rows.append(["ECG5000", a, seed, 0, f"{a_:.6f}", f"{f_:.6f}"])
        print(f"  seed {seed} done in {time.perf_counter() - t0:.1f}s")

    rows = []
    for a in ECG_ARMS:
        rows.append([a, C.cell(acc[a]), C.cell(mf1[a]), f"{len(acc[a])}"])
        print(f"  {a:34s} acc {C.cell(acc[a])}  macro-F1 {C.cell(mf1[a])}")
    facts = (
        f"COVER: the transductive cover selects k={k} blocks whose cores hold "
        f"{cov:.3f} of the 5000 points, and `MS.assign` disagrees with "
        f"`argmax(U)` on {disagree:.3f} of them; the inductive cover selects "
        f"{np.mean(k_ind):.1f} blocks from the 3500 training rows. "
        if k_ind
        else f"COVER: k={k}, block-core coverage {cov:.3f}. "
    )
    return rows, facts


def main() -> int:
    print("C3 -- fitting a supervised model on Chapter 5's block memberships")
    print(f"  seeds = {C.SEEDS}")
    raw_rows = []

    print("\n[bodyfat]")
    bf = run_bodyfat()
    if bf is None:
        print("  bodyfat unavailable; nothing to emit.")
        return 1
    rows, diag_rows, bf_raw, r2_mean = bf
    raw_rows.extend(bf_raw)
    verdict = verdict_line(r2_mean)
    print(f"\n  {verdict}\n")

    C.emit(
        "ch5_end_to_end",
        title=(
            "C3 — supervised held-out R² from Chapter 5 `block_membership` "
            "antecedents, beside the Chapter 4 arm and trivial floors, on "
            "identical bodyfat splits"
        ),
        header=[
            "arm",
            "held-out R²",
            "held-out RMSE",
            "worst fold R²",
            "seeds",
            "folds",
        ],
        rows=rows,
        note=(
            verdict + "Ten seeds x 5-fold `KFold(shuffle=True, "
            "random_state=seed)` on all 252 bodyfat rows; every arm consumes the "
            "identical (train, test) index pairs from one split object per seed, "
            "and the `folds` column reports how many folds each arm actually "
            "recorded (an arm is scorable only where a cover was selected). R² is "
            "the mean over folds of the per-fold `r2_score`, then mean ± sample "
            "std over seeds — the definition "
            "`FuzzySystemsExperiments/bodyfat.py:242-259` uses, so the Chapter 4 "
            "cell is comparable with its reported 0.647. `Density` is dropped by "
            "every arm: Siri's equation reproduces `BodyFat` from it at R² 0.977, "
            "so it is the target in another coordinate. Feature scalers are "
            "fitted on TRAIN ROWS ONLY. `y` is read on train rows only, in every "
            "arm. The Chapter 5 antecedents — `MS.block_membership` stacked and "
            "Ruspini-normalized, the matrix `table_5_4_ch5_g1_scaling.py:91-92` "
            "builds and then uses only for its own tautological partition-of-"
            "unity residual — are fed as a precomputed firing matrix to "
            "`solve_tsk_consequents_from_firing` and scored held out with "
            "`apply_tsk_consequents` (NOT `predict_tsk`, which recomputes firing "
            "from a GaussianMixtureModel and cannot accept a precomputed U). This "
            "is a SUPERVISED held-out score: no ARI, NMI or silhouette is "
            "reported anywhere, because a clustering score is the proxy C3 exists "
            "to replace. STRUCTURE OF U: a non-member of a single-linkage block "
            "can only reach it across the edge that dissolves the block, so "
            "d_B(x) ≥ death_B, and a kernel with half-max at death caps every "
            "non-member at μ ≤ 0.5 while every member reads exactly 1 — measured "
            "in the diagnostics, and true on any dataset rather than a property "
            "of bodyfat. TRANSDUCTION: the default Chapter 5 arms build D* and "
            "select the cover over all 252 rows' features (never their targets), "
            "which is Chapter 5's own standing convention; the `inductive` arms "
            "remove even that, building D* and the cover on train rows only and "
            "inserting each held-out point by exact single-point bottleneck "
            "extension. Both a graded and a crisp inductive arm run, so the "
            "graded-vs-crisp question is answered under the leak-free protocol as "
            "well as the conventional one — they disagree in sign, and the "
            "inductive pair is the one to read. The registered prior was that "
            "Chapter 5 LOSES — `phase6_soft_validation.py` found the normalized "
            "minimax membership is a constant step per cluster with no boundary "
            "resolution (Brier 0.136/0.200/0.208/0.122 fuzzy vs "
            "0.096/0.042/0.016/0.000 hard, MF_PROGRESS_LOG.md:259-264). The arms that "
            "exist to refute a comfortable reading: two crisp controls "
            "(`MS.assign`, the argmin-over-minimax-distance labels every other "
            "Chapter 5 path uses, and `argmax(U)`, the true hardening of this U — "
            "they are NOT the same partition and their disagreement is emitted), "
            "the single-rule global polynomial (identical features, order and "
            "ridge with the antecedents deleted — a Chapter 5 arm that only ties "
            "it is scoring with its consequent polynomial, not with its "
            "antecedents), and the floors: train-mean (R² = 0 by construction) "
            "and ordinary least squares on the same columns, which "
            "`bodyfat.py:312-314` already scores and `bodyfat_report.md:41` "
            "already calls the best model on this dataset. OLS is invariant to "
            "affine scaling, so its raw-column rows are the z-scored rows that "
            "report gives; the log1p-scaled OLS row is a different model and runs "
            "separately, on exactly the design the Chapter 5 trio arms see. NO "
            "TUNING HERE, ONE PIECE INHERITED: every TSK arm is pinned to the "
            "Chapter 4 arm's own consequent configuration (order='2nd', "
            "l2_reg=1e-2, pin_extremes=False, firing_exponent=1.0), so the only "
            "difference between the Chapter 4 and Chapter 5 rows is where the "
            "antecedents came from — but `bodyfat.py:159-160` selected that "
            "configuration as the best of 64 scored on these same partitions, so "
            "it flatters the CHAPTER 4 arm, not the Chapter 5 arms, and a "
            "negative Chapter 5 result is robust to it. The all-13 rows "
            "transplant that same ridge onto a 13-column z-scored design it was "
            "never re-swept for, and `l2_reg` is not scale-free "
            "(`bodyfat_report.md:51`); read them as an artifact of an out-of-"
            "scale ridge on an ill-posed design, plus a handful of held-out rows "
            "whose graded firing underflows to zero and are therefore predicted "
            "as exactly 0.0 against a 0–45 target (both counted in the "
            "diagnostics) — not as a modelling result. ADDED AFTER THE FIRST RUN, "
            "in response to an adversarial review and itemized in the script's "
            "docstring: the three OLS floors, the inductive crisp arm, the "
            "`argmax(U)` control, the verdict line above, and most of the "
            "diagnostics. None of them is a hyperparameter and none changes an "
            "existing cell — every pre-existing arm reproduces its first-run "
            "value bit-for-bit — but they are not part of the pre-registration "
            "and are marked so. They moved the reading AGAINST Chapter 5 (the "
            "OLS floors bound the whole table; the verdict line names FAIL-FLOOR "
            "where the first pass reported a tie), with one exception in "
            "Chapter 5's favour: the inductive crisp arm reverses the sign of the "
            "graded-versus-crisp comparison. Diagnostics in the sibling "
            "`ch5_end_to_end_diagnostics`; per-seed, per-fold values in "
            "`ch5_end_to_end_raw.csv`."
        ),
    )

    C.emit(
        "ch5_end_to_end_diagnostics",
        title=(
            "C3 diagnostics — why the Chapter 5 antecedent arm scores what it "
            "scores on bodyfat"
        ),
        header=["quantity", "mean ± std over folds"],
        rows=diag_rows,
        note=(
            "Aggregated over every (seed, fold) pair of the run that produced "
            "`ch5_end_to_end`; the paired rows aggregate over seeds. THE SHAPE OF "
            "U: `max mu over NON-members` is 0.5 and `min mu over members` is 1.0 "
            "by construction, not by luck — a non-member of a single-linkage "
            "block reaches it only across the edge that dissolves the block, so "
            "its minimax distance is at least the block's death height, and the "
            "kernel puts half-max exactly there. U therefore has nothing in "
            "(0.5, 1), and with only a small `fraction of points in some block "
            "core` the normalized partition sits near `uniform reference 1/k`: "
            "every rule fires almost equally for every point, each rule's design "
            "block is a near-copy of every other's, and the TSK collapses toward "
            "one global polynomial. `bottleneck-equivalence classes` counts "
            "distinct membership columns — points sharing an MST bottleneck edge "
            "share a column — and is NOT a measure of boundary resolution; an "
            "earlier pass of this file over-read it as one. CONDITIONING, WITH "
            "ITS OWN CONTROL: every conditioning row is reported for the graded "
            "AND the crisp arm, because neither rank nor condition number "
            "separates the arm that blew up from the arm that did not. On all 13 "
            "features the two arms have the same column count, and the CRISP arm "
            "is both the more rank-deficient and the worse-conditioned of the "
            "two — and it is the one that scores ~19 R² points better. What "
            "distinguishes them is where the collinearity lives: a crisp arm's "
            "rule blocks have disjoint row support and so cannot be collinear "
            "with each other at all, while near-uniform graded firing makes all k "
            "blocks nearly proportional on the SAME rows, which is the "
            "ill-posedness l2_reg=1e-2 does not control. Two caveats on reading "
            "these rows: σ_max/σ_min is not comparable across arms of different "
            "numerical rank, and a design whose condition number is below 1/tol "
            "is full rank at that tolerance by arithmetic — so a rank row says "
            "something about the data only where the condition number beside it "
            "exceeds 1e8, which here is every arm except the graded trio one. "
            "ZERO FIRING: rows whose memberships underflow "
            "ZERO_FIRING_THRESHOLD = 1e-6 are left all-zero by "
            "`_normalize_firing_strengths`; a HELD-OUT such row is predicted as "
            "exactly 0.0 (on a 0–45 target one row can dominate a fold) and a "
            "TRAIN such row contributes nothing to the ridge normal equations, so "
            "the graded arms are fitted on slightly fewer rows than the crisp "
            "arms beside them. Both sides are counted for every graded arm, with "
            "the crisp arms' analogue (a rule with no training mass, whose "
            "min-norm consequent also predicts 0.0) beside them. The two "
            "prediction-agreement rows are POST-HOC, added after a first "
            "three-seed pass showed the Chapter 5 trio arm and the "
            "antecedent-free single-rule polynomial agreeing to within one "
            "seed-sigma: they compare the two arms' PREDICTIONS rather than their "
            "scores, which is the difference between Chapter 5 tying the baseline "
            "and Chapter 5 being the baseline. They measure no new model and "
            "change no score. The paired rows are the right statistic for arms "
            "that share their splits by construction; positive means the Chapter "
            "5 arm is ahead."
        ),
    )

    print("\n[ECG5000]")
    ecg, ecg_facts = run_ecg5000(raw_rows)
    if ecg is None:
        ecg = [[a, C.NA, C.NA, C.NA] for a in ECG_ARMS]
    C.emit(
        "ch5_end_to_end_ecg5000",
        title=(
            "C3, second venue — supervised held-out accuracy from Chapter 5 "
            "`block_membership` antecedents on ECG5000 (DTW)"
        ),
        header=["arm", "held-out accuracy", "held-out macro-F1", "seeds"],
        rows=ecg,
        note=(
            ecg_facts + "Ten seeds of `StratifiedShuffleSplit(1, test_size=0.3)` "
            "on the 5000-row ECG5000 train+test concatenation; every arm consumes "
            "the identical split. Antecedents come from the DTW dissimilarity and "
            "the low-memory minimax transform that "
            "`reproduce/tables/table_3_7_g2_downstream.py` already builds — that "
            "table is untouched, since it is the cited evidence for Chapter 3's "
            "G2 claim. The DTW matrix is cached under a key that includes a "
            "SHA-256 of the fetched series, and the labels are cached beside it, "
            "so a stale or reordered cache is rebuilt rather than silently paired "
            "with the wrong labels. Five classes, so the regression solver runs "
            "one-vs-rest on indicator targets and the prediction is the argmax; "
            '`order="0th"` because 140 raw timepoints at 1st order would be k·141 '
            "free parameters against 3500 training rows. At 0th order `l2_reg` is "
            "INERT whatever it is set to — the penalty vector is zeroed at every "
            "`n_coeffs_per_rule`-th entry and there is one coefficient per rule "
            "(`regression.py:969-970`) — so the pinned l2_reg=1e-2 neither helps "
            "nor hurts here. Class balance is 2919/1767/194/96/24, so accuracy "
            "alone is nearly uninformative: the majority-class floor is 0.584 and "
            "macro-F1 swings hard on the 24-example class. Falsifier #2 is "
            "present at this venue and degenerates: at 0th order with one rule the "
            "antecedent-free control fits one constant per class — the class "
            "prior — so its argmax is the majority class, and `global 1-rule TSK` "
            "is expected to equal `floor: majority class` exactly. Both "
            "transductive arms (D* and the cover over all 5000 series, only the "
            "LABELS held out) and inductive arms (D*, cover and memberships from "
            "the 3500 training series only, each held-out series inserted by "
            "exact single-point bottleneck extension) are reported. There is no "
            "Chapter 4 arm on this dataset to sit beside, so the crisp "
            "`MS.assign` control is the within-chapter comparison. `N/A` means "
            "`aeon` was not available in the run environment; add `--with aeon`."
        ),
    )

    C.write_csv(
        os.path.join(C.OUTPUT_DIR, "ch5_end_to_end_raw.csv"),
        ["venue", "arm", "seed", "fold", "rmse_or_accuracy", "r2_or_macro_f1"],
        raw_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
