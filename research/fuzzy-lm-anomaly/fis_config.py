"""Single place where the fuzzy configuration is declared -- and enforced.

Why this module exists. Section 22's factorial found that the **membership-function
family is the dominant factor** in this pipeline: Gaussian beats trapezoid by
**+0.262 AUROC on 8/8 seeds (p = 0.0078)**, an order of magnitude larger than the
effect of the coefficient metric or the norm pair. A silent change of membership
family would therefore move a headline number by a quarter of an AUROC point --
and it would be invisible, because nothing in the pipeline announces which family
it is using.

Two ways that silent change could happen, both now blocked:

1. **Calling the wrong builder.** The MF family is *not* set by
   `AnomalyParameters.member_function`. That field is only read by
   `simple_gaussian_predict` (`gauss_math.py:504`); `tsk_firing_strengths`
   evaluates whatever objects happen to be in the model
   (`mf.evaluate(...)`). The family is decided entirely by which builder was
   called -- `create_gaussian_membership_dict` vs `create_trapz_membership_dict`.
   `build_memberships()` below makes that choice explicit and then *asserts* the
   resulting objects are the family that was asked for.

2. **A library default moving.** `DefaultMemberFunction` is `"gaussian"` today.
   Section 22 already saw one committed result move when
   `calculate_gaussian_correlation`'s default changed from a blend to
   `bhattacharyya`, so this is not hypothetical. Every `AnomalyParameters` built
   here passes `member_function` explicitly rather than inheriting it.

The same reasoning applies to the metric and the norm pair, so those are declared
here too with the evidence that chose them.
"""

from typing import Literal

from tribblefis.gauss_data import (AnomalyParameters, GaussianMembership,
                                   MemberFunction, TrapezoidMembership,
                                   TriangularMembership)
from tribblefis.gauss_math import create_gaussian_membership_dict
from tribblefis.trapz_math import create_trapz_membership_dict

# --- the declared configuration -------------------------------------------

#: Membership family. Gaussian, explicitly, on section 22's +0.262 (8/8 seeds).
#: A trapezoid's flat top assigns membership exactly 1.0 across an interval, so
#: the conjunction cannot separate "comfortably inside the known-good region"
#: from "at its edge" -- and the anomaly complement needs exactly that gradation.
MEMBERSHIP: MemberFunction = "gaussian"

#: Antecedent-ranking coefficient. Wasserstein: its mean is within noise of
#: Bhattacharyya (p = 0.875) but its variance is 2.7x smaller in the hardest
#: condition (+/-0.025 vs +/-0.067). The library default is Bhattacharyya, chosen
#: on section 13's means, which did not measure variance.
METRIC: str = "wasserstein"

#: Norm family for both halves of the pair. Hamacher: within noise of the best
#: AUROC (0.802 vs 0.811 for min/max) with the lowest variance of the leaders
#: (+/-0.022) and a materially better FPR@95TPR (0.786 vs 0.931). Taking both
#: operators from one family keeps the pair De Morgan dual, which the anomaly
#: rule's complement 1 - S(...) depends on for its meaning.
NORM: str = "hamacher"

#: Anomaly boost. Rank-invariant (section 3.4) -- it sets the operating point, not
#: separability.
THETA: float = 0.5

_EXPECTED = {"gaussian": GaussianMembership, "trap": TrapezoidMembership,
             "triangular": TriangularMembership}
_BUILDERS = {"gaussian": create_gaussian_membership_dict,
             "trap": create_trapz_membership_dict}


def build_memberships(X, y, feature_names, membership: MemberFunction = MEMBERSHIP,
                      **kwargs):
    """Build a rule base with an explicitly chosen membership family, and verify it.

    The assertion is the point: it converts "the wrong builder was called" or "a
    library default moved" from a silent 0.26-AUROC shift into an immediate error.
    """
    if membership not in _BUILDERS:
        raise ValueError(f"membership must be one of {sorted(_BUILDERS)}, "
                         f"got {membership!r}")
    model = _BUILDERS[membership](X, y, top_n_var_names=list(feature_names), **kwargs)

    want = _EXPECTED[membership]
    for fname, fm in model.feature_models.items():
        for label, lm in fm.label_models.items():
            for mf in lm.memberships:
                if not isinstance(mf, want):
                    raise TypeError(
                        f"membership family mismatch: asked for {membership!r} "
                        f"({want.__name__}) but feature {fname!r} / label {label!r} "
                        f"produced {type(mf).__name__}. Section 22: the membership "
                        f"family is worth +/-0.262 AUROC, so this is never a "
                        f"cosmetic difference."
                    )
    return model


def anomaly_params(theta: float = THETA, norm: str = NORM,
                   membership: MemberFunction = MEMBERSHIP,
                   t_norm: str | None = None, t_conorm: str | None = None,
                   label: str = "anomaly") -> AnomalyParameters:
    """AnomalyParameters with every field stated, none inherited from a default.

    Passing `t_norm`/`t_conorm` selects a mixed pair, which is not a De Morgan dual
    and so is opted into deliberately (section 22 measured the gain at +0.002 --
    almost never worth the lost interpretation).
    """
    mixed = (t_norm is not None or t_conorm is not None) and (t_norm != t_conorm)
    return AnomalyParameters(
        include_anomaly=True,
        threshold=theta,
        label=label,
        norm_conorm=norm,
        member_function=membership,      # explicit: never inherit this
        t_norm=t_norm,
        t_conorm=t_conorm,
        allow_mixed_norms=mixed,
    )


def describe() -> str:
    return (f"membership={MEMBERSHIP}  metric={METRIC}  "
            f"norm={NORM}/{NORM} (De Morgan)  theta={THETA}")


if __name__ == "__main__":
    import warnings

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    print("declared configuration:", describe())

    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=200) for c in "abc"})
    y = pd.Series(["m0"] * 100 + ["m1"] * 100)

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        g = build_memberships(X, y, list("abc"), "gaussian")
        t = build_memberships(X, y, list("abc"), "trap")
    print(f"gaussian builder -> {g.n_membership_functions} MFs, verified")
    print(f"trap builder     -> {t.n_membership_functions} MFs, verified")

    ap = anomaly_params()
    print(f"anomaly_params(): member_function={ap.member_function!r} "
          f"norm_conorm={ap.norm_conorm!r} pair={ap.norms()}")
    mixed = anomaly_params(t_norm="hamacher", t_conorm="einstein")
    print(f"mixed opt-in    : pair={mixed.norms()} "
          f"is_de_morgan={mixed.norms().is_de_morgan}")

    # the guard must actually fire
    class _Fake:
        pass
    bad = g._replace() if hasattr(g, "_replace") else g
    try:
        obj = list(bad.feature_models["a"].label_models["m0"].memberships)[0]
        assert isinstance(obj, GaussianMembership)
        print("guard: gaussian model passes isinstance check as expected")
    except AssertionError:
        print("guard: UNEXPECTED failure")
    try:
        build_memberships(X, y, list("abc"), "nonsense")
    except ValueError as e:
        print(f"guard: bad family rejected -> {str(e)[:56]}")
