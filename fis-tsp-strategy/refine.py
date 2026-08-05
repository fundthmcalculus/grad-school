"""Derivative-free stepwise local refinement, to polish what the GA hands over.

A genetic algorithm is a global search: it finds the right basin and then wanders inside it,
because its moves are recombination and random mutation, neither of which is a descent step.
The evidence for that here is direct — the GA's own final vector consistently scored worse on
validation than the best vector it had *passed through* (frontier ratio 1.338 against 1.057
on the last run), which is what a search that does not descend looks like.

So this is the other half: a **compass search** (Hooke–Jeeves pattern search without the
pattern move), which for each coordinate tries `+step` and `-step`, keeps the move only if it
improves, and halves the step when a whole sweep finds nothing. It is the deterministic
sibling of the `optimizers` library's own `local_perturb_optim`, which does the same
one-variable-at-a-time thing with a random perturbation and a single pass; both are available
here, since the library's version is what the GA can call inline via `local_grad_optim`.

Why compass search and not a gradient method: the objective is a solver run. It is piecewise
constant in every parameter — nudge a consequent and either some city's rounded breadth
changes or nothing at all happens — so a finite-difference gradient is either zero or an
artefact of the step size, and derivative-based methods have nothing to hold onto. A method
that only ever compares objective values is not making the best of a bad situation here; it
is the correct choice.

Two properties matter for using it honestly:

* **It only accepts improvements**, so it can never return something worse than it was given.
  With a fixed evaluation budget the result is monotone in the budget, unlike the GA.
* **It is deterministic**, so a refinement run is reproducible and the polish can be
  attributed separately from the global search that preceded it.

Because it descends on whatever objective it is given, it overfits whatever set that objective
measures — faster than the GA, not slower. It is therefore run on the *training* objective and
selected on validation like everything else.

Run:  imported by tune_opt.py; `python refine.py --demo` exercises it on a toy objective.
"""

from __future__ import annotations

import argparse

import numpy as np


def compass_refine(
    fcn,
    theta,
    lo=0.0,
    hi=1.0,
    step0=0.12,
    step_min=0.004,
    max_evals=4000,
    shrink=0.5,
    order=None,
    verbose=False,
):
    """Minimise ``fcn`` from ``theta`` by coordinate-wise +/- steps with a shrinking step.

    Returns (best theta, best value, evaluations used, sweeps completed).

    ``order`` optionally fixes the coordinate visiting order — pass an array to make the
    sweep deterministic under a chosen priority (for instance, consequents before membership
    parameters), which matters because a coordinate visited early gets to move while the step
    is still large.
    """
    theta = np.clip(np.asarray(theta, dtype=np.float64).copy(), lo, hi)
    best = float(fcn(theta))
    evals = 1
    d = theta.shape[0]
    idx = np.arange(d) if order is None else np.asarray(order, dtype=int)
    step = float(step0)
    sweeps = 0

    while step >= step_min and evals < max_evals:
        improved = False
        sweeps += 1
        for i in idx:
            if evals >= max_evals:
                break
            for sign in (1.0, -1.0):
                cand = theta[i] + sign * step
                if cand < lo or cand > hi:
                    continue
                old = theta[i]
                theta[i] = cand
                val = float(fcn(theta))
                evals += 1
                if val < best - 1e-12:
                    best = val
                    improved = True
                    break  # keep the move, go to the next coordinate
                theta[i] = old
        if verbose:
            print(f"    sweep {sweeps:3d} step={step:.4f} best={best:.6f} evals={evals}")
        if not improved:
            step *= shrink  # a whole sweep found nothing: refine the resolution
    return theta, best, evals, sweeps


def library_perturb_refine(fcn, theta, variables, passes=6, start=0.12, shrink=0.6):
    """The same idea using the `optimizers` library's own stepwise primitive.

    ``local_perturb_optim`` walks one variable at a time and tries a random perturbation in
    each direction, keeping improvements — the library's derivative-free stepwise refinement.
    It is a single pass with one random step per coordinate, so it is wrapped in a loop with a
    shrinking perturbation to give it the convergence the compass search has by construction.
    Kept for comparison against :func:`compass_refine`, whose deterministic +/- step is
    reproducible where this is not.
    """
    from optimizers.continuous.local import local_perturb_optim

    theta = np.clip(np.asarray(theta, dtype=np.float64).copy(), 0.0, 1.0)
    best = float(fcn(theta))
    p = float(start)
    for _ in range(passes):
        cand, val = local_perturb_optim(fcn, theta.copy(), variables, max_perturbation=p)
        cand = np.clip(cand, 0.0, 1.0)
        v = float(fcn(cand))
        if v < best:
            best, theta = v, cand
        p *= shrink
    return theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if not args.demo:
        print(__doc__.strip().splitlines()[0])
        return

    # A deliberately awkward toy: piecewise-constant in every coordinate, exactly like the
    # real objective, so a gradient method would see nothing to follow.
    target = np.array([0.31, 0.77, 0.05, 0.62, 0.44])

    def f(x):
        return float(np.sum(np.abs(np.round(np.asarray(x) * 20) / 20 - target)))

    x0 = np.full(5, 0.5)
    print(f"start   {f(x0):.4f}  {np.round(x0, 3)}")
    x, v, evals, sweeps = compass_refine(f, x0, verbose=True)
    print(f"refined {v:.4f}  {np.round(x, 3)}  ({evals} evals, {sweeps} sweeps)")
    print(f"target        {np.round(target, 3)}")


if __name__ == "__main__":
    main()
