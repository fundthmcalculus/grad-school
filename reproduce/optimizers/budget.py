"""One evaluation budget, enforced identically for every optimizer.

The question this study asks is "how much can each method improve on the
heuristic start *in finite time*", and that question is only meaningful if
"finite" means the same thing for every arm. It cannot be left to each
optimizer's own accounting: SciPy's differential evolution spends
``maxiter x popsize x D`` evaluations, `L-BFGS-B` spends however many the line
search needs, and the `optimizers` package spends ``population_size x
num_generations`` plus whatever its local-gradient option adds. Configuring
those to match by hand would be arithmetic that silently goes stale the first
time a default changes.

So the budget lives here, in a wrapper around the objective, and it is a hard
stop: past the cap every call raises `BudgetExhausted`, the arm unwinds, and the
driver keeps the best point seen. Every arm therefore gets exactly ``max_evals``
objective evaluations and no arm can be accused of having been given more.

The wrapper also records the trace -- (evaluation index, wall-clock second,
best-so-far value) at every improvement -- which is what the convergence figure
plots and what makes "it got there in the first 200 evaluations and then sat
still" visible rather than inferred from a final number.

**Threads, not processes, and one job.** The counter has to be a true global
count, and under joblib's process backend each worker would hold its own copy.
Every arm in this study runs single-threaded for that reason, which also removes
core count from the comparison. It is a real constraint on what the study can
say: these are sequential-evaluation results, and an optimizer that parallelises
well is not being given credit for it here.
"""

from __future__ import annotations

import time

import numpy as np


class BudgetExhausted(RuntimeError):
    """Raised by the objective once the evaluation cap is reached."""


class BudgetedObjective:
    """Wrap a scalar objective with a hard evaluation cap and a trace.

    Usage:

        obj = BudgetedObjective(fitness, max_evals=2000)
        try:
            arm(obj, ...)
        except BudgetExhausted:
            pass
        best_x, best_f = obj.best_x, obj.best_f
    """

    def __init__(self, fn, max_evals, x0=None, checkpoints=()):
        self._fn = fn
        self.max_evals = int(max_evals)
        self.n_evals = 0
        self.best_x = None
        self.best_f = np.inf
        self.trace = []  # (eval_index, seconds, best_so_far)
        self._t0 = None
        # Checkpoints answer the question the study is actually asked: not "what
        # does this optimizer find", but "how much does it find *in finite
        # time*". The trace already gives the objective at any budget, but the
        # objective is not the quantity that matters -- held-out R^2 is, and
        # scoring that needs the parameter vector as it stood at each budget.
        # Keeping a copy at each checkpoint turns one run into the whole
        # budget curve, instead of one run per budget.
        self.checkpoints = sorted(int(c) for c in checkpoints)
        self.snapshots = {}  # eval budget -> best_x as of that budget
        self._next_cp = 0
        # The hot start is scored outside the budget, deliberately. Every arm
        # begins from it, so charging one evaluation for it to some arms and not
        # others (SciPy's L-BFGS-B evaluates x0 itself; a population method may
        # not) would make the budgets unequal in the one place it is easiest to
        # miss.
        self.x0 = None if x0 is None else np.asarray(x0, dtype=float)
        self.f0 = None

    def start(self):
        """Begin the clock and score the hot start off-budget."""
        self._t0 = time.perf_counter()
        if self.x0 is not None:
            self.f0 = float(self._fn(self.x0))
            self.best_x, self.best_f = self.x0.copy(), self.f0
            self.trace.append((0, 0.0, self.f0))
        return self

    @property
    def seconds(self):
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    @property
    def exhausted(self):
        return self.n_evals >= self.max_evals

    def __call__(self, x, *args, **kwargs):
        if self.exhausted:
            raise BudgetExhausted(f"{self.max_evals} evaluations spent")
        self.n_evals += 1
        value = float(self._fn(np.asarray(x, dtype=float)))
        if not np.isfinite(value):
            value = 1e6
        if value < self.best_f:
            self.best_f = value
            self.best_x = np.asarray(x, dtype=float).copy()
            self.trace.append((self.n_evals, self.seconds, value))
        # Snapshot after the update, so a checkpoint reflects everything the
        # budget bought including the evaluation that reached it.
        while (
            self._next_cp < len(self.checkpoints)
            and self.n_evals >= self.checkpoints[self._next_cp]
        ):
            cp = self.checkpoints[self._next_cp]
            self.snapshots[cp] = (
                None if self.best_x is None else self.best_x.copy(),
                self.best_f,
                self.seconds,
            )
            self._next_cp += 1
        return value

    def finalize(self):
        """Fill any checkpoint the run never reached with the final state.

        An arm that stops early -- SciPy's L-BFGS-B converges and returns long
        before the cap on some seeds -- would otherwise have missing cells at the
        larger budgets, which reads as "no data" when the truth is "it had
        already finished and nothing changed after this point".
        """
        for cp in self.checkpoints:
            if cp not in self.snapshots:
                self.snapshots[cp] = (
                    None if self.best_x is None else self.best_x.copy(),
                    self.best_f,
                    self.seconds,
                )
        return self

    def improvement(self):
        """Fractional drop in the objective from the hot start, or None."""
        if self.f0 is None or self.f0 <= 0:
            return None
        return (self.f0 - self.best_f) / self.f0

    def beat_start(self, tol=1e-12):
        """Did this arm actually improve on the heuristic start at all?

        Reported per arm and per seed rather than averaged away, because §6.3.5's
        claim is precisely that there is little left to find once the model has
        been built from the data's own structure -- so "how often did it find
        anything" is the headline, not a footnote.
        """
        return self.f0 is not None and self.best_f < self.f0 - tol
