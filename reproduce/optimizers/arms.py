"""One adapter per optimizer, all given the same start, box, objective and budget.

An arm's only job is to spend its evaluation budget. It does not decide when to
stop (the budget raises), it does not score the result (the driver does), and it
does not get to keep a point worse than the hot start (the driver enforces that
uniformly). What differs between arms is purely the search.

## The hot start, and which arms can actually take one

This is the finding that shaped the module, so it is documented here rather than
in a commit message.

`InputContinuousVariable` takes an `initial_value`, and reading the package it
looks like the warm-start hook. It is only honoured by two of the five
optimizers: `GradientDescentOptimizer` reads `variable.initial_value` directly,
and `MultiTypeOptimizer` seeds its defaults from it. **GA, PSO and ACO ignore
it** -- they populate the initial solution deck from
`variable.initial_random_value()`, so a hot start passed that way is silently
discarded and the arm begins from a uniform sample of the box.

There is a second seam that does work for all of them.
`SolutionDeck.initialize_solution_deck` preserves the first
`int(archive_size * preserve_percent)` rows of the archive, and `solve()` takes
`preserve_percent` as an argument. Writing `x0` into row 0 and solving with
`preserve_percent = 1 / archive_size` therefore injects the incumbent into the
initial population of any deck-based optimizer, and it is evaluated with the
rest. Verified against a quadratic whose optimum was injected: the run returns it.

So every arm here is warm-started **three** ways, and each is recorded so the
contribution can be separated later:

  1. `initial_value` on each variable  -- honoured by GD and MultiType;
  2. deck injection at row 0           -- honoured by GA, PSO, ACO;
  3. the trust region on the bounds    -- honoured by everything, since all of
     them sample inside the box (see `problem.build(radius=...)`).

The SciPy arms take `x0` directly and need none of it.
"""

from __future__ import annotations

import numpy as np

from budget import BudgetExhausted


# --------------------------------------------------------------------------- #
# SciPy incumbents -- what the shipped refinement uses today
# --------------------------------------------------------------------------- #
def scipy_lbfgsb(obj, problem, seed, **hp):
    """`refine_antecedents_local`'s optimizer: L-BFGS-B from the hot start."""
    from scipy.optimize import minimize
    minimize(obj, problem.x0, method="L-BFGS-B", bounds=problem.bounds,
             options={"maxiter": hp.get("maxiter", 10_000),
                      "maxfun": obj.max_evals,
                      "ftol": hp.get("ftol", 1e-10),
                      "eps": hp.get("eps", 1e-4)})


def scipy_de(obj, problem, seed, **hp):
    """`refine_antecedents_de`'s optimizer: differential evolution, x0 seeded."""
    from scipy.optimize import differential_evolution
    differential_evolution(
        obj, problem.bounds, x0=problem.x0, seed=seed,
        maxiter=hp.get("maxiter", 10_000), popsize=hp.get("popsize", 8),
        tol=hp.get("tol", 1e-6), mutation=hp.get("mutation", (0.5, 1.0)),
        recombination=hp.get("recombination", 0.7),
        init=hp.get("init", "sobol"), updating="immediate",
        polish=hp.get("polish", False))


def scipy_powell(obj, problem, seed, **hp):
    """A derivative-free local method, as a control on L-BFGS-B's finite differences.

    L-BFGS-B on this objective is estimating ~100 partial derivatives by finite
    difference, at one evaluation each, so a single gradient step costs what a
    small population costs. Powell is here to separate "local search wins" from
    "gradient estimation is affordable".
    """
    from scipy.optimize import minimize
    minimize(obj, problem.x0, method="Powell", bounds=problem.bounds,
             options={"maxiter": hp.get("maxiter", 10_000),
                      "maxfev": obj.max_evals, "xtol": 1e-8, "ftol": 1e-10})


# --------------------------------------------------------------------------- #
# the `optimizers` package
# --------------------------------------------------------------------------- #
def _variables(problem, perturbation=0.0):
    """Bounded continuous variables, each carrying the hot start as its initial.

    `perturbation=0.0` matters: `InputContinuousVariable` offsets a supplied
    initial value by `perturbation * domain` deterministically, so the default
    of 0.1 would place the "hot start" a tenth of the domain away from the
    incumbent in every coordinate at once.
    """
    from optimizers.continuous.variables import InputContinuousVariable
    return [InputContinuousVariable(f"p{i}", float(lo), float(hi),
                                    initial_value=float(c),
                                    perturbation=perturbation)
            for i, ((lo, hi), c) in enumerate(zip(problem.bounds, problem.x0))]


def _generations(obj, population):
    """Enough generations that the budget, not the generation count, binds."""
    return max(2, int(np.ceil(obj.max_evals / max(population, 1))) + 2)


def _solve_with_injection(optimizer, problem, archive_size):
    """Inject the hot start at deck row 0 and solve preserving exactly that row."""
    try:
        optimizer.soln_deck.solution_archive[0] = problem.x0
        preserve = 1.0 / max(archive_size, 1)
    except Exception:  # noqa: BLE001 -- no deck on this optimizer; still runnable
        preserve = 0.0
    optimizer.solve(preserve_percent=preserve)


def _common(config_cls, name, obj, population, archive, **extra):
    """Config shared by every package arm: single-threaded, budget-bound."""
    return config_cls(
        name=name,
        population_size=population,
        num_generations=_generations(obj, population),
        solution_archive_size=archive,
        # The budget is the stop condition. Early stopping on stagnation would
        # end an arm before it had spent what the others spent, which is exactly
        # the asymmetry this study is set up to avoid.
        stop_after_iterations=10 ** 6,
        target_score=-np.inf,
        n_jobs=1,
        joblib_prefer="threads",
        **extra)


def opt_ga(obj, problem, seed, **hp):
    from optimizers import (GeneticAlgorithmOptimizer,
                            GeneticAlgorithmOptimizerConfig, set_seed)
    set_seed(seed)
    population = hp.get("population_size", 30)
    archive = hp.get("archive", 100)
    cfg = _common(GeneticAlgorithmOptimizerConfig, "ga", obj, population, archive,
                  local_grad_optim=hp.get("local_grad_optim", "none"))
    opt = GeneticAlgorithmOptimizer(config=cfg, variables=_variables(problem),
                                    fcn=obj)
    _solve_with_injection(opt, problem, archive)


def opt_pso(obj, problem, seed, **hp):
    from optimizers import (ParticleSwarmOptimizer,
                            ParticleSwarmOptimizerConfig, set_seed)
    set_seed(seed)
    population = hp.get("population_size", 30)
    archive = hp.get("archive", 100)
    cfg = _common(ParticleSwarmOptimizerConfig, "pso", obj, population, archive,
                  local_grad_optim=hp.get("local_grad_optim", "none"))
    opt = ParticleSwarmOptimizer(config=cfg, variables=_variables(problem), fcn=obj)
    _solve_with_injection(opt, problem, archive)


def opt_aco(obj, problem, seed, **hp):
    from optimizers import (AntColonyOptimizer, AntColonyOptimizerConfig,
                            set_seed)
    set_seed(seed)
    population = hp.get("population_size", 30)
    archive = hp.get("archive", 100)
    cfg = _common(AntColonyOptimizerConfig, "aco", obj, population, archive,
                  learning_rate=hp.get("learning_rate", 0.5),
                  q=hp.get("q", 1.0),
                  local_grad_optim=hp.get("local_grad_optim", "none"))
    opt = AntColonyOptimizer(config=cfg, variables=_variables(problem), fcn=obj)
    _solve_with_injection(opt, problem, archive)


def opt_gd(obj, problem, seed, **hp):
    """The package's own gradient descent -- one of the two arms that reads
    `initial_value`, so its hot start needs no injection."""
    from optimizers import (GradientDescentOptimizer,
                            GradientDescentOptimizerConfig, set_seed)
    set_seed(seed)
    population = hp.get("population_size", 10)
    archive = hp.get("archive", 30)
    cfg = _common(GradientDescentOptimizerConfig, "gd", obj, population, archive)
    opt = GradientDescentOptimizer(config=cfg, variables=_variables(problem),
                                   fcn=obj)
    _solve_with_injection(opt, problem, archive)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ARMS = {
    "none": None,                      # the hot start itself, scored as the reference
    "scipy-lbfgsb": scipy_lbfgsb,
    "scipy-powell": scipy_powell,
    "scipy-de": scipy_de,
    "opt-ga": opt_ga,
    "opt-pso": opt_pso,
    "opt-aco": opt_aco,
    "opt-gd": opt_gd,
}

# Which warm-start mechanism each arm actually uses, for the table's own note.
HOT_START = {
    "none": "—",
    "scipy-lbfgsb": "x0 argument",
    "scipy-powell": "x0 argument",
    "scipy-de": "x0 argument",
    "opt-ga": "deck row 0 + trust region",
    "opt-pso": "deck row 0 + trust region",
    "opt-aco": "deck row 0 + trust region",
    "opt-gd": "initial_value + trust region",
}


def run(arm, obj, problem, seed, **hp):
    """Spend the budget; swallow the stop signal. Returns nothing -- the budget
    object holds the best point seen, which is the only result that counts."""
    fn = ARMS[arm]
    if fn is None:
        return
    try:
        fn(obj, problem, seed, **hp)
    except BudgetExhausted:
        pass
