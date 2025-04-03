from abc import abstractmethod
from typing import Callable

import joblib
import numpy as np
import tqdm
from numpy.random import Generator
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, stdev=1, low=0, high=10):
    if stdev == 0:
        stdev = 1
    return truncnorm(
        (low - mean) / stdev, (high - mean) / stdev, loc=mean, scale=stdev
    ).rvs()


class AcoVariable:
    def __init__(self, name: str):
        self.name = name
        self.initial_value = 0.0

    @abstractmethod
    def random_value(
        self,
        rng: Generator | None = None,
        current_value: np.float64 = np.nan,
        other_values: np.array = None,
        learning_rate: float = 0.7,
    ) -> np.float64:
        pass

    @abstractmethod
    def initial_random_value(
        self, rng: Generator | None = None, perturbation: float = 0.1
    ) -> np.float64:
        pass


class AcoDiscreteVariable(AcoVariable):
    def __init__(self, name: str, values: list, initial_value: int = None):
        super().__init__(name)
        self.values = values
        self.initial_value = initial_value or self.random_value()

    def __repr__(self):
        return f"ACO_DV:{self.name} in {self.values}"

    def random_value(
        self,
        rng: Generator = None,
        current_value: np.float64 = np.nan,
        other_values: np.array = None,
        learning_rate: float = 0.7,
    ):
        if rng is None:
            rng = np.random.default_rng()
        if other_values is not None:
            # Convert into a weighted count, but ensure every option has a non-zero probability
            all_values = np.concatenate((self.values, other_values))
            unique, counts = np.unique(all_values, return_counts=True)
            # Unity normalize - TODO - Utilize the learning rate to adjust the non-base weights
            p_count = counts / np.sum(counts)
            return rng.choice(self.values, p=p_count)
        return rng.choice(self.values)

    def initial_random_value(
        self, rng: Generator | None = None, perturbation: float = 0.1
    ) -> np.float64:
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self.values)


class AcoContinuousVariable(AcoVariable):
    def __init__(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        initial_value: float = None,
        perturbation: float = 0.1,
    ):
        super().__init__(name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.initial_value = self.initial_random_value()
        if initial_value is not None:
            # Use perturbation theory around the initial value
            self.initial_value = min(
                self.upper_bound,
                max(
                    self.lower_bound,
                    initial_value + perturbation * (upper_bound - lower_bound),
                ),
            )

    def __repr__(self):
        return f"ACO_CV:{self.name} in [{self.lower_bound}, {self.upper_bound}]"

    def random_value(
        self,
        rng: Generator = None,
        current_value: np.float64 = np.nan,
        other_values: np.array = None,
        learning_rate: float = 0.7,
    ):
        if rng is None:
            rng = np.random.default_rng()
        if other_values is not None:
            # TODO - Other than Mahattan distance, what other distance metrics can be used?
            D2 = np.sum(np.abs(other_values - current_value)) / len(other_values)
            return get_truncated_normal(
                mean=current_value,
                stdev=learning_rate * D2,
                low=self.lower_bound,
                high=self.upper_bound,
            )
        return rng.uniform(self.lower_bound, self.upper_bound)

    def initial_random_value(
        self, rng: Generator | None = None, perturbation: float = 0.1
    ) -> np.float64:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lower_bound, self.upper_bound)


def test_ackley(x: np.array) -> np.float64:
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        + np.exp(1)
        + 20
    )


def solve_aco(
    fcn: Callable[[np.array], np.float64],
    variables: list[AcoVariable],
    num_generations: int = 30,
    solution_archive_size: int = -1,
    learning_rate: float = 0.7,
    num_ants: int = -1,
    q: float = 1.0,
) -> tuple[np.array, np.array]:
    if solution_archive_size < 0:
        solution_archive_size = len(variables) * 2
    if num_ants < 0:
        num_ants = solution_archive_size // 3
    # TODO - Debug allow picking the seed for predictable results
    rng = np.random.default_rng()
    # Construct the solution archive
    solution_archive = np.zeros((solution_archive_size, len(variables)))
    solution_values = np.zeros(solution_archive_size)
    # The solution weights don't ever change
    j = np.r_[1 : solution_archive_size + 1]
    solution_weights = (
        1
        / (q * solution_archive_size * np.sqrt(2 * np.pi))
        * np.exp(-((j - 1 / 2) ** 2) / (2 * (q * solution_archive_size) ** 2))
    )
    # Precompute the probability of selection
    p_j = solution_weights / np.sum(solution_weights)
    cp_j = np.cumsum(p_j)
    for k in range(solution_archive_size):
        for i, variable in enumerate(variables):
            solution_archive[k, i] = variable.initial_random_value(rng)
        solution_values[k] = fcn(solution_archive[k])
    # insert the initial solutions to the archive
    for i, variable in enumerate(variables):
        solution_archive[0, i] = variable.initial_value
    solution_values[0] = fcn(solution_archive[0])

    # Sort the solutions by their values
    sorted_indices = np.argsort(solution_values)
    solution_archive = solution_archive[sorted_indices]
    solution_values = solution_values[sorted_indices]
    best_soln_history = np.zeros(num_generations)

    # Add the progress bar
    generation_pbar = tqdm.trange(num_generations, desc=f"ACO Solver generation")
    n_jobs = min(8, joblib.cpu_count() - 1)
    ants_per_job = max(1, num_ants // n_jobs)
    parallel = joblib.Parallel(n_jobs=n_jobs, prefer="processes")
    for generation in generation_pbar:
        job_output = parallel(
            joblib.delayed(run_ants)(
                ants_per_job,
                cp_j,
                fcn,
                generation,
                learning_rate,
                rng,
                solution_archive,
                variables,
            )
            for job in range(n_jobs)
        )
        ant_solutions = job_output[0][0]
        ant_values = job_output[0][1]
        # After all ants have generated their solutions, update the solution archive
        solution_archive = np.vstack((solution_archive, ant_solutions))
        solution_values = np.hstack((solution_values, ant_values))
        # Sort the solutions by their values
        sorted_indices = np.argsort(solution_values)
        solution_archive = solution_archive[sorted_indices]
        solution_values = solution_values[sorted_indices]
        # Chop off the worst solutions
        solution_archive = solution_archive[:solution_archive_size]
        solution_values = solution_values[:solution_archive_size]
        # Clear the temp archive
        ant_solutions[:, :] = 0.0
        ant_values[:] = 0.0
        # Store the ongoing best value
        best_soln_history[generation] = solution_values[0]
        generation_pbar.set_postfix(best_value=solution_values[0])

    # Return the best solution
    return solution_archive[0, :], best_soln_history


def grad_descent(
    fcn: Callable[[np.array], np.float64],
    x0: np.array,
    learning_rate: float = 0.5,
    num_generations: int = 10,
    h=0.001,
) -> tuple[np.array, np.array]:
    x = x0
    x_history = np.zeros((num_generations, len(x0)))
    x_plus = x.copy()
    x_minus = x.copy()
    for generation in range(num_generations):
        grad = np.zeros(len(x0))
        x_plus[:] = x
        x_minus[:] = x
        for i in range(len(x0)):
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (fcn(x_plus) - fcn(x_minus)) / (2 * h)
            x_plus[i] = x[i]
            x_minus[i] = x[i]
        x -= learning_rate * grad
        x_history[generation, :] = x
    return x, x_history


def run_ants(
    n_ants,
    cp_j,
    fcn,
    generation,
    learning_rate,
    rng,
    solution_archive,
    variables,
    step_explore=False,
):
    ant_solutions = np.zeros((n_ants, len(variables)))
    ant_values = np.zeros(n_ants)
    cv_selector = [
        1 if isinstance(variable, AcoContinuousVariable) else 0
        for variable in variables
    ]
    x0 = np.zeros(len(cv_selector))
    for ant in range(n_ants):
        new_solution = np.zeros(len(variables))
        # Generate a new solution from an existing one as a base
        p = rng.uniform()
        # Find the entry based upon cdf
        base_solution_idx = np.searchsorted(cp_j, p)
        base_solution = solution_archive[base_solution_idx, :]
        for i, variable in enumerate(variables):
            # Compute the weighted value for the variable
            new_solution[i] = variable.random_value(
                current_value=base_solution[i],
                other_values=solution_archive[:, i],
                learning_rate=learning_rate,
            )
        # Evaluate the new solution
        new_value = fcn(new_solution)
        # Step around and find something better?
        if step_explore:
            x0 = new_solution[cv_selector]
            new_grad_soln = new_solution.copy()

            def cv_only_fcn(x):
                new_grad_soln[cv_selector] = x
                return fcn(new_grad_soln)

            x_new, x_history = grad_descent(cv_only_fcn, x0)
            new_solution[cv_selector] = x_new
            new_value = fcn(new_solution)

        # Store the new solution in the temporary archive.
        ant_solutions[ant, :] = new_solution
        ant_values[ant] = new_value
    return ant_solutions, ant_values


if __name__ == "__main__":
    best_soln, soln_history = solve_aco(
        test_ackley,
        [AcoContinuousVariable("x", -15, 30), AcoContinuousVariable("y", -15, 30)],
    )
    print(f"Solution history: {soln_history}")
    print(f"Best solution: {best_soln} with value: {soln_history[-1]}")
