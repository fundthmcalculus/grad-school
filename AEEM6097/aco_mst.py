import logging

import joblib
import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numpy._typing import NDArray
from tqdm import trange

@njit
def run_ant_mst(network_routes: NDArray, tau_xy: NDArray, alpha: float, beta: float) -> tuple[list[tuple[int,int]], float]:
    # Start at city 1, and join each city to the ever-growing spanning tree.
    eta_shape_ = network_routes.shape[0]
    # City pairings - [from, to]
    city_links: list[tuple[int,int]] = []
    city_indices = list(range(eta_shape_))
    # For performance, permute the city_indices once, since this "random"
    visited_cities = list()
    total_length = 0
    # NOTE - We can precompute probabilities
    p_mat = (tau_xy ** alpha) * (network_routes ** -beta)
    # While cities haven't been allocated
    from_city = _np_choice(city_indices)
    visited_cities.append(from_city)
    while len(visited_cities) < eta_shape_:
        # Randomly pick a city off the allowlist
        # Do the weighted choice array of where to join to exist
        p = p_mat[from_city, :]
        # Remove negatives, which are self-reference
        p[p < 0] = 0.0
        for vc in visited_cities:
            p[vc] = 0.0
        # p[visited_cities] = 0.0
        # Remove anything not in the allowed cities
        p = p / p.sum()
        # Choose which city to attach
        to_city = _np_choice(city_indices, p=p)
        if to_city not in visited_cities:
            # Store in the city-links in low-high order, since we have a bidirectional graph
            city_links.append((from_city, to_city))
            total_length += network_routes[from_city, to_city]
            visited_cities.append(to_city)
        from_city = to_city

    return city_links, total_length

@njit
def _np_choice(options, p = None):
    if p is None:
        return options[np.random.randint(len(options))]

    r = np.random.random()
    c = 0
    for i, p_i in enumerate(p):
        c += p_i
        if c >= r:
            return options[i]
    return options[-1]


def aco_mst_solve(network_routes: np.ndarray, n_ants=10, n_iter=10,
                  hot_start=None,
                  hot_start_length=0.0):
    network_routes = network_routes.copy()
    network_routes[network_routes == 0] = -1
    # Define the pheromone influence
    alpha = 1
    beta = 1
    # Update constant - from: https://doi.org/10.1016/j.tcs.2010.02.012
    # H = N^3 * L
    L = 1.0
    H = network_routes.shape[0]**3 * L
    # Default trail level
    tau = np.ones(network_routes.shape)
    # If we have a hot start, preload it 4x
    optimal_city_order = None
    tour_lengths = []
    optimal_tour_length = np.inf
    if hot_start is not None:
        optimal_tour_length = hot_start_length
        optimal_city_order = hot_start
        for i in range(len(hot_start) - 1):
            tau[hot_start[i], hot_start[i + 1]] += H

    n_jobs = 1 # joblib.cpu_count()//4 * 3
    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        for generation in trange(n_iter, desc="ACO Generation"):
            def parallel_ant(num_ants):
                best_links = []
                best_total_length = np.inf
                for _ in range(num_ants):
                    ant_links, ant_total_length = run_ant_mst(network_routes, tau, alpha, beta)
                    if ant_total_length < best_total_length:
                        best_links = ant_links
                        best_total_length = ant_total_length
                return best_links, best_total_length
            all_results = parallel(delayed(parallel_ant)(n_ants // n_jobs) for i_ant in range(n_jobs))

            optimal_ant_city_order, optimal_ant_len = get_optimal_ant_result(all_results)
            # Update the per-generation information
            if optimal_ant_len < optimal_tour_length:
                optimal_tour_length = optimal_ant_len
                optimal_city_order = optimal_ant_city_order
            delta_tau = update_pheromone(H,L, optimal_city_order, tau.shape)
            tour_lengths.append(optimal_tour_length)
            # If the last 10 generations are the same, stop early.
            n_stops = 50
            if len(tour_lengths) > n_stops and np.all(np.diff(tour_lengths[-n_stops:]) == 0):
                print("Stopping early due to lack of improvement.")
                break
            # Once all ants are done, update the pheromone
            tau = delta_tau / delta_tau.max()
    return optimal_city_order, optimal_tour_length, tour_lengths


def get_optimal_ant_result(all_results):
    optimal_ant_len = np.inf
    optimal_ant_city_order = None
    for ant in range(len(all_results)):
        tour_length = all_results[ant][1]
        city_order = all_results[ant][0]
        # If a dead-end, skip!
        if tour_length == np.inf:
            continue
        # Update the relative ant pheromone
        if tour_length < optimal_ant_len:
            optimal_ant_len = tour_length
            optimal_ant_city_order = city_order
    return optimal_ant_city_order, optimal_ant_len


def update_pheromone(H,L, city_order, tau_shape):
    # Compute the change in pheromone!
    delta_tau = L * np.ones(tau_shape)
    for i in range(len(city_order)):
        delta_tau[city_order[i][0], city_order[i][1]] += H
        # Because this is a bigraph, update the transpose as well.
        delta_tau[city_order[i][1], city_order[i][0]] += H
    return delta_tau


def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy =  (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()