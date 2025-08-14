import logging

import numpy as np
from joblib import Parallel, delayed
from numpy._typing import NDArray
from tqdm import trange


def run_ant_mst(network_routes: NDArray, tau_xy: NDArray, alpha: float, beta: float) -> tuple[list[tuple[int,int]], float]:
    # Start at city 1, and join each city to the ever-growing spanning tree.
    cur_row = 1 # Offset by 1, so we start at city 1
    eta_shape_ = network_routes.shape[0]
    # City pairings - [from, to]
    city_links: list[tuple[int,int]] = []
    city_indices = list(range(eta_shape_))
    # For performance, permute the city_indices once, since this "random"
    allowed_cities = city_indices.copy()
    total_length = 0
    # NOTE - We can precompute probabilities
    p_mat = (tau_xy ** alpha) * (network_routes ** -beta)
    # While cities haven't been allocated
    while cur_row < eta_shape_:
        # Randomly pick a city off the allowlist
        # new_city = np.random.choice(allowed_cities)
        new_city = _np_choice(allowed_cities)
        # Do the weighted choice array of where to join to exist
        p = p_mat[new_city, :]
        # Remove negatives, which are self-reference
        p[p < 0] = 0.0
        # Remove the reciprocal pairings.
        for fc, tc in city_links:
            if fc == new_city:
                p[tc] = 0.0
            elif tc == new_city:
                p[fc] = 0.0
        p = p / p.sum()
        # Choose which city to attach
        # from_city = np.random.choice(city_indices, p=p)
        from_city = _np_choice(city_indices, p=p)

        # Store in the city-links in low-high order, since we have a bidirectional graph
        city_links.append((from_city, new_city))
        total_length += network_routes[from_city, new_city]
        cur_row += 1
        allowed_cities.remove(new_city)

    return city_links, total_length


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
    # Define the pheromone decay rate
    rho = 0.5
    # Define the pheromone influence
    alpha = 1
    beta = 1
    # Update constant
    Q = 10
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
            tau[hot_start[i], hot_start[i + 1]] += 10*Q / hot_start_length

    n_jobs = 1
    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        for generation in trange(n_iter, desc="ACO Generation"):
            # Compute the change in pheromone!
            delta_tau = np.zeros(tau.shape)
            optimal_ant_len = np.inf
            optimal_ant_city_order = None
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

            for ant in range(len(all_results)):
                tour_length = all_results[ant][1]
                city_order = all_results[ant][0]
                # If a dead-end, skip!
                if tour_length == np.inf:
                    continue
                # Update the relative ant pheromone
                if tour_length <= optimal_ant_len:
                    optimal_ant_len = tour_length
                    optimal_ant_city_order = city_order
                for i in range(len(city_order)):
                    delta_tau[city_order[i][0], city_order[i][1]] += Q / tour_length
                    # Because this is a bigraph, update the transpose as well.
                    delta_tau[city_order[i][1], city_order[i][0]] += Q / tour_length
            # Update the per-generation information
            if optimal_ant_len < optimal_tour_length:
                optimal_tour_length = optimal_ant_len
                optimal_city_order = optimal_ant_city_order
            tour_lengths.append(optimal_tour_length)
            # If the last 10 generations are the same, stop early.
            n_stops = 50
            if len(tour_lengths) > n_stops and np.all(np.diff(tour_lengths[-n_stops:]) == 0):
                print("Stopping early due to lack of improvement.")
                break
            # Once all ants are done, update the pheromone
            tau = pheromone_update(tau, delta_tau, rho)
    return optimal_city_order, optimal_tour_length, tour_lengths



def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy =  (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()