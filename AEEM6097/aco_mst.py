import joblib
import numpy as np
from joblib import Parallel, delayed
from numpy._typing import NDArray
from tqdm import trange


def run_ant_mst(network_routes: NDArray, eta: NDArray, tau_xy: NDArray, alpha: float, beta: float) -> tuple[NDArray, float]:
    # Start at city 1, and join each city to the ever-growing spanning tree.
    cur_row = 0 # Offset by 1, so we start at city 1
    eta_shape_ = eta.shape[0]
    order_len = eta_shape_
    # City pairings - [from, to]
    city_links: list[tuple[int,int]] = []
    city_indices = list(range(eta_shape_))
    allowed_cities = city_indices.copy()
    total_length = 0
    # While cities haven't been allocated
    while cur_row < eta_shape_:
        # Randomly pick a city off the allow-list
        new_city = np.random.choice(allowed_cities, replace=False)
        # Do the weighted choice array of where to join to exist
        p = p_xy(eta, tau_xy,alpha,beta, new_city)
        # Remove the reciprocal pairings.
        p[[tc for fc,tc in city_links if fc == new_city]] = 0.0
        p = p / p.sum()
        if p.sum() == 0.0:
            return city_links, total_length
        # Choose which city to attach
        from_city = np.random.choice(city_indices, p=p)
        # Store in the city-links
        city_links.append((from_city, new_city))
        total_length += network_routes[from_city, new_city]
        cur_row += 1
        allowed_cities.remove(new_city)

    return city_links, total_length



def p_xy(eta_xy, tau_xy, alpha, beta, x):
    # Because this is an MST, any city pairing is allowed.
    p = (tau_xy[x,:] ** alpha) * eta_xy[x,:] ** beta
    # Remove negatives, which are self-reference
    p[p < 0] = 0.0
    # Normalize the probabilities
    if np.sum(p) == 0.0:
        return 0
    p = p / np.sum(p)
    return p



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
    # Desirability of a given move.
    eta = 1 / network_routes
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

    with Parallel(n_jobs=1, prefer="threads") as parallel:
        for generation in trange(n_iter, desc="ACO Generation"):
            # Compute the change in pheromone!
            delta_tau = np.zeros(tau.shape)
            optimal_ant_len = np.inf
            optimal_ant_city_order = None
            def parallel_ant(local_ant):
                return run_ant_mst(network_routes, eta, tau, alpha, beta)
            all_results = parallel(delayed(parallel_ant)(i_ant) for i_ant in range(n_ants))

            for ant in range(n_ants):
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
            # Update the per-generation information
            if optimal_ant_len < optimal_tour_length:
                optimal_tour_length = optimal_ant_len
                optimal_city_order = optimal_ant_city_order
            tour_lengths.append(optimal_tour_length)
            # Once all ants are done, update the pheromone
            tau = pheromone_update(tau, delta_tau, rho)
    return optimal_city_order, optimal_tour_length, tour_lengths



def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy =  (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()