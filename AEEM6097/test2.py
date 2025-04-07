from typing import Iterator

import joblib
import numpy as np
import plotly.graph_objects as go
import tqdm
from joblib import Parallel, delayed


# Source: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Algorithm_and_formula

def main():
    print("Test 2 - TSP solution")
    # Route is symmetric, so define only the top-half of the matrix
    # 9x9 grid:                1  2  3  4  5  6  7  8  9
    network_routes = np.array([[0, 5, 0,20, 4, 0, 0,14, 0,],
                              [0, 0, 6, 0, 7, 0, 0, 0, 0,],
                              [0, 0, 0,15,10, 0, 0, 0, 0,],
                              [0, 0, 0, 0,20, 7,12, 0, 0,],
                              [0, 0, 0, 0, 0, 3, 5,13, 6,],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                              [0, 0, 0, 0, 0, 0, 0, 7, 0,],
                              [0, 0, 0, 0, 0, 0, 0, 0, 5,],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                              ])
    # Because it's symmetric, we can use upper triangular matrix to define the lower triangular matrix.
    network_routes = network_routes + network_routes.T
    # Anything with distance = 0 is not a valid route
    optimal_city_order, optimal_tour_length, tour_lengths = aco_tsp_solve(network_routes)
    # Get the optimal route!
    print(f"Optimal route: {optimal_city_order + 1}. Tour length={optimal_tour_length}")
    # Plot the convergence of optimal route length!
    plot_convergence(tour_lengths)

    # Use recursion to find the optimal solution exhaustively, to prove that my ACO solution finds the best on the first try. 8-)
    results: list[tuple[list,int]] = list(recursive_find_best_soln(network_routes))
    print("Number of solutions found:", len(results))
    recur_best_path, recur_best_len = min(results, key=lambda x: x[1])
    print(f"Recursive best path: {np.array(recur_best_path)+1}. Tour length={recur_best_len}")


def aco_tsp_solve(network_routes: np.ndarray, n_ants=10, n_iter=10,
                  back_to_start=False,
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

    with Parallel(n_jobs=joblib.cpu_count()//2) as parallel:
        for generation in tqdm.trange(n_iter, desc="ACO Generation"):
            # Compute the change in pheromone!
            delta_tau = np.zeros(tau.shape)
            optimal_ant_len = np.inf
            optimal_ant_city_order = None
            all_city_order = [[]]*n_ants
            all_tour_length = [0]*n_ants
            def parallel_ant(local_ant):
                return run_ant(network_routes, eta, tau, alpha, beta, back_to_start)
            # for ant in range(n_ants):
            #     all_city_order[ant], all_tour_length[ant] = run_ant(network_routes, eta, tau, alpha, beta, back_to_start)#     all_city_order[ant], all_tour_length[ant] = run_ant(network_routes, eta, tau, alpha, beta, back_to_start)
            all_results = parallel(delayed(parallel_ant)(i_ant) for i_ant in range(n_ants))
            # all_city_order[ant], all_tour_length[ant] = run_ant(network_routes, eta, tau, alpha, beta, back_to_start)#     all_city_order[ant], all_tour_length[ant] =

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
                for i in range(len(city_order) - 1):
                    delta_tau[city_order[i], city_order[i + 1]] += Q / tour_length
                if back_to_start:
                    delta_tau[city_order[-1], city_order[0]] += Q / tour_length
            # Update the per-generation information
            if optimal_ant_len < optimal_tour_length:
                optimal_tour_length = optimal_ant_len
                optimal_city_order = optimal_ant_city_order
            tour_lengths.append(optimal_tour_length)
            # Once all ants are done, update the pheromone
            tau = pheromone_update(tau, delta_tau, rho)
    return optimal_city_order, optimal_tour_length, tour_lengths

    # Use recursion to find the optimal solution exhaustively, to prove that my ACO solution finds the best on the first try. 8-)
    results: list[tuple[list,int]] = list(recursive_find_best_soln(network_routes))
    # Sort by length
    results.sort(key=lambda x: -x[1])
    print("Number of solutions found:", len(results))
    print("Solutions in order: ", results)
    # print(f"Recursive best path: {np.array(recur_best_path)+1}. Tour length={recur_best_len}")

def plot_convergence(tour_lengths):
    # Create the figure
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(
        go.Scatter(
            x=np.r_[0:len(tour_lengths)],
            y=tour_lengths,
            mode='lines+markers',
            name='Tour Length',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        )
    )

    # Update layout
    fig.update_layout(
        title='ACO Convergence',
        xaxis_title='Generation',
        yaxis_title='Tour Length',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show the figure
    fig.show()


def recursive_find_best_soln(network_routes, cur_city = 0, used_cities=None, cur_length = 0) -> Iterator[tuple[list,int]]:
    if used_cities is None:
        used_cities = []
    if len(used_cities) > 0:
        cur_length += network_routes[used_cities[-1], cur_city]
    used_cities.append(cur_city)
    # If we have all cities, we're done!
    if len(used_cities) == network_routes.shape[0]:
        yield used_cities, cur_length
    # Look at the possible cities to move from here.
    for i in range(network_routes.shape[0]):
        if i in used_cities:
            continue
        if network_routes[cur_city, i] == -1:
            continue
        yield from recursive_find_best_soln(network_routes, i, used_cities.copy(), cur_length)


def run_ant(network_routes, eta, tau_xy, alpha, beta, back_to_start: bool):
    # Start at city 1, and visit each city exactly once
    cur_city = 0 # Offset by 1, so we start at city 1
    eta_shape_ = eta.shape[0]
    order_len = eta_shape_
    if back_to_start:
        order_len += 1
    city_order = np.zeros(order_len, dtype=int)
    idx = 0
    total_length = 0
    allowed_cities = np.ones(eta_shape_, dtype=bool)
    choice_indexes = np.arange(eta_shape_) # TODO - Cache!
    while np.any(allowed_cities):
        # Mark off the current city
        allowed_cities[cur_city] = False
        city_order[idx] = cur_city
        # Compute the probability of each city
        p = p_xy(eta, tau_xy, allowed_cities, alpha, beta, cur_city)
        # If the probability is zero, we're stuck, this is a dead end!
        if np.sum(p) == 0 or np.any(np.isnan(p)):
            # If we have hit every city, we're done! We don't need to go back to the start, since we solved in reverse.
            if np.sum(allowed_cities) != 0:
                # Invalid route!
                total_length = np.inf
            # IF back-to-start, include that option
            if back_to_start:
                city_order[-1] = 0
                total_length += network_routes[city_order[-2], city_order[-1]]
            break
        # Choose the next city
        cur_city = np.random.choice(choice_indexes, p=p)
        total_length += network_routes[city_order[idx], cur_city]
        idx += 1

    return city_order, total_length


def pheromone_update(tau_xy, delta_tau_xy, rho):
    new_tau_xy =  (1 - rho) * tau_xy + delta_tau_xy
    return new_tau_xy / new_tau_xy.max()


def p_xy(eta_xy, tau_xy, allowed_y, alpha, beta, x):
    p = (tau_xy[x,:] ** alpha) * eta_xy[x,:] ** beta
    # Remove negative probabilities, those are not allowed
    p[~allowed_y] = 0
    p[p < 0] = 0
    # Normalize the probabilities
    if np.sum(p) == 0.0:
        return 0
    p = p / np.sum(p)
    return p



if __name__ == "__main__":
    main()