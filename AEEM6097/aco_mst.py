import os

from tqdm import tqdm

# needs to appear before `from numba import cuda`
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

import numpy as np
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numpy._typing import NDArray


def _cuda_ant_mst_solve(n_iter: np.int32, n_ants: np.int32,
                        alpha: np.float64, beta: np.float64, H: np.float64, L: np.float64,
                        network_routes: NDArray, city_indices: NDArray, city_links: NDArray, mst_dist: NDArray,
                        visited_cities: NDArray,
                        p_mat: NDArray, tau: NDArray, optimal_tour_lengths: NDArray, optimal_links: NDArray):
    blocks_per_grid, rng_states, threads_per_block = get_cuda_details(n_ants)
    blk, tpb = compute_cuda_details(p_mat.size)
    for generation in tqdm(range(n_iter), desc="ACO Iteration"):
        _cuda_update_p_mat[blk, tpb](p_mat, tau, network_routes, alpha, beta)
        cuda.synchronize()
        _cuda_ant_mst[blocks_per_grid, threads_per_block](rng_states, network_routes, p_mat, city_indices, city_links, mst_dist, visited_cities)
        cuda.synchronize()
        _cuda_find_optimal_tour_links[blocks_per_grid, threads_per_block](generation, optimal_tour_lengths, city_links, mst_dist, optimal_links)
        cuda.synchronize()
        _cuda_update_pheromone[blk, tpb](H, L, optimal_links, tau)
        cuda.synchronize()


@cuda.jit
def _cuda_min_mst_dist(mst_dist, chunk=16, stride=1):
    # TODO - Make this actually work!
    pos = get_cuda_pos()
    for p in range(chunk):
        p_stride = pos + p * stride
        if p_stride >= mst_dist.size:
            return
        if mst_dist[p_stride] < mst_dist[pos]:
            mst_dist[pos] = mst_dist[p_stride]


@cuda.jit
def _cuda_find_optimal_tour_links(generation, optimal_tour_lengths, city_links, mst_dist, optimal_links):
    pos = get_cuda_pos()
    # Post in-place min, we know only the first call matters
    if pos >= mst_dist.size:
        return

    old_optimal_len = optimal_tour_lengths[generation-1]

    # Get minimum mst_dist using reduction
    optimal_ant_len = mst_dist[0]
    for i in range(1, mst_dist.size):
        if mst_dist[i] < optimal_ant_len:
            optimal_ant_len = mst_dist[i]

    if old_optimal_len >= optimal_ant_len == mst_dist[pos]:
        optimal_tour_lengths[generation] = optimal_ant_len
        for i in range(city_links.shape[1]):
            for j in range(city_links.shape[2]):
                optimal_links[i, j] = city_links[pos, i, j]
    elif old_optimal_len >= optimal_ant_len:
        optimal_tour_lengths[generation] = optimal_ant_len
    else:
        optimal_tour_lengths[generation] = old_optimal_len


@cuda.jit
def _cuda_update_p_mat(p_mat, tau, network_routes, alpha, beta):
    pos = get_cuda_pos()
    if pos >= p_mat.size:
        return
    i = pos // tau.shape[1]
    j = pos % tau.shape[1]
    p_mat[i, j] = tau[i, j] ** alpha * network_routes[i, j] ** -beta


@cuda.jit
def _cuda_ant_mst(rng_states, network_routes, p_mat, city_indices, city_links2, mst_dist2, visited_cities):
    pos = get_cuda_pos()

    if pos < city_links2.shape[0]:
        run_ant_mst(rng_states, network_routes, p_mat, city_indices, city_links2, mst_dist2, visited_cities)

@cuda.jit
def get_cuda_pos():
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute the flattened index inside the array
    pos = tx + ty * bw
    return pos


@cuda.jit
def run_ant_mst(rng_states, network_routes: NDArray, p_mat: NDArray, city_indices: NDArray, city_links: NDArray, mst_dist: NDArray, visited_cities: NDArray):
    pos = get_cuda_pos()
    # Start at city 1, and join each city to the ever-growing spanning tree.
    eta_shape_ = network_routes.shape[0]
    # City pairings - [from, to]
    cur_row = 0
    total_length = 0
    # While cities haven't been allocated
    from_city = _np_choice(rng_states, city_indices)
    # Clear this cache
    mst_dist[pos] = 0
    for idx in range(eta_shape_):
        visited_cities[pos, idx] = 0
        if idx < eta_shape_ - 1:
            city_links[pos, idx, :] = 0

    visited_cities[pos, from_city] = 1
    while cur_row < eta_shape_ - 1:
        # Randomly pick a city off the allowlist
        # Do the weighted choice array of where to join to exist
        # Choose which city to attach
        to_city = _np_choice2(rng_states, city_indices, p_mat[from_city, :], visited_cities[pos, :])
        if visited_cities[pos, to_city] == 0:
            # Store in the city-links in low-high order, since we have a bidirectional graph
            city_links[pos, cur_row, 0] = from_city
            city_links[pos, cur_row, 1] = to_city
            total_length += network_routes[from_city, to_city]
            visited_cities[pos, to_city] = 1
            cur_row += 1
        from_city = to_city

    mst_dist[pos] = total_length


@cuda.jit
def _np_choice(rng_states, options):
    thread_id = cuda.grid(1)
    r = xoroshiro128p_uniform_float32(rng_states, thread_id)
    return options[int(len(options)*r)]

@cuda.jit
def _np_choice2(rng_states, options, p, visited_cities):
    thread_id = cuda.grid(1)
    r = xoroshiro128p_uniform_float32(rng_states, thread_id)

    # Calculate total probability for unvisited cities
    total_prob = 0.0
    for i in range(len(p)):
        if visited_cities[i] == 0 and p[i] > 0:
            total_prob += p[i]

    if total_prob <= 0:
        return options[-1]

    # Get normalized random value
    r *= total_prob

    # Do weighted selection
    cumsum = 0.0
    for i in range(len(p)):
        if visited_cities[i] == 0 and p[i] > 0:
            cumsum += p[i]
            if cumsum >= r:
                return options[i]

    return options[-1]


@cuda.jit
def _cuda_update_pheromone(H,L, city_order, tau):
    pos = get_cuda_pos()
    if pos >= tau.size:
        return
    i = pos // tau.shape[1]
    j = pos % tau.shape[1]
    tau[i, j] = L
    # TODO - This only needs to be done once, but it's a bit tricky to do it in Numba'
    for i in range(len(city_order)):
        tau[city_order[i,0], city_order[i,1]] = H
        tau[city_order[i,1], city_order[i,0]] = H


def aco_mst_solve(network_routes: np.ndarray, n_ants=10, n_iter=10):
    network_routes = network_routes.copy()
    network_routes[network_routes == 0] = -1
    # Define the pheromone influence
    alpha = 1
    beta = 1
    # Update constant - from: https://doi.org/10.1016/j.tcs.2010.02.012
    # H = N^3 * L
    L = 1.0
    N = network_routes.shape[0]
    H = N ** 3 * L
    # Prenormalize
    L /= H
    H = 1.0
    # Default trail level
    tau = L*np.ones(network_routes.shape)
    # If we have a hot start, preload it 4x
    tour_lengths = []

    city_links = np.zeros((n_ants, N - 1, 2), dtype=np.int32)

    mst_dist = np.zeros(n_ants, dtype=np.float64)
    city_indices = np.arange(N, dtype=np.int32)
    visited_cities = np.zeros((n_ants, N), dtype=np.int32)
    optimal_links = np.zeros((N-1,2), dtype=np.int32)
    optimal_tour_lengths = np.inf*np.ones(n_iter, dtype=np.float64)
    p_mat = np.zeros_like(tau)

    nv_city_links = numba.cuda.to_device(city_links)
    nv_network_routes = numba.cuda.to_device(network_routes)
    nv_mst_dist = numba.cuda.to_device(mst_dist)
    nv_city_indices = numba.cuda.to_device(city_indices)
    nv_visited_cities = numba.cuda.to_device(visited_cities)
    nv_optimal_tour_lengths = numba.cuda.to_device(optimal_tour_lengths)
    nv_optimal_links = numba.cuda.to_device(optimal_links)
    nv_p_mat = numba.cuda.to_device(p_mat)
    nv_tau = numba.cuda.to_device(tau)

    # nv_city_links = (city_links)
    # nv_network_routes = (network_routes)
    # nv_mst_dist = (mst_dist)
    # nv_city_indices = (city_indices)
    # nv_visited_cities = (visited_cities)
    # nv_optimal_tour_lengths = (optimal_tour_lengths)
    # nv_optimal_links = (optimal_links)
    # nv_p_mat = (p_mat)
    # nv_tau = (tau)

    _cuda_ant_mst_solve(
         n_iter, n_ants, alpha, beta, H, L,
        nv_network_routes, nv_city_indices, nv_city_links, nv_mst_dist, nv_visited_cities,
        nv_p_mat, nv_tau, nv_optimal_tour_lengths, nv_optimal_links)

    # Copy back results from GPU
    # city_links = nv_city_links.copy_to_host()
    # mst_dist = nv_mst_dist.copy_to_host()
    optimal_tour_lengths = nv_optimal_tour_lengths.copy_to_host()
    optimal_links = nv_optimal_links.copy_to_host()
    # optimal_tour_lengths = nv_optimal_tour_lengths
    # optimal_links = nv_optimal_links

    return optimal_links, optimal_tour_lengths.min(), optimal_tour_lengths


def get_cuda_details(n_ants):
    blocks_per_grid, threads_per_block = compute_cuda_details(n_ants)
    rng_states = create_xoroshiro128p_states(int(threads_per_block * blocks_per_grid), seed=1)
    return blocks_per_grid, rng_states, threads_per_block


def compute_cuda_details(N):
    threads_per_block = 1 if os.environ["NUMBA_ENABLE_CUDASIM"] == "1" else 64
    blocks_per_grid = np.ceil(N / threads_per_block).astype(np.int32)
    return blocks_per_grid, threads_per_block


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