import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from AEEM6097.aco_mst import aco_mst_solve
from AEEM6097.mod_vat import compute_ivat_ordered_dissimilarity_matrix2, compute_ordered_dis_njit2
from AEEM6097.test2 import aco_tsp_solve, check_path_distance

N_CITIES_CLUSTER = 32
N_CLUSTERS = N_CITIES_CLUSTER//2

N_ANTS = 10*N_CITIES_CLUSTER
N_GENERATIONS = 10*N_CLUSTERS

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 10*CLUSTER_DIAMETER

HALF_CIRCLE = False


def random_cities(center_x, center_y, n_cities=-1) -> np.ndarray:
    if n_cities == -1:
        n_cities = N_CITIES_CLUSTER
    # Randomly distribute cities in a uniform circle?
    theta = np.linspace(0, 2 * np.pi, n_cities+1, dtype=np.float32)
    theta = theta[:-1]
    city_x = np.cos(theta) * CLUSTER_DIAMETER/2.0 + center_x
    city_y = np.sin(theta) * CLUSTER_DIAMETER/2.0 + center_y
    return np.c_[city_x, city_y]


def circle_random_clusters(n_clusters=-1, n_cities=-1) -> np.ndarray:
    if n_clusters == -1:
        n_clusters = N_CLUSTERS
    city_locations = np.zeros(shape=(0,2), dtype=np.float32)
    for theta in np.linspace(0, 2*np.pi, n_clusters):
        if HALF_CIRCLE:
            theta /= 2.0
        else:
            theta *= n_clusters/(n_clusters+1)
        cx = CLUSTER_SPACING * np.cos(theta)
        cy = CLUSTER_SPACING * np.sin(theta)
        city_locations = np.concatenate((city_locations, random_cities(cx, cy, n_cities=n_cities)), axis=0)
    return city_locations


def poly_perimeter(n_sides, r):
    # Compute perimeter of inscribed polygon in circle of radius, r.
    return n_sides* 2 * r * np.sin(2 * np.pi /(2*n_sides))


def start_at_idx(v, x=0):
    # Find index of 0, move to front
    x_idx=np.argwhere(v==x)[0][0]
    return np.concatenate((v[x_idx:], v[:x_idx]), axis=0)


def main():
    print("Configuring random")
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: np.ndarray = pairwise_distances(all_cities)
    print("Distance-shape",distances.shape)

    # Create the report dataframe
    df_rows = []
    approx_optimal_dist = N_CLUSTERS * poly_perimeter(N_CITIES_CLUSTER, r=CLUSTER_DIAMETER / 2.0) + poly_perimeter(
        N_CITIES_CLUSTER, r=CLUSTER_SPACING) - N_CLUSTERS*CLUSTER_DIAMETER
    if HALF_CIRCLE:
        approx_optimal_dist /= 2.0
    rand_dist = check_path_distance(distances, np.random.permutation(np.arange(N_CLUSTERS*N_CITIES_CLUSTER)))
    # Add the optimal and random rows:
    df_rows.append({"Method": "Approx Optimal", "Time": 0.0, "Distance": approx_optimal_dist,
         "%Change": approx_optimal_dist / approx_optimal_dist * 100.0})
    df_rows.append(
        {"Method": "Random", "Time": 0.0, "Distance": rand_dist,
         "%Change": rand_dist / approx_optimal_dist * 100.0}
    )
    # Use the VAT technique to organize
    t0 = time.time()
    ivat_dist, ivat_path, vat_dist, vat_path = compute_ivat_ordered_dissimilarity_matrix2(all_cities)
    t1 = time.time()
    # Ensure we start at city-0
    vat_path = start_at_idx(vat_path)
    ivat_path = start_at_idx(ivat_path)
    vat_dist_len = vat_dist.diagonal(offset=1).sum() + vat_dist[0,-1]
    ivat_dist_len = ivat_dist.diagonal(offset=1).sum() + ivat_dist[0,-1]

    # Append row for VAT method.
    vat_time = t1 - t0
    df_rows.append(
        {"Method": "VAT", "Time": vat_time, "Distance": vat_dist_len,
         "%Change": vat_dist_len / approx_optimal_dist * 100.0,
         "Path": vat_path}
    )
    df_rows.append(
        {"Method": "IVAT", "Time": vat_time, "Distance": ivat_dist_len,
         "%Change": ivat_dist_len / approx_optimal_dist * 100.0,
         "Path": ivat_path}
    )
    # Compute TSP optimized distance
    t2 = time.time()
    hs_optimal_city_order, hs_optimal_tour_length, hs_tour_lengths = (
        aco_tsp_solve(distances,n_ants=N_ANTS, n_iter=N_GENERATIONS, hot_start=vat_path,
                      hot_start_length=vat_dist_len)
    )
    t3 = time.time()
    optimal_city_order, optimal_tour_length, tour_lengths = (
        aco_tsp_solve(distances,n_ants=N_ANTS, n_iter=N_GENERATIONS)
    )
    t4 = time.time()

    df_rows.append(
        {"Method": "HS-ACO", "Time": t3 - t2 + vat_time, "Distance": hs_optimal_tour_length,
         "%Change": hs_optimal_tour_length / approx_optimal_dist * 100.0,
         "Path": hs_optimal_city_order}
    )

    df_rows.append(
        {"Method": "ACO", "Time": t4 - t3, "Distance": optimal_tour_length,
         "%Change": optimal_tour_length / approx_optimal_dist * 100.0, "Path": optimal_city_order}
    )

    print("VAT order: ", vat_path[0:20])
    print("IVAT order: ", ivat_path[0:20])
    print("HS-ACO order: ", hs_optimal_city_order[0:20])
    print("ACO order: ", optimal_city_order[0:20])
    print("Report")
    df = pd.DataFrame(df_rows, columns=["Method", "Time", "Distance", "%Change", "Path"]).set_index("Method")
    print(df.drop('Path', axis=1))

    # plot_convergence(tour_lengths)
    plot_results(all_cities, distances, vat_dist, vat_path, optimal_city_order, hs_optimal_city_order, df)


def plot_results(all_cities: np.ndarray, distances: np.ndarray,
                 vat_dist: np.ndarray, vat_path,
                 aco_path, hs_aco_path, df: pd.DataFrame):
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Add titles to subplots
    subplot_titles = [f"ACO Path {df["Distance"]["ACO"]:.0f}",
                      f"VAT Path {df["Distance"]["VAT"]:.0f}",
                      "VAT Distances",
                      "HS-ACO Distances"]
    for i, ax in enumerate(axs.flat):
        ax.set_title(subplot_titles[i])

    # Add ACO path
    aco_path_coords = np.array([all_cities[i] for i in aco_path])
    axs[0, 0].scatter(all_cities[:, 0], all_cities[:, 1], marker='o', label='Cities')
    axs[0, 0].plot(aco_path_coords[:, 0], aco_path_coords[:, 1], 'r--', label='ACO Path')

    # Add VAT path
    vat_path_coords = np.array([all_cities[i] for i in vat_path])
    axs[0, 1].scatter(all_cities[:, 0], all_cities[:, 1], marker='o', label='Cities')
    axs[0, 1].plot(vat_path_coords[:, 0], vat_path_coords[:, 1], 'g-', label='VAT Path')

    # Add VAT distances heatmap (third subplot)
    im2 = axs[1, 0].imshow(vat_dist, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axs[1, 0], shrink=0.8)

    # Add IVAT distances heatmap (fourth subplot)
    im3 = axs[1, 1].imshow(distances[hs_aco_path,:][:,hs_aco_path], cmap='viridis', aspect='auto')
    fig.colorbar(im3, ax=axs[1, 1], shrink=0.8)

    # Add legend to the first subplot
    axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=4)
    axs[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=4)

    # Set overall title
    fig.suptitle(f"City Paths and Distance Matrices {N_CITIES_CLUSTER*N_CLUSTERS}", fontsize=16)

    # Adjust spacing between subplots
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for the overall title

    # Show the plot
    plt.show()


def get_optimal_order(links):
    """Get optimal order by recursively following connected nodes starting from node 0"""
    order = []  # Start with node 0
    seen = set()

    def add_connected(node):
        order.append(node)
        # Find all links containing this node
        for link in links:
            other = link[1] if link[0] == node else link[0] if link[1] == node else None
            if other is not None and link not in seen:
                seen.add(link)
                add_connected(other)

    add_connected(0)

    return order


def vat_mst(matrix_of_pairwise_distance):
    # Run the ACO MST solution.
    optimal_links, optimal_length, tour_lengths = aco_mst_solve(matrix_of_pairwise_distance, n_ants=N_ANTS, n_iter=N_GENERATIONS)
    # Plot the convergence
    plt.figure()
    plt.plot(tour_lengths)
    plt.xlabel("Iteration")
    plt.ylabel("MST Length")
    plt.title(f"ACO MST Convergence N_ANTS={N_ANTS}, N_ITER={N_GENERATIONS}")
    new_matrix = np.array(matrix_of_pairwise_distance, copy=True)
    # Sort further?
    # optimal_links = np.sort(optimal_links, axis=0)
    # optimal_links = sorted(optimal_links, key=lambda x: x[1])
    # Get the unique numbers in order in the list of tuples
    # optimal_order = list(dict.fromkeys([s for p in optimal_links for s in p]))
    # Get optimal order by following connected nodes
    optimal_order = get_optimal_order(optimal_links)

    new_matrix = new_matrix[optimal_order,:][:, optimal_order]
    return new_matrix, optimal_order


def vat_scaling():
    # for city_exp in range(1, 4):
    #     city_count = 2**city_exp
    #     print(f"City count: {city_count**2}")
    city_count = N_CLUSTERS*N_CITIES_CLUSTER
    all_cities = circle_random_clusters(n_clusters=N_CLUSTERS, n_cities=N_CITIES_CLUSTER)
    matrix_of_pairwise_distance = pairwise_distances(all_cities)
    # Scramble the order to ensure we sort it!
    cols = np.arange(len(all_cities), dtype="int")
    rand_col_order = np.random.permutation(cols)
    matrix_of_pairwise_distance = matrix_of_pairwise_distance[:, rand_col_order][rand_col_order, :]
    t1 = time.time()
    # Cluster using the library VAT
    ordered_matrix, observation_path = compute_ordered_dis_njit2(matrix_of_pairwise_distance)
    t2 = time.time()
    # Cluster using the ACO VAT
    ordered_matrix3, observation_path3 = vat_mst(matrix_of_pairwise_distance)
    t3 = time.time()

    # Ensure all values are equal!
    print("VAT-Lib: ", observation_path)
    print("VAT-ACO: ", observation_path3)

    # Print the results
    print(f"VAT-lib time: {t2-t1:.4f} seconds, distance: {ordered_matrix.diagonal(1).sum() + ordered_matrix[0,-1]:.1f}")
    print(f"VAT-ACO time: {t3-t2:.4f} seconds, distance: {ordered_matrix3.diagonal(1).sum() + ordered_matrix3[0,-1]:.1f}")

    plt.figure(figsize=(8, 6))
    plt.subplot(1,2,1)
    plt.imshow(ordered_matrix, cmap='viridis', aspect='auto', label='VAT-lib')
    plt.title(f"VAT Library: {t2-t1:.2f} seconds")
    plt.subplot(1,2,2)
    plt.imshow(ordered_matrix3, cmap='viridis', aspect='auto', label='VAT-ACO-MST')
    plt.title(f"VAT ACO-MST: {t3-t2:.2f} seconds")
    plt.show()


if __name__ == "__main__":
    # main()
    vat_scaling()
