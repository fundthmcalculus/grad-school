import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyclustertend import compute_ivat_ordered_dissimilarity_matrix
from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances

from AEEM6097.mod_vat import compute_ivat_ordered_dissimilarity_matrix2, compute_ordered_dis_njit2, \
    compute_merge_sort_dissimilarity_matrix
from AEEM6097.test2 import aco_tsp_solve, check_path_distance

N_CITIES_CLUSTER = 16
N_CLUSTERS = N_CITIES_CLUSTER

N_ANTS = 3*N_CITIES_CLUSTER
N_GENERATIONS = 3*N_CLUSTERS

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 10*CLUSTER_DIAMETER

HALF_CIRCLE = False


def random_cities(center_x, center_y, n_cities=-1) -> np.ndarray:
    if n_cities == -1:
        n_cities = N_CITIES_CLUSTER
    # Randomly distribute cities in a uniform circle?
    theta = np.linspace(0, 2 * np.pi, n_cities+1)
    theta = theta[:-1]
    city_x = np.cos(theta) * CLUSTER_DIAMETER/2.0 + center_x
    city_y = np.sin(theta) * CLUSTER_DIAMETER/2.0 + center_y
    return np.c_[city_x, city_y]


def circle_random_clusters(n_clusters=-1, n_cities=-1) -> np.ndarray:
    if n_clusters == -1:
        n_clusters = N_CLUSTERS
    city_locations = np.zeros(shape=(0,2))
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
        N_CITIES_CLUSTER, r=CLUSTER_SPACING)
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
         "%Change": vat_dist_len / approx_optimal_dist * 100.0}
    )
    df_rows.append(
        {"Method": "IVAT", "Time": vat_time, "Distance": ivat_dist_len,
         "%Change": ivat_dist_len / approx_optimal_dist * 100.0}
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
         "%Change": hs_optimal_tour_length / approx_optimal_dist * 100.0}
    )

    df_rows.append(
        {"Method": "ACO", "Time": t4 - t3, "Distance": optimal_tour_length,
         "%Change": optimal_tour_length / approx_optimal_dist * 100.0}
    )

    print("VAT order: ", vat_path[0:20])
    print("IVAT order: ", ivat_path[0:20])
    print("HS-ACO order: ", hs_optimal_city_order[0:20])
    print("ACO order: ", optimal_city_order[0:20])
    print("Report")
    df = pd.DataFrame(df_rows, columns=["Method", "Time", "Distance", "%Change"])
    print(df)

    # plot_convergence(tour_lengths)
    plot_results(all_cities, distances, vat_dist, vat_path, ivat_dist, ivat_path, optimal_city_order, hs_optimal_city_order)


def plot_results(all_cities: np.ndarray, distances: np.ndarray,
                 vat_dist: np.ndarray, vat_path,
                 ivat_dist: np.ndarray, ivat_path,
                 aco_path, hs_aco_path):
    # Create a subplot with 2x2 grid
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Cities with Paths", "Ordered Distances", "VAT Distances", "IVAT Distances"])

    # Add cities scatter plot (first subplot)
    fig.add_trace(
        go.Scatter(x=all_cities[:, 0], y=all_cities[:, 1], mode='markers', name='Cities'),
        row=1, col=1
    )

    # Add VAT path
    vat_path_coords = np.array([all_cities[i] for i in vat_path])
    fig.add_trace(
        go.Scatter(x=vat_path_coords[:, 0], y=vat_path_coords[:, 1],
                   mode='lines', name='VAT Path', line=dict(color='green')),
        row=1, col=1
    )

    # Add ACO path
    aco_path_coords = np.array([all_cities[i] for i in aco_path])
    fig.add_trace(
        go.Scatter(x=aco_path_coords[:, 0], y=aco_path_coords[:, 1],
                   mode='lines', name='ACO Path', line=dict(color='red')),
        row=1, col=1
    )

    # Add HS-ACO path
    ivat_path_coords = np.array([all_cities[i] for i in ivat_path])
    fig.add_trace(
        go.Scatter(x=ivat_path_coords[:, 0], y=ivat_path_coords[:, 1],
                   mode='lines', name='IVAT Path', line=dict(color='purple')),
        row=1, col=1
    )

    # Add Random Distances heatmap (second subplot)
    # For large distance matrices, use go.Heatmap with improved performance settings
    fig.add_trace(
        go.Heatmap(
            z=distances,
            colorscale='Viridis',
            # Performance optimization settings
            zsmooth='fast',  # 'fast' is another option
            hoverongaps=False,
            showscale=True,
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            z=vat_dist,
            colorscale='Viridis',
            # Performance optimization settings
            zsmooth='fast',
            hoverongaps=False,
            showscale=True,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=ivat_dist,
            colorscale='Viridis',
            # Performance optimization settings
            zsmooth='fast',
            hoverongaps=False,
            showscale=True,
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="City Paths and Distance Matrices",
        height=600,
        width=600,
        showlegend=True,
        uirevision='constant',
        hovermode='closest',
        # Legend configuration
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.05,  # Position above the plots
            xanchor="center",
            x=0.5,  # Center it horizontally
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
            bordercolor="Black",
            borderwidth=1
        )
    )

    # Show the combined figure
    fig.show()


def vat_scaling():
    for city_exp in range(1, 4):
        city_count = 2**city_exp
        print(f"City count: {city_count**2}")
        all_cities = circle_random_clusters(n_clusters=city_count, n_cities=city_count)
        matrix_of_pairwise_distance = pairwise_distances(all_cities)
        # Scramble the order to ensure we sort it!
        cols = np.arange(len(all_cities), dtype="int")
        rand_col_order = np.random.permutation(cols)
        matrix_of_pairwise_distance = matrix_of_pairwise_distance[:, rand_col_order][rand_col_order, :]
        # Cluster using our IVAT
        t0 = time.time()
        ordered_matrix2 = compute_merge_sort_dissimilarity_matrix(matrix_of_pairwise_distance)
        t1 = time.time()
        # Cluster using the library IVAT
        ordered_matrix = compute_ordered_dis_njit(matrix_of_pairwise_distance)
        t2 = time.time()

        # Ensure all values are equal!
        assert np.all(ordered_matrix == ordered_matrix2)
        # assert np.all(observation_path == observation_path2)


        # Print the results
        print(f"IVAT-merge time: {t1-t0:.4f} seconds")
        print(f"IVAT-lib time: {t2-t1:.4f} seconds")


if __name__ == "__main__":
    # main()
    vat_scaling()
