import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyclustertend import compute_ivat_ordered_dissimilarity_matrix, compute_ordered_dissimilarity_matrix
from pyclustertend.visual_assessment_of_tendency import compute_ordered_dis_njit
from sklearn.metrics import pairwise_distances

from AEEM6097.test2 import aco_tsp_solve, plot_convergence

N_CITIES_CLUSTER = 8
N_CLUSTERS = N_CITIES_CLUSTER

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 8*CLUSTER_DIAMETER


def random_cities(center_x, center_y) -> np.ndarray:
    # Randomly distribute cities in a uniform circle?
    theta = np.linspace(0, 2 * np.pi, N_CITIES_CLUSTER+1)
    theta = theta[:-1]
    city_x = np.cos(theta) * CLUSTER_DIAMETER/2.0 + center_x
    city_y = np.sin(theta) * CLUSTER_DIAMETER/2.0 + center_y
    return np.c_[city_x, city_y]


def circle_random_clusters() -> np.ndarray:
    city_locations = np.zeros(shape=(0,2))
    for theta in np.linspace(0, 2 * np.pi, N_CLUSTERS+1):
        if theta == 2*np.pi:
            break
        cx = CLUSTER_SPACING * np.cos(theta)
        cy = CLUSTER_SPACING * np.sin(theta)
        city_locations = np.concatenate((city_locations, random_cities(cx, cy)), axis=0)
    return city_locations


def get_permutation(arr1: np.ndarray, arr2: np.ndarray) -> list[int]:
    # Find on the off-diagonals the correct row exchanges.
    return []


def poly_perimeter(n_sides, r):
    # Compute perimeter of inscribed polygon in circle of radius, r.
    return n_sides* 2 * r * np.sin(2 * np.pi /(2*n_sides))


def start_at_idx(v, x=0):
    # Find index of 0, move to front
    x_idx=np.argwhere(v==x)[0][0]
    return np.concatenate((v[x_idx:], v[:x_idx]), axis=0)


def check_path_distance(distances, order_path, return_to_start=False):
    total_dist = 0.0
    for ij in range(len(order_path)):
        p0 = order_path[ij]
        if ij == len(order_path) - 1:
            if return_to_start:
                total_dist += distances[p0,0]
        else:
            p1 = order_path[ij+1]
            total_dist += distances[p0,p1]
    return total_dist


def main():
    print("Configuring random")
    all_cities = circle_random_clusters()
    # Compute all distances
    distances = pairwise_distances(all_cities)
    print("Distance-shape",distances.shape)
    # Use the IVAT technique to organize
    t0 = time.time()
    ivat_dist, ivat_path, vat_dist, vat_path = compute_ivat_ordered_dissimilarity_matrix(all_cities)
    t1 = time.time()
    vat_path = start_at_idx(vat_path)
    ivat_path = start_at_idx(ivat_path)
    # print("Min, Max Distances", np.min(distances), np.max(distances))
    # print("Min, Max IVAT Distances", np.min(ivat_dist), np.max(ivat_dist))
    # print("Min, Max VAT Distances", np.min(vat_dist), np.max(vat_dist))
    print("Approx Optimum Distance=", N_CLUSTERS*poly_perimeter(N_CITIES_CLUSTER, r=CLUSTER_DIAMETER/2.0)+poly_perimeter(N_CITIES_CLUSTER, r=CLUSTER_SPACING))
    print("Random Distance=", distances[0, :].sum())
    vat_dist_len = vat_dist.diagonal(offset=1).sum() + vat_dist[0, -1]
    print("VAT Distance=", vat_dist_len)
    print("VAT checked Distance=", check_path_distance(distances, vat_path))
    ivat_dist_len = ivat_dist.diagonal(offset=1).sum() + ivat_dist[0, -1]
    print("IVAT Distance=", ivat_dist_len)
    print("IVAT checked Distance=", check_path_distance(distances, ivat_path))
    print(f"IVAT Time: {t1 - t0:.2f}s")
    # TODO - Compute TSP optimized distance, ignoring to-start route.
    t2 = time.time()
    optimal_city_order, optimal_tour_length, tour_lengths = (
        aco_tsp_solve(distances,20,20, hot_start=vat_path,
                      hot_start_length=vat_dist_len,)
    )
    t3 = time.time()
    print(f"ACO Time:{t3 - t2:.2f}s")
    print("ACO Distance=", optimal_tour_length)
    print("ACO order: ", optimal_city_order[0:20])
    print("VAT order: ", vat_path[0:20])
    print("IVAT order: ", ivat_path[0:20])

    plot_convergence(tour_lengths)
    plot_results(all_cities, distances, ivat_dist,ivat_path, vat_dist, vat_path, optimal_city_order)


def plot_results(all_cities, distances, ivat_dist, ivat_path, vat_dist, vat_path, aco_path):
    # Assuming these variables are defined earlier in your code:
    # distances, ivat_dist, all_cities, ivat_path, aco_path

    # Create a subplot with 2x2 grid
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Cities with Paths", "Random Distances", "VAT Distances", "IVAT Distances"])

    # Add cities scatter plot (first subplot)
    fig.add_trace(
        go.Scatter(x=all_cities[:, 0], y=all_cities[:, 1], mode='markers', name='Cities'),
        row=1, col=1
    )

    # Add IVAT path
    ivat_path_coords = np.array([all_cities[i] for i in ivat_path])
    fig.add_trace(
        go.Scatter(x=ivat_path_coords[:, 0], y=ivat_path_coords[:, 1],
                   mode='lines', name='IVAT Path', line=dict(color='blue')),
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


if __name__ == "__main__":
    main()