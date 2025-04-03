import numpy as np
import plotly.express as px
from pyclustertend import compute_ivat_ordered_dissimilarity_matrix

N_CITIES_CLUSTER = 100
N_CLUSTERS = 20

CLUSTER_DIAMETER = 3
CLUSTER_SPACING = 8*CLUSTER_DIAMETER


def random_cities(center_x, center_y) -> np.ndarray:
    # Randomly distribute cities in a uniform circle?
    return -CLUSTER_DIAMETER / 2.0 + CLUSTER_DIAMETER * np.random.standard_normal(size=(N_CITIES_CLUSTER, 2)) + np.array([center_x, center_y])

# TODO - Distribute along a great circle
def circle_random_clusters() -> np.ndarray:
    city_locations = np.zeros(shape=(0,2))
    for theta in np.linspace(0, 2 * np.pi, N_CLUSTERS+1):
        cx = CLUSTER_SPACING * np.cos(theta)
        cy = CLUSTER_SPACING * np.sin(theta)
        city_locations = np.concatenate((city_locations, random_cities(cx, cy)), axis=0)
    return city_locations


def distance_matrix(pt: np.ndarray) -> np.ndarray:
    # [X1 Y1] * [X1 X2 ...
    # [X2 Y2]   [Y1 Y2 ...
    D = np.zeros(shape=(pt.shape[0], pt.shape[0]))
    for ij in range(pt.shape[0]):
        for jk in range(ij+1,pt.shape[0]):
            D[ij, jk] = np.linalg.norm(pt[ij, :] - pt[jk, :])
    # It's a symmetric matrix!
    D += D.transpose()
    return D


def main():
    print("Configuring random")
    all_cities = circle_random_clusters()
    # Compute all distances
    distances = distance_matrix(all_cities)
    print("Distance-shape",distances.shape)
    print("Min, Max Distances", np.min(distances), np.max(distances))
    # Use the IVAT technique to organize
    ivat_dist: np.ndarray = compute_ivat_ordered_dissimilarity_matrix(all_cities)
    # TODO - Identify the permuted order
    permute_trip = []
    print("Permutation order: ", permute_trip)
    fig = px.imshow(distances)
    fig.update_layout(title="Random Distances")
    fig.show()

    fig2 = px.imshow(ivat_dist)
    fig2.update_layout(title="IVAT Distances")
    fig2.show()
    # TODO - Compute TSP optimized distance, ignoring to-start route.
    ivat_dist = np.sum(ivat_dist.diagonal(offset=1))
    print("Random Distance=",distances[0,:].sum(), "IVAT Distance=",ivat_dist)

    fig0 = px.scatter(x=all_cities[:, 0], y=all_cities[:, 1])
    fig0.update_layout(title="Cities")
    fig0.show()


if __name__ == "__main__":
    main()