import typing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from AEEM6097.ivat_tsp import circle_random_clusters

N_SOLUTION_DECK = 1
# N_NEW_SOLNS = N_SOLUTION_DECK // 3
N_GENERATIONS = 10000

def ga_solve_permutation(f: typing.Callable[[np.ndarray], np.float64],
                         x0: np.ndarray) -> tuple[np.float64, np.ndarray, list[np.float64]]:
    """Use a GA to solve for the optimal permutation of the given function input arguments"""
    # Create a solution deck
    soln_deck = np.zeros(shape=(N_SOLUTION_DECK,len(x0)+1), dtype=np.float64)
    soln_weights = np.exp(-np.arange(N_SOLUTION_DECK))
    soln_weights /= np.sum(soln_weights)
    # Randomly permute the input deck for each entry in solution deck
    soln_deck[0,0] = f(x0)
    soln_deck[0,1:] = x0
    for isoln in range(1,N_SOLUTION_DECK):
        x1 = np.random.permutation(x0)
        soln_deck[isoln,0] = f(x1)
        soln_deck[isoln,1:] = x1
    choice_idx = np.arange(N_SOLUTION_DECK)
    best_soln = []
    # For some many iterations
    for i_generation in tqdm(range(N_GENERATIONS), desc="Simulating GA"):
        # Sort by minimum value on the first column, chop to length
        soln_deck = soln_deck[soln_deck[:, 0].argsort()]
        soln_deck = soln_deck[:N_SOLUTION_DECK,:]
        best_soln.append(soln_deck[0,0])
        # Randomly pick a solution from the deck, and permute a steadily decreasing number of parameters by generation
        pick_idx = np.random.choice(choice_idx,p=soln_weights)
        pick_soln = soln_deck[pick_idx,:]
        # Permute the relevant number of variables - TODO exponential decay?
        r0 = np.random.randint(low=1,high=len(pick_soln))
        r1 = np.random.randint(low=1,high=len(pick_soln))
        p0 = pick_soln[r0]
        p1 = pick_soln[r1]
        pick_soln[r0] = p1
        pick_soln[r1] = p0
        # Check performance.
        p_new = f(pick_soln[1:])
        if p_new < pick_soln[0]:
            pick_soln[0] = p_new
        else:
            # If the new solution is worse, then don't change it.
            pick_soln[r0] = p0
            pick_soln[r1] = p1
        # Append to the end of the array, sort and chop
        soln_deck = np.vstack([soln_deck,pick_soln])
    return soln_deck[0,0], soln_deck[0,1:], best_soln


def plot_ga_results(distances: np.ndarray, permuted_distances: np.ndarray, optimal_order: np.ndarray, soln_history):
    optimal_order = np.int32(optimal_order)

    # Create figure with customized layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Solution History plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(len(soln_history)), soln_history, '-', label='Performance')
    ax1.set_title('Solution History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance')
    ax1.legend()

    # True optimal order
    ax2 = fig.add_subplot(gs[0, 1])
    random_heatmap = ax2.imshow(
        distances,
        cmap='viridis',
        aspect='auto',
        interpolation='nearest'
    )
    ax2.set_title('Ideal Distances')
    plt.colorbar(random_heatmap, ax=ax2)

    # Random Distances heatmap (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    random_heatmap = ax3.imshow(
        permuted_distances,
        cmap='viridis',
        aspect='auto',
        interpolation='nearest'
    )
    ax3.set_title('Random Distances')
    plt.colorbar(random_heatmap, ax=ax3)

    # GA Distances heatmap (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    optimal_heatmap = ax4.imshow(
        permuted_distances[:, optimal_order][optimal_order, :],
        cmap='viridis',
        aspect='auto',
        interpolation='nearest'
    )
    ax4.set_title('GA Distances')
    plt.colorbar(optimal_heatmap, ax=ax4)

    # Update overall layout
    plt.suptitle("TSP approximation of VAT", fontsize=16)
    plt.tight_layout()

    fig.show()

    return fig


def main():
    print("Configuring random")
    all_cities = circle_random_clusters()
    # Compute all distances
    distances: np.ndarray = pairwise_distances(all_cities)
    print("Distance-shape", distances.shape)

    # Create a permuted random distance layout, and try to order it.
    permuted_distances = distances.copy()
    cols = np.arange(len(distances), dtype="int")
    rand_col_order = np.random.permutation(cols)
    permuted_distances = permuted_distances[rand_col_order, :][:, rand_col_order]

    def f(x0: np.ndarray) -> np.float64:
        x0 = np.int32(x0)
        permuted_distances_x_x_ = permuted_distances[x0, :][:, x0]
        diag_sum = 0.0
        for ij in range(int(np.sqrt(permuted_distances_x_x_.shape[0]//2))):
            diag_sum += permuted_distances_x_x_.diagonal(offset=ij).sum()
        return diag_sum

    ga_dist, ga_order, best_distances = ga_solve_permutation(f, cols)
    ga_order = np.int32(ga_order)
    print(f"Random distance={permuted_distances.diagonal(offset=1).sum():.2f}, "
          f"GA dist={permuted_distances[ga_order,:][:,ga_order].diagonal(offset=1).sum():.2f}")
    plot_ga_results(distances, permuted_distances, ga_order, best_distances)

if __name__ == "__main__":
    main()