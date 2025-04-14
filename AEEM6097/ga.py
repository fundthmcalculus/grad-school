import typing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from AEEM6097.ivat_tsp import circle_random_clusters

N_SOLUTION_DECK = 1
# N_NEW_SOLNS = N_SOLUTION_DECK // 3
N_GENERATIONS = 300

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
        # Randomly pick a solution from the deck, and permute a steadily decreasing number of parameters by generation
        pick_idx = np.random.choice(choice_idx,p=soln_weights)
        pick_soln = soln_deck[pick_idx,:].copy() # Copy before changing, so we don't ruin things!
        # Permute the relevant number of variables
        for exchange in range(max(1, (N_GENERATIONS - i_generation)//3)):
            # Leave the start point alone
            r0 = np.random.randint(low=1,high=len(pick_soln))
            r1 = np.random.randint(low=1,high=len(pick_soln))
            p0 = pick_soln[r0]
            p1 = pick_soln[r1]
            pick_soln[r0] = p1
            pick_soln[r1] = p0
            # Check if this is an improvement
            new_value = f(pick_soln[1:])
            if new_value > pick_soln[0]:
                # Undo this permutation
                pick_soln[r0] = p0
                pick_soln[r1] = p1
        # Check performance.
        pick_soln[0] = f(pick_soln[1:])
        # Append to the end of the array, sort and chop
        soln_deck = np.vstack([soln_deck,pick_soln])
        # Sort by minimum value on the first column, chop to length
        soln_deck = soln_deck[soln_deck[:, 0].argsort()]
        soln_deck = soln_deck[:N_SOLUTION_DECK, :]
        best_soln.append(soln_deck[0, 0])
    return soln_deck[0,0], soln_deck[0,1:], best_soln


def plot_ga_results(distances: np.ndarray, rand_order: np.ndarray, optimal_order: np.ndarray, soln_history):
    rand_order = np.int32(rand_order)
    optimal_order = np.int32(optimal_order)

    # Create a subplot with 2x2 grid
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title('Solution History')
    axes[0, 0].scatter(x=np.arange(len(soln_history)), y=soln_history)
    axes[0,1].set_title('Optimal Distances')
    axes[0,1].imshow(distances, cmap='viridis')
    axes[1,0].set_title('Random Distances')
    rand_distances = distances[rand_order, :][:, rand_order]
    axes[1, 0].imshow(rand_distances, cmap='viridis')
    axes[1,1].set_title('Ordered Distances')
    axes[1,1].imshow(rand_distances[:,optimal_order][optimal_order,:], cmap='viridis')

    fig.show()


def permute_matrix(m: np.ndarray, col_order: np.ndarray) -> np.ndarray:
    # TODO - Handle that the permutation is symmetric.
    pass


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
        x0 = x0.astype("int32")
        repermute_dist = permuted_distances[:, x0][x0,:]
        # Get the central 6 diagonals
        diag_sum = np.float64(0.0)
        for offset in range(0,int(np.sqrt(len(all_cities)))):
            diag_sum += repermute_dist.diagonal(offset=offset).sum()
        return diag_sum

    ga_dist, ga_order, best_distances = ga_solve_permutation(f, cols)
    ga_order = ga_order.astype("int32")
    print(f"Random distance={permuted_distances.diagonal(offset=1).sum():.2f}, "
          f"GA dist={permuted_distances[:,ga_order][ga_order,:].diagonal(offset=1).sum():.2f}")
    plot_ga_results(distances, rand_col_order, ga_order, best_distances)

if __name__ == "__main__":
    main()