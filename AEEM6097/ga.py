import typing

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from AEEM6097.ivat_tsp import circle_random_clusters

N_SOLUTION_DECK = 24
# N_NEW_SOLNS = N_SOLUTION_DECK // 3
N_GENERATIONS = 4*N_SOLUTION_DECK

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
        for exchange in range(max(1, (N_GENERATIONS - i_generation)//3)):
            r0 = np.random.randint(low=1,high=len(pick_soln))
            r1 = np.random.randint(low=1,high=len(pick_soln))
            p0 = pick_soln[r0]
            p1 = pick_soln[r1]
            pick_soln[r0] = p1
            pick_soln[r1] = p0
        # Check performance.
        pick_soln[0] = f(pick_soln[1:])
        # Append to the end of the array, sort and chop
        soln_deck = np.vstack([soln_deck,pick_soln])
    return soln_deck[0,0], soln_deck[0,1:], best_soln


def plot_ga_results(distances: np.ndarray, rand_order: np.ndarray, optimal_order: np.ndarray, soln_history):
    rand_order = np.int32(rand_order)
    optimal_order = np.int32(optimal_order)
    # Create a subplot with 2x2 grid
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Solution History", "", "Random Distances","GA Distances"])

    # Add cities scatter plot (first subplot)
    fig.add_trace(
        go.Scatter(x=np.r_[0:len(soln_history)], y=soln_history, mode='lines', name='Performance'),
        row=1, col=1
    )

    # Add Random Distances heatmap (second subplot)
    # For large distance matrices, use go.Heatmap with improved performance settings
    fig.add_trace(
        go.Heatmap(
            z=distances[:,rand_order][rand_order,:],
            colorscale='Viridis',
            # Performance optimization settings
            zsmooth='fast',  # 'fast' is another option
            hoverongaps=False,
            showscale=True,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=distances[:,optimal_order][optimal_order,:],
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
        title_text="Simulated Annealing Ordering",
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
        return permuted_distances[x0, :][:, x0].diagonal(offset=1).sum()

    ga_dist, ga_order, best_distances = ga_solve_permutation(f, cols)
    print(f"Random distance={permuted_distances.diagonal(offset=1).sum():.2f}, "
          f"GA dist={ga_dist:.2f}")
    plot_ga_results(distances, permuted_distances, ga_order, best_distances)

if __name__ == "__main__":
    main()