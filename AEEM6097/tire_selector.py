# Pick the 16 Left tires and 16 Right tires
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

N_tires = 16

# Set random seed for reproducibility
np.random.seed(42)

tires_l = np.random.randint(200, 250, size=N_tires)
tires_r = np.random.randint(210, 260, size=N_tires)

# Create a cost matrix of all possible stagger combinations
cost_matrix = np.zeros((N_tires, N_tires))
for i in range(N_tires):
    for j in range(N_tires):
        cost_matrix[i, j] = abs(tires_l[i] - tires_r[j])

# Use VAT to find the optimal pairings


# Use the Hungarian algorithm to find optimal pairing
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Create the optimal pairings
optimal_pairs = np.column_stack((tires_l[row_ind], tires_r[col_ind]))
pair_stagger = optimal_pairs[:, 0] - optimal_pairs[:, 1]

# Calculate some statistics
avg_stagger = np.mean(np.abs(pair_stagger))
max_stagger = np.max(np.abs(pair_stagger))
min_stagger = np.min(np.abs(pair_stagger))

# Sort pairs by stagger for better visualization
sorted_indices = np.argsort(pair_stagger)
sorted_pairs = optimal_pairs[sorted_indices]
sorted_stagger = pair_stagger[sorted_indices]

# Plotting
plt.figure(figsize=(14, 10))

# Plot 1: Cost matrix heatmap (VAT visualization)
plt.subplot(2, 2, 1)
plt.imshow(cost_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Stagger magnitude')
plt.title('Stagger Cost Matrix')
plt.xlabel('Right Tire Index')
plt.ylabel('Left Tire Index')

# Plot 2: Optimal pairings
plt.subplot(2, 2, 2)
plt.scatter(tires_l, tires_r, alpha=0.3, label='All possible pairs')
for i in range(N_tires):
    plt.plot([tires_l[row_ind[i]], tires_r[col_ind[i]]], [i, i], 'r-', alpha=0.7)
    plt.text(tires_l[row_ind[i]]-3, i, f"{tires_l[row_ind[i]]}", fontsize=8)
    plt.text(tires_r[col_ind[i]]+3, i, f"{tires_r[col_ind[i]]}", fontsize=8)
plt.title('Optimal Tire Pairings')
plt.xlabel('Tire Circumference')
plt.ylabel('Pair Index')
plt.grid(True, alpha=0.3)

# Plot 3: Stagger by pair
plt.subplot(2, 2, 3)
bars = plt.bar(range(N_tires), sorted_stagger)
for i, bar in enumerate(bars):
    if sorted_stagger[i] < 0:
        bar.set_color('blue')
    else:
        bar.set_color('red')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title(f'Stagger by Pair (Avg: {avg_stagger:.2f}, Max: {max_stagger:.2f})')
plt.xlabel('Sorted Pair Index')
plt.ylabel('Stagger (Left - Right)')
plt.grid(True, alpha=0.3)

# Plot 4: Comparison with naive approach
plt.subplot(2, 2, 4)

# Naive approach (sorting both arrays)
tires_l_sorted = np.sort(tires_l)
tires_r_sorted = np.sort(tires_r)
naive_pairs = np.column_stack((tires_l_sorted, tires_r_sorted))
naive_stagger = naive_pairs[:, 0] - naive_pairs[:, 1]
naive_avg_stagger = np.mean(np.abs(naive_stagger))

# Alternative approach (sorting in opposite directions)
tires_l_asc = np.sort(tires_l)
tires_r_desc = np.sort(tires_r)[::-1]
alt_pairs = np.column_stack((tires_l_asc, tires_r_desc))
alt_stagger = alt_pairs[:, 0] - alt_pairs[:, 1]
alt_avg_stagger = np.mean(np.abs(alt_stagger))

width = 0.25
x = np.arange(3)
plt.bar(x, [avg_stagger, naive_avg_stagger, alt_avg_stagger])
plt.xticks(x, ['Optimal', 'Naive\n(Both Sorted)', 'Alternative\n(Opposite Sorted)'])
plt.ylabel('Average Absolute Stagger')
plt.title('Comparison of Pairing Strategies')

plt.tight_layout()
plt.savefig('tire_stagger_analysis.png', dpi=300)
plt.show()

# Print results
print(f"Optimal Pairing Results:")
print(f"Average Stagger: {avg_stagger:.2f}")
print(f"Maximum Stagger: {max_stagger}")
print(f"Minimum Stagger: {min_stagger}")
print("\nOptimal Pairings (Left, Right, Stagger):")
for i in range(N_tires):
    print(f"Pair {i+1}: Left={optimal_pairs[i,0]}, Right={optimal_pairs[i,1]}, Stagger={pair_stagger[i]}")