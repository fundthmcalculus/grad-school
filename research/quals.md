# PhD Quals

_Scott Phillips_

Advisor: Dr Kelly Cohen

---

# About Me

* BSME 2016 University of Cincinnati
* First year PhD student under Dr Cohen
* 10 years of industry experience: P&G, consulting, VC-startup life
* Now the `Vice Nerd` of Nexigen.

---

# NAFIPS Paper 1: Utilization of VAT for Hot-start of TSP Solutions

---

# VAT Background & Limitations

* Visual Assessment for Tendency (VAT) is a method for cluster identification pioneered by Bezdek
* It converts, usually via the _L2-norm_, an $N \times M$ matrix of samples into an $N \times N$ dissimilarity matrix $D$
* It permutes the matrix to minimize the distances off the principal diagonal – Minimum Spanning Tree (MST)
* The core algorithm is a greedy one, similar to Prim's Algorithm for MSTs
* It is computationally expensive, $O(N)=N^3$

---

# ACO Background & Limitations

* Ant Colony Optimization is a stochastic optimization technique used for combinatorics, commonly with the Traveling Salesperson Problem (TSP)
* It doesn’t guarantee finding the “best” solution, but often finds a “good enough” solution
* It is trivially parallelizable – important on multicore processors and GPUs

* It does not require the cost function to continuous, or differentiable, only comparable
* It is susceptible to initialization issues, since it is not guaranteed to find the local optima on a given attempt (unlike gradient descent)
* Having a good initial guess, a “hot-start” can greatly reduce the convergence time.

---

# The Connection

* The dissimilarity matrix $D$, and the optimized VAT matrix $D'$ are symmetric permutations of rows and columns.
* It has been proven that the MST provides an upper bound on the length of the optimal tour:

$T_{best} \le 2T_{MST}$
> An intuitive tour is to visit the permuted cities in $D'$ sequentially, then wrap back from city $N$ to $1$.
---

# Example - Circular Cities

* A constructed dataset with obvious structure, clusters, and an analytic nearly optimal tour length
* A large circle with smaller circular clusters distributed evenly around the perimeter
* Optimal tour length approximation:

$T_{optimal} = P_{polygon} + N_{cities}P_{city} - N_{cities}D_{city}$

$D_{polygon} > D_{city}$

---
layout: image-right
image: ./image.png
backgroundSize: contain

# Initial Performance Observation - 256

![clusterPaths](quals/image.png)

|Method |Time [s]|Distance|Change|
|-------|--------|--------|------|
|Optimal|0.00    |289     |100%  |
|Random |0.00    |10,074  |3500% |
|VAT    |0.35    |408     |140%  |
|IVAT   |0.35    |281     |97%   |
|HS-ACO |4.95    |408     |140%  |
|ACO    |4.10    |1592    |550%  |

> Unfortunately, IVAT mutates the matrix, making it unsuitable for hot-starting

---
layout: image-right
image: ./image-1.png
backgroundSize: contain

# Larger Scale - 2048

![largerClusterPaths](quals/image-1.png)

|Method |Time [s]|Distance|Change |
|-------|--------|--------|-------|
|Optimal|       0|   394  |   100%|
|Random |       0|78,104  |19,829%|
|VAT    |     196|   582  |   150%|
|HS-ACO |     543|   582  |   150%|
|ACO    |     258|24,723  |  6300%|

---

# Refinement

* The hot-start ACO ends up a little cleaner in some cases
![alt text](quals/image-2.png)

---

# Can ACO approximate VAT? - Somewhat

![alt text](quals/image-3.png)

* VAT - based on Prim's (greedy) algorithm
* ACO MST - often Broder's algorithm

> A naive permutation method which minimizes the primary diagonals tends to produce perpendicular lines

![alt text](quals/image-4.png)

---

# ACO MST - Scaling

|Column 1|Column 2|Column 3|Column 4|
|--------|--------|--------|--------|
|  ![alt text](image-10.png)    |  |  ![alt text](image-5.png)  |    |  ![alt text](image-6.png)      |
|        | 50x    2x   |  230x 3x      |        |
|    ![alt text](image-7.png)    |  | ![alt text](image-8.png)      |        |   ![alt text](image-9.png)     |
|     128 4x   |  512      | 6.25x       |  800      |
|        |        |        |        |

---

# Conclusions and Future Work

* VAT provides a great initial guess to solving TSP problems with ACO
* Permutation methods with ACO are not effective
* ACO MST methods show promise, but need further development