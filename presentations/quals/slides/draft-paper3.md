# (Future) Paper 3: Fuzzy C Means and Cluster Detail Extraction

* Centroid equation:

$$c_k = {{\sum_x w_k(x)^m x}\over{\sum_x w_k(x)^m}}$$

* Objective function:

$$J(W,C) = \sum_{i=1}^{n} \sum_{j=1}^{c} w_{ij}^{m} \|\vec{x}_i - \vec{c}_j\|^{2}$$

* Weights:

$$w_{ij} = \frac{1}{\sum_{k=1}^{c} \left ( {\|\vec{x}_i - \vec{c}_j\|}\over{\|\vec{x}_i - \vec{c}_k\|} \right )^2}$$



> This method handles points with partial membership in multiple clusters, but it is susceptible to initialization
> issues.


---

# IVAT Initialization

1. Since the VAT provides a permuted list of the rows and columns of $D$, we can use this to identify memberships in
   clusters

2. Use the difference of off-by-1 diagonal of the IVAT matrix to identify the boundaries of each cluster.

3. Sort the trace of the IVAT matrix, and find the point of the maximum change.

4. This is the initial guess for the count of cluster centroids.

5. Look back to the

![img_1.png](./quals/image-21.png)

![img.png](./quals/image-20.png)




---

# Now: Current Research Direction

1. Accelerating Fuzzy C Means methods with gradient-descent optimization

    1. Still subject to initial point selection

2. Utilizing VAT/IVAT for automatic cluster (and cluster centroid) identification

    1. This guarantees we don't initialize FCM with points which have primary membership in the same cluster.

    2. This also provides the initial steps towards 2-OPT check points identification

    3. Automatic cluster counting

3. Mixture of Gaussians (MoG) FIS membership function and rule identification

    1. This is showing promise for orders-of-magnitude speed up in model training

    2. It trains on a phishing dataset with 235K entries to 97% accuracy in 6 seconds

    3. No post-training GD or GA required

    4. It does this with 2 rules and a handful of clauses

    5. It extends to TSK order-1 and order-2 with linear regression parameter estimation.

4. 2D-rotation AND-rule selection

    1. Uniformly distributes rules across possible space

    2. Provides a good initial solution deck for GA/ACO methods

---

# Future: Goal

> Make Training of Fuzzy Inference Systems (FIS) models 1000x faster, whether in time, or in usable scale

1. Preliminary Data Review - VAT/IVAT

2. Initial model skeleton - FCM

3. Membership function selection - MoG

4. Rulebase development - MoG

5. Model refinement - Optimization methods

6. Any suggestions?