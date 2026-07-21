# VAT/IVAT → Membership Function: Alternative Mathematical Frameworks

## Overview

Current approach (single-linkage dendrogram persistence) works for non-convex structure detection but has inherent limitations (chaining, multi-scale gaps). This explores **three parallel mathematical strategies** for extracting membership functions with stronger guarantees, with **Ruspini partitioning** as the target and **prototype-selection** as the fallback.

---

## 1. RUSPINI PARTITIONING FROM ULTRAMETRIC STRUCTURE

### The Promise
A **Ruspini partition** is a special case of fuzzy partition where the membership functions tile the space: they form a **partition of unity** (∑_c μ_c(x) = 1 everywhere) AND each MF is a **unimodal core with linear descent** (triangular or trapezoidal). This gives:

- **Strong metric guarantees**: membership is normalized and structured
- **Linguistic interpretability**: a point's membership across all terms always sums to 1
- **Defuzzification clarity**: any defuzzifier that respects the partition (max, fuzzy argmax, barycentric) works identically
- **Approximation bounds**: multilinear B-spline approximation guarantees on the fuzzy rule base

### Path 1A: Core Extraction → Ruspini Construction

**Steps:**
1. **Extract disjoint fuzzy cores** from the ultrametric hierarchy (single clusters, non-overlapping)
   - For each selected block B with birth h_b, death h_d:
     - Core μ^core_B(x) = max(0, 1 - (D*(x, medoid_B) - h_b) / (h_d - h_b))
     - Clip to 1.0 at h ≤ h_b
   
2. **Compute "support boundary" for each core**
   - Each core has implicit support = {x : D*(x, medoid) < h_d}
   - Find the **fuzzy boundary** where cores transition

3. **Normalize to partition of unity**
   - At each point x, sum all active cores: S(x) = ∑_B μ^core_B(x)
   - Normalize: μ_B(x) := μ^core_B(x) / max(S(x), 1)
   - This preserves unimodality and ensures ∑_B μ_B(x) ≥ 1 (can exceed 1 at boundaries if desired, or cap to 1)

4. **Extract Ruspini parameters** (center, width, overlap coefficients)
   - For each normalized μ_B:
     - peak_B = argmax_x μ_B(x) (located at medoid or its neighborhood)
     - width_B = the distance scale at which μ_B drops to 0.5
     - slope_B = (1 - 0) / width_B (left/right may differ for asymmetric cores)
   - Store (peak_B, width_L, width_R, overlap_ij) for reconstruction

**Advantage:** Guarantees partition of unity, direct Ruspini form, linguistically sound.  
**Limitation:** Normalization at boundaries may blend distinct clusters; multi-scale clusters still fight for mass.

### Path 1B: Direct Ruspini from Reachability

**Concept:** Use the VAT/IVAT reachability structure directly to define Ruspini centers and supports.

1. **Seed selection** (find cluster prototypes)
   - Use block birth points (local connectivity extrema in the minimax hierarchy)
   - Or: select k medoids that maximize coverage and separation in D* space

2. **Define membership functions as normalized distances in D* space**
   - For each seed s_c, define a "reachability ramp":
     - μ_c(x) ∝ exp(-α · D*(x, s_c))  for some α > 0
   - Normalize: μ_c(x) := μ_c(x) / ∑_c' μ_c'(x)
   - This is **not Ruspini** but **Ruspini-like** (soft partition of unity)

3. **Extract linguistic parameters**
   - Once normalized, fit a triangular or trapezoidal template to each μ_c
   - Extract (center, left_width, right_width) via least-squares projection onto Ruspini MF shapes

**Advantage:** Clean normalization, exploits ultrametric structure directly, fewer intermediate steps.  
**Limitation:** Exponential mapping is a *fit*, not a derivation; Ruspini extraction loses information from the exponential profile.

---

## 2. PROTOTYPE MEMBERSHIP FUNCTION FRAMEWORK

### The Concept
Instead of hard-coding a single membership function type (e.g., triangular Ruspini), allow the **model builder to specify a prototype MF** and then instantiate it for each cluster using the ultrametric structure.

### Supported Prototype Families

#### 2A. Triangular (Ruspini baseline)
```
μ(x) = {
  0,                                     x < a
  (x - a) / (b - a),                     a ≤ x < b
  1,                                     x = b
  (c - x) / (c - b),                     b < x ≤ c
  0,                                     x > c
}
Parameters: (a, b, c) — left foot, peak, right foot
```
**Instantiation:** a, b, c extracted from the birth/death interval and the dissimilarity profile.

#### 2B. Trapezoidal
```
μ(x) = {
  0,                                     x < a
  (x - a) / (b - a),                     a ≤ x < b
  1,                                     b ≤ x ≤ c
  (d - x) / (d - c),                     c < x ≤ d
  0,                                     x > d
}
Parameters: (a, b, c, d) — left foot, left peak, right peak, right foot
```
**Instantiation:** Core region [b, c] = {x : D*(x, medoid) ≤ h_b}; feet at ±σ·(h_d - h_b) from the core.

#### 2C. Gaussian (smooth approximation)
```
μ(x) = exp(-((x - μ₀) / σ)²)
Parameters: (μ₀, σ) — center and width
```
**Instantiation:** μ₀ = medoid in D* space; σ derived from birth/death interval or variance within block.

#### 2D. Sigmoid (asymmetric, soft step)
```
μ(x) = 1 / (1 + exp(-k·(x - x₀)))
Parameters: (x₀, k) — inflection point and steepness
```
**Instantiation:** Useful for "at least as close as this threshold" semantics; extract from persistence curves.

#### 2E. Exponential Decay
```
μ(x) = exp(-λ·d(x, medoid))
Parameters: λ — decay rate
```
**Instantiation:** λ derived from the D* distance scale; recovers the "soft reachability" interpretation.

### Extraction Algorithm for Each Prototype

**Input:** 
- Block B with medoid m_B, birth h_b, death h_d
- Selected prototype class (e.g., Triangular)
- Points X in/around block B

**Generic extraction pipeline:**
```
1. Compute dissimilarity ramp r(x) = D*(x, m_B)
2. Normalize to unit interval: r_norm(x) = (r(x) - r_min) / (r_max - r_min)
3. Identify core: core_points = {x : r_norm(x) ≤ h_b / (h_d - h_b)}
4. Fit prototype to (r_norm, μ_target) where:
   - μ_target(x) = 1 if x in core_points
   - μ_target(x) = 1 - r_norm(x) if x in support but outside core (linear descent)
   - μ_target(x) = 0 if x outside support
5. Solve least-squares: min ∑_x (μ_fit(x, params) - μ_target(x))²
6. Return fitted parameters
```

### Configuration: Model Builder Selection

**User-facing API:**
```python
mf_extractor = VAT_MF_Extractor(
    prototype='triangular',  # or 'trapezoidal', 'gaussian', 'sigmoid', 'exponential'
    prototype_params={
        'triangular': {'foot_scale': 1.5},  # support = core ± foot_scale·σ
        'gaussian': {'sigma_ref': 'birth_death_width'},  # or 'block_variance'
    },
    core_selection='persistence_gap',  # or 'coverage', 'manually_specified'
    ruspini_normalize=True,  # post-fit: normalize to partition of unity
)

# Then:
membership_functions = mf_extractor.fit(D_star, blocks, X_features)
```

---

## 3. METRICS-FIRST MEMBERSHIP EXTRACTION

### Motivation
Current approach: block → persistence → MF shape.  
**Alternative:** block → metric properties → MF (decouples the metric understanding from the MF parametrization).

### 3A. Density-Informed Membership (Via Chernoff Information)

**Key insight:** In the ultrametric (MST) space, density of a cluster is inversely proportional to the average intra-cluster MST edge weight.

```
For block B:
  density_B = |B| / (mean_edge_weight_within_B)
  
Membership ramp:
  μ_B(x) = {
    1,                           if x in core of B
    max(0, 1 - (D*(x,m) / d_B)), if x in support of B
    0,                           otherwise
  }
  where d_B is a density-scaled threshold
```

**Advantage:** Automatically scales the MF to match the cluster's intrinsic density; two clusters at different σ have different fall-off speeds.

### 3B. Separation-Aware Membership (Via Metric Margin)

**Key insight:** The gap between cluster B and its nearest cluster neighbor C determines the MF boundary sharpness.

```
For clusters B, C with metrics d_B, d_C:
  separation(B, C) = min{D*(x, m_B) : x in C} - max{D*(x, m_B) : x in B}
  
Membership function steepness:
  α_B = separation(B, nearest_neighbor(B)) / (h_d - h_b)_B
  
Ramp: μ_B(x) = max(0, 1 - α_B · (D*(x, m_B) - h_b))
```

**Advantage:** Sharp boundaries where clusters are well-separated; gradual blend where overlap is unavoidable. Respects the geometry.

### 3C. Signature-Based Prototype Matching

**Idea:** Extract a **metric signature** from each block and use it to select the *best-fitting* prototype class automatically.

**Signature components:**
1. **Cohesion** = (h_d - h_b) / mean{D*(x, y) for x, y in B}
   - High cohesion → clusters are tight; prefer Gaussian or sharp triangular
   - Low cohesion → clusters are diffuse; prefer trapezoidal with wide support
   
2. **Symmetry** = ∑{D*(x, m_B) for x < m_B} / ∑{D*(x, m_B) for x > m_B}
   - Near 1.0 → symmetric; use symmetric MF
   - ≠ 1.0 → asymmetric; use exponential or sigmoid (asymmetric)
   
3. **Concentration** = (# points with D*(x, m) < 0.5·(h_d - h_b)) / |B|
   - High → tight core, wide tails; trapezoidal
   - Low → gradual gradient; Gaussian or exponential

**Selection rule (example):**
```python
if cohesion > 0.7 and concentration > 0.6:
    prototype = 'triangular'
elif cohesion < 0.5:
    prototype = 'trapezoidal'
elif 0.8 < symmetry < 1.2:
    prototype = 'gaussian'
else:
    prototype = 'exponential'
```

---

## 4. ALTERNATIVE DENDROGRAMS (BEYOND SINGLE-LINKAGE)

### Motivation
Single-linkage has known pathologies (chaining). Other linkage methods may extract different hierarchical structure with different metric properties. Explore them as alternative "VAT bases."

### Options & Their Metric Guarantees

| Linkage | Distance | Metric? | Ultrametric? | Advantage | Disadvantage |
|---------|----------|---------|-------------|-----------|--------------|
| Single | min{d(x,y)} | ✓ | ✗ (violates transitivity) | Non-convex clusters | Chaining |
| Complete | max{d(x,y)} | ✓ | ✓ | Tighter clusters, more robust | Cuts small outliers too aggressively |
| Average (UPGMA) | mean{d(x,y)} | ✓ | ✓ | Compromise | Still hierarchical but less stable |
| Weighted (WPGMA) | (h_a + h_b)/2 | ✓ | ✓ | Theoretically sound | Loss of information from MST |
| Ward | min{increase in variance} | ~ | ~ | Convex clusters balanced | **Not metric-preserving** |

**Key finding:** Only **single-linkage** preserves the MST (and thus the minimax distances). Complete/average linkage reinterpret the same dissimilarity matrix under different merge criteria.

### Exploration Path: Chained-SL + Dendrogram Repair

Rather than abandon SL, **repair its output**:

1. **Build SL dendrogram as usual**
2. **Detect chaining** via topological inspection:
   - If a merge includes a node with degree >> 2 in the MST, flag it as a potential chain
3. **Replace the merge** with a "constrained" variant:
   - Instead of linking the two worst clusters by their closest pair, link them by their *core centroids* (in the minimax space)
   - Or: insert a constraint that any merge at height h requires both clusters to have internal density ≥ h
4. **Re-estimate the hierarchy** from the repaired merge sequence

**Outcome:** Recovers some of ConiVAT's robustness without the supervised metric-learning overhead.

---

## 5. FEATURE-SPACE PARAMETER EXTRACTION

### The Challenge
VAT/IVAT work in dissimilarity space, but membership functions must be **executable in the original feature space** (or at least interpretable back to features).

### Path 5A: Prototype Projection onto Features

**For vector data X ∈ ℝ^d:**

1. **Compute D** from X (e.g., Euclidean distances)
2. **Extract VAT hierarchy and blocks**
3. **For each block B** with medoid m_B (an index):
   - Extract the dissimilarity-space MF as before: μ_B(x) = f(D*(x, x_m_B))
   - Fit a **surrogate MF in feature space**:
     - μ̃_B(x_feat) = g(‖x_feat - x_m_B‖, Σ_B)
     - where Σ_B is the within-block covariance, and g is Gaussian/Mahalanobis
   - Optimize (least-squares on a grid sample):
     - min ∑_{x_i in B} (μ_B(x_i) - μ̃_B(x_feat_i))²
   - Return (center, covariance, MF_type) in feature coordinates

2. **Execute the fuzzy rule base** in feature space using the fitted parameters

**Advantage:** Parametric MFs executable on new data without recomputing D.  
**Limitation:** Surrogate may not perfectly preserve ultrametric properties; Mahalanobis distance may overfit on small blocks.

### Path 5B: Ruspini Grid Inference (Feature-Space Partitioning)

**Goal:** Discretize the feature space into a **Ruspini linguistic grid** and assign membership to each grid cell based on the ultrametric hierarchy.

1. **Identify feature ranges** from data (or from domain knowledge)
2. **Partition each feature into linguistic bins** (e.g., "low", "medium", "high" for each variable)
3. **For each grid cell** (combination of bins):
   - Compute a representative point (cell center)
   - Look up its block membership in the D* hierarchy
   - Assign that cell the corresponding MF value(s)
4. **Construct a piecewise-constant or bilinear interpolation** MF over the grid

**Advantage:** Directly Ruspini, human-interpretable ("if x1 is low and x2 is medium, then cluster A has membership 0.8"), executable without D.  
**Limitation:** Curse of dimensionality; loses smoothness; grid resolution is a tuning parameter.

### Path 5C: Kernel Density Re-estimation (For Relational Data)

**If the data is relational** (no natural feature coordinates), extract **kernelized membership**:

1. **Fit RBF/Gaussian kernels** to each block's points (unsupervised kernel density estimation)
2. **Store kernel centers and covariances** as the Ruspini parameters
3. **Define membership** as the kernel mixture: μ_B(x) ∝ ∑_{k in B} K(x, center_k, Σ_k)
4. **Normalize** to partition of unity

**Advantage:** Works with any dissimilarity; no feature assumption.  
**Limitation:** Adds another layer of fitting; two fits (dissimilarity → blocks, then blocks → kernels) compound approximation error.

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Framework Setup (1–2 weeks)
- [ ] Define base class `VAT_MF_Extractor` with pluggable prototype strategies
- [ ] Implement prototype classes: Triangular, Trapezoidal, Gaussian, Sigmoid, Exponential
- [ ] Build a `PrototypeSelector` that uses metric signatures to auto-choose
- [ ] Unit tests: ensure each prototype correctly parametrizes synthetic known blocks

### Phase 2: Ruspini Extraction (1–2 weeks)
- [ ] Path 1A: Core extraction → normalization → Ruspini parameters
- [ ] Path 1B: Direct reachability ramp + Ruspini fitting
- [ ] Validate partition of unity property (∑ μ_c(x) ≥ 1 everywhere; ≈ 1 in regions with single dominant cluster)
- [ ] Compare to baseline on synthetic data (two_gaussians, rings, varying_density)

### Phase 3: Feature-Space Bridge (1–2 weeks)
- [ ] Path 5A: Surrogate MF fitting for vector data
- [ ] Path 5B: Ruspini grid inference (if domain has natural feature semantics)
- [ ] Test on real-world datasets (if available) or synthetic + projection experiments

### Phase 4: Prototype Selection & Auto-Tuning (1–2 weeks)
- [ ] Implement metric signatures (cohesion, symmetry, concentration)
- [ ] Build the auto-selector heuristic
- [ ] A/B test: auto-selected prototype vs. manual selection vs. oracle (ground-truth cluster generator)
- [ ] Tune the signature thresholds on the synthetic battery

### Phase 5: Dendrogram Alternatives (1–2 weeks, optional)
- [ ] Implement complete-linkage and UPGMA extraction
- [ ] Test chaining robustness: bridged_gaussians and other problematic cases
- [ ] Implement SL dendrogram repair (chaining detection + core-centroid merge)
- [ ] Measure cost vs. ConiVAT: Does repair recover robustness without metric learning?

### Phase 6: Validation & Baselines (1–2 weeks)
- [ ] Run the battery against the new prototype-selection approach
- [ ] Compare Ruspini MFs (Path 1A/1B) against baseline persistence MFs on ARI, coverage, defuzzification
- [ ] Benchmark feature-space surrogates: how well do they preserve dissimilarity-space semantics?

---

## 7. SUCCESS CRITERIA

### For Ruspini Path (1A/1B)
- ✓ Membership functions form a partition of unity (∑_c μ_c(x) = 1 everywhere, within floating-point error)
- ✓ ARI on concentric_rings ≥ 0.95 (maintain non-convex win)
- ✓ ARI on bridged_gaussians ≥ 0.95 (improve chaining robustness without ConiVAT)
- ✓ Honest defuzzification: membership → decision is unambiguous (no saturation artifacts)

### For Prototype-Selection Path (2A–E)
- ✓ Auto-selector achieves ARI within 0.05 of oracle (best-hand-tuned prototype)
- ✓ Metric signatures are stable: same block → same prototype across multiple random draws
- ✓ No prototype class needed manual intervention for >90% of blocks in the battery

### For Feature-Space Bridge (5A–C)
- ✓ Surrogate MF in feature space reproduces dissimilarity-space MF with L2 error < 0.1 (on held-out validation points)
- ✓ Can execute the inferred MFs on new, out-of-sample data (e.g., new test points not in training)
- ✓ Ruspini parameters are interpretable: (center, widths) align with human-readable cluster descriptions

### Cross-Cutting
- ✓ Avoid introducing new tuning parameters; anything not discoverable from D* or the data should be automatically set
- ✓ Code is modular: prototype swapping, core-selection strategy swapping, and linkage swapping should each be 1-line configuration changes
- ✓ Comprehensive test suite and reproducible results on synthetic + (eventually) real data

---

## 8. OPEN QUESTIONS TO RESOLVE IN EXPERIMENTS

1. **Ruspini normalization at cluster boundaries:** If ∑_c μ_c(x) > 1 in the overlap region, which normalization scheme (max, sum-to-1, clipping) best preserves cluster structure?

2. **Multi-scale prototype selection:** For varying_density, does the metric signature approach correctly identify that tight and diffuse clusters need different prototype shapes? Or do they need different shapes *and* different parameterization?

3. **Chaining in alternative dendrograms:** Does complete-linkage dendrogram repair succeed on bridged_gaussians, and at what cost to non-convex structure (rings)?

4. **Feature-space surrogate fidelity:** For high-dimensional data, does Mahalanobis distance (5A) preserve the ultrametric ranking well enough? Need empirical test on UCI datasets.

5. **Automatic knee detection vs. Ruspini extraction:** If Ruspini forces a partition structure, does that change how many clusters we should extract? (Ruspini implies full coverage; does it force over-segmentation?)

---

## References & Anchors

- **Ruspini Partitions:** Ruspini (1969), "A new approach to clustering." IEEE Trans. Syst., Man, Cybern.
- **Fuzzy Inference from Partitions:** Korner & Klawonn (2010), "Cosine fuzzy partitions of unity."
- **Metric Signatures:** Inspired by cluster validity indices (Davies-Bouldin, Calinski-Harabasz); adapted to ultrametric context.
- **Feature-Space Surrogates:** Related to the "back-projection" problem in kernel clustering; e.g., kernel k-means.
- **Dendrogram Repair:** Inspired by the consensus-clustering and agglomerative-hierarchical-robustness literature (Carlsson & Mémoli).

