---
title: PhD Quals
---

# PhD Quals

**Scott Phillips**

Advisor: Dr Kelly Cohen

---

## About Me

* BSME 2016 University of Cincinnati
* First year PhD student under Dr Cohen
* 10 years of industry experience: P&G, consulting, VC-startup life
* Now the `Vice Nerd` of Nexigen.

---

## Research Goal

> Make the training and development of Fuzzy Inference Systems (FIS) models 1000x faster, whether in time, or in usable scale

1. Preliminary Data Review - [Paper 2: mergeVAT: 58K x 58K in 60 seconds](paper_combined.md)
2. Initial model skeleton - Fuzzy C Means 
3. Membership function selection - Mixture of Gaussians - [Draft Paper 3: VAT/IVAT Direct to Clusters](draft-paper3.md)
4. Rulebase development - Mixture of Gaussians
5. Model refinement - Optimization methods
6. Any suggestions?

---

## Now: Current Research Direction

1. Accelerating Fuzzy C Means methods with gradient-descent optimization
    1. Still subject to initial point selection
2. Utilizing VAT/IVAT for automatic cluster (and cluster centroid) identification
    1. This guarantees we don't initialize FCM with points which have primary membership in the same cluster.
    2. This also provides the initial steps towards 2-OPT check points identification
    3. Automatic cluster counting
3. Mixture of Gaussians (MoG) FIS membership function and rule identification
    1. This is showing promise for order-of-magnitude speed-up in model training
    2. It trains on a phishing dataset with 235K entries to 97% accuracy in 6 seconds
    3. No post-training GD or GA required
    4. It does this with 2 rules and a handful of clauses
    5. It extends to TSK order-1 and order-2 with linear regression parameter estimation.
    6. The model can be trained in a semi-supervised manner, making it easy to incorporate future data into the model.
4. 2D-rotation AND-rule selection
    1. Uniformly distributes rules across possible space
    2. Provides a good initial solution deck for GA/ACO methods

---

# Thank You!

* Dr Kelly Cohen - my advisor and mentor, allowing me to explore topics like this which interest me.
* Jon Salisbury - for support/employment and opening the door for me to do this.
* UC AI / Bio Lab - y'all know what you do. :)
* Hannah Phillips - my wife, you support me and take on so much.
* Dr Phillips, aka _Dad_

---