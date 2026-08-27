# The Tribble Approach

**Scott Phillips**

Advisor: Dr Kelly Cohen

---

## What is Tribble?

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* A structure-first pipeline for building Fuzzy Inference Systems (FIS)
* Core idea — **structure before search**: recover the clusters, hierarchy, and
  density already in the data, then read the membership functions and rules off
  it, instead of paying for stochastic search (GA / gradient descent / ANFIS)
* Named for the *Star Trek* tribble: a FIS left to grow naively reproduces its
  rule base until it fills the room — I want the *good* version of that trick,
  scaling cleanly from hundreds to hundreds of thousands of samples
* No stochastic search on the critical path — training is closed-form and
  measured in seconds, not generations

</div>
<div style="flex: 1; padding: 10px;">

![pipeline-roadmap](img/01-pipeline-roadmap.png)

</div>
</div>

---

## Result 1 — mergeVAT: cluster structure at scale

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Classical VAT (cluster-tendency visualization) is $O(N^3)$ — unusable past a
  few thousand points
* mergeVAT reorders the same matrix exactly, in $O(N^2)$, with an in-place
  memory scheme
* **673× faster** than the classical reference at $N=1{,}024$ (10 seeds,
  workstation)
* Measured out to **135,000 points**; a matrix-free variant holds memory flat
  at ~65 MB regardless of $N$
* NASA shuttle-reentry telemetry, 58,000 samples: classical VAT extrapolates
  to **~4 days**; mergeVAT runs it in **~1 minute**

</div>
<div style="flex: 1; padding: 10px;">

![complexity-fit](img/03-complexity-fit.png)

*Measured reorder time vs. reference $N^2$, $N^2\log N$, $N^3$ curves — mergeVAT
tracks quadratic, not cubic.*

</div>
</div>

---

## Result 2 — Fast, interpretable FIS synthesis

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Build rules **answer-first**: fit a per-feature Gaussian mixture per output
  class, then combine — no grid, no GA, no gradient descent
* Rule count is $K$ (one per class), not exponential in the number of features

| Dataset | Size | Rules | Accuracy / $R^2$ | Train time |
|---|---:|---:|---:|---:|
| PhiUSIIL (phishing) | 235K × 54 | 2 | 0.997 ± 0.001 | 0.28 ± 0.02 s |
| RT-IOT2022 (12-class) | 123K × 82 | 12 | 0.927 ± 0.002 | 37.4 ± 0.6 s |
| Concrete (regression) | 1,030 × 8 | 3 buckets | $R^2=$ 0.861 ± 0.026 | seconds |

* A tuned Random Forest still edges out accuracy on RT-IOT2022 (0.999) — the
  win here is **12 readable rules, trained in under 40 seconds**, not a bigger
  black box
* Free bonus: the complement of every rule is automatically a *"none of the
  above"* rule — open-set / anomaly detection at no extra training cost

</div>
<div style="flex: 1; padding: 10px;">

![mog-classification](img/04-mog-classification.png)

*A real fit on the Glass dataset: per-feature membership functions combine
into one readable IF–THEN rule.*

</div>
</div>
