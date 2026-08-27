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

---

## Result 2, continued — open-set detection for free

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Every class rule is an explicit membership function, so the model knows not
  just how strongly each class fires but how strongly *anything* fires
* The **complement** of every rule's aggregate is automatically an *"unseen"*
  rule — one scalar knob ($\theta$), no second model to train
* Tested the hard way — leave-one-class-out on Glass: hide a whole class,
  score how well the complement rule flags it as unseen
* Best operating point ($\theta = 0.8$): **70% detection at 55% false alarms**
  ($J = +0.154$); the band from $\theta = 0.5$–$0.8$ stays within
  $J \in [+0.119, +0.154]$, so the knob is forgiving to tune

| Method | Detection | False alarm | $J$ | Separate model? |
|---|---:|---:|---:|:---:|
| **Complement rule (this work)** | 0.491 | 0.362 | +0.129 | **no** |
| Isolation Forest | 0.493 | 0.317 | +0.176 | yes |
| One-class SVM | 0.399 | 0.288 | +0.111 | yes |

* Comparable to purpose-built detectors — **no separation in the table beats
  its own error bar** — while riding entirely on rules the model already has
* Caveat, stated plainly: Glass is 214 samples / 6 classes, a stress test, not
  a demonstration — the bigger-scale run (BETH host telemetry) is proposed work

</div>
<div style="flex: 1; padding: 10px;">

![anomaly-sweep](img/04-anomaly-sweep.png)

*Detection vs. false-alarm rate as the anomaly boost $\theta$ sweeps — the
shaded band is $J$, and it stays wide and flat across most of the range.*

</div>
</div>
