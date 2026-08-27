# The Tribble Approach

**Scott Phillips**

Advisor: Dr Kelly Cohen

---

## What is Tribble?

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Structure-first FIS: read membership functions and rules off the data's own
  structure — no GA, no gradient descent
* Deterministic, closed-form construction — same data, same model, every run
* Scales from hundreds to hundreds of thousands of samples, in seconds

</div>
<div style="flex: 1; padding: 10px;">

![pipeline-roadmap](img/01-pipeline-roadmap.png)

</div>
</div>

---

## Result 1 — mergeVAT: cluster structure at scale

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Classical VAT: $O(N^3)$. mergeVAT: exact $O(N^2)$ reorder — same result, not
  an approximation
* **673× faster** at $N=1{,}024$; measured to **135,000 points**; ~65 MB flat
  memory
* NASA shuttle-reentry telemetry, 58,000 samples: **~4 days → ~1 minute**
* Fast *and* exact together: rebuilding the model is cheap enough to be
  routine — **repeatable construction doubles as a verification check**, not
  just faster training

</div>
<div style="flex: 1; padding: 10px;">

![complexity-fit](img/03-complexity-fit.png)

</div>
</div>

---

## Result 2 — Fast, interpretable FIS synthesis

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* Answer-first construction: $K$ rules for $K$ classes — no grid, no search

| Dataset | Rules | Accuracy / $R^2$ | Train time |
|---|---:|---:|---:|
| PhiUSIIL (235K × 54) | 2 | 0.997 ± 0.001 | 0.28 s |
| RT-IOT2022 (123K × 82, 12-class) | 12 | 0.927 ± 0.002 | 37.4 s |
| Concrete (regression) | 3 buckets | $R^2=$ 0.861 ± 0.026 | seconds |

* Free: the rule-base complement is automatically an open-set / anomaly
  detector, at no extra training cost

</div>
<div style="flex: 1; padding: 10px;">

![mog-classification](img/04-mog-classification.png)

</div>
</div>

---

## Result 3 — open-set detection, for free, at scale

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* One scalar knob ($\theta$) — no second model to train
* **BETH host telemetry, 1.14M rows, trained on benign traffic only:**
  **99.3% detection at 15% false alarms** ($J=+0.843$)

| Method | Detection | False alarm | $J$ |
|---|---:|---:|---:|
| **Complement rule (this work)** | 0.993 | 0.150 | +0.843 |
| One-class SVM | 0.993 | 0.152 | +0.841 |

* Parity with a one-class SVM — no separate model required
* Mechanism visualized on Glass (right): $\theta$ trades detection against
  false alarms smoothly

</div>
<div style="flex: 1; padding: 10px;">

![anomaly-sweep](img/04-anomaly-sweep.png)

</div>
</div>
