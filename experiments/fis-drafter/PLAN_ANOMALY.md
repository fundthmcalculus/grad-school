# Plan: FIS activation-monitoring for prompt anomaly / injection detection

The concept was tested (see `FINDINGS_DETECTION.md` Part 2) and it holds up: a
one-class model of a small LM's per-layer activations flags malformed prompts,
the signal survives the matched dose-response control, and the FIS "none of the
above" rule is competitive once its inputs are decorrelated. This is the first
regime in the project that suits the FIS. This plan turns that from a demo into
a defensible result, and points it at the application where it is actually
valuable.

## The pivot that makes it matter

Word-salad detection is a clean *scientific* control but a weak *application*
(who needs to detect shuffled questions?). The strongest detected class was
**instruction-injection at 0.946 AUROC**, and that is a real, current security
problem. Reframe the contribution as:

> An interpretable, one-class monitor over an LM's own activations that flags
> anomalous / adversarial prompts **without needing any attack examples**, and
> that says *which layers* fired — a property an opaque classifier does not have.

"No attack examples" is the selling point: supervised injection detectors need a
labelled attack corpus and generalise poorly to new attacks; a one-class monitor
trained only on benign traffic does not.

## Phase 1 — make the existing result solid (days)

1. **Real injection corpus.** Replace the 12 hand-written injections with a
   public set (e.g. the `deepset/prompt-injections`, `jackhhao/jailbreak-classification`,
   or Lakera Gandalf-style prompts — verify licence and availability first).
   Target ≥300 attacks across families (override, role-play, encoding, payload
   splitting) so per-family AUROC is meaningful.
2. **Benign traffic that looks like traffic**, not just dolly questions —
   include long prompts, code, multi-turn, system-prompted. The one-class model
   must be trained on the *actual* benign distribution or it will flag every
   unusual-but-safe prompt.
3. **Nail the two confounds already found.** (a) The BPE-boundary leak: recompute
   the word-order control with token-count and boundary matched, or drop it as a
   scientific aside and lead with injection. (b) Length/vocabulary: keep the
   surface-only control as a permanent baseline in every table.
4. **Multi-model.** Repeat on SmolLM2-360M, pythia-410m, Qwen2.5-0.5B. A monitor
   that only works on one model is not a method.

## Phase 2 — the FIS's distinctive contribution (the thesis angle)

Mahalanobis matches the whitened FIS on the scalar AUROC, so AUROC alone will
not justify the FIS. Its edge has to be **attribution**, which the "none of the
above" rule provides natively and Mahalanobis does not:

1. **Per-layer, per-feature anomaly attribution.** The FIS firing strengths say
   *which* learned patterns failed to fire and *where* in the network. Build the
   layer-resolved anomaly profile for each attack family and show that families
   have distinct signatures (an override attack lights up different layers than
   an encoding attack). That is an interpretability result Mahalanobis cannot
   produce, and it is the kind of readable-rules argument the thesis already
   makes elsewhere.
2. **Compare against the honest baselines**: Mahalanobis, isolation forest, a
   supervised linear probe (which *has* seen attacks — the unfair upper bound),
   and the surface-only control. The claim is not "highest AUROC" but "within ε
   of the best unsupervised detector, needs no attacks, and explains itself".
3. **IT2 for the abstention band.** Now that the IT2 regressor is fixed
   (PR #103), test whether an interval-valued anomaly score gives a calibrated
   "uncertain — send to review" band between clearly-benign and clearly-attack,
   rather than a single threshold. This is the one genuinely-motivated Type-2
   use the project has surfaced.

## Phase 3 — online / streaming (the ambitious version)

Everything so far reads the *prompt*. The richer signal is watching activations
*during generation* — flag a response that starts benign and drifts into a
policy violation. This is a per-step version of the same one-class monitor and
reuses the capture harness (`capture.py` already records per-step hidden states).
Higher risk, higher payoff; attempt only if Phase 2 lands.

## What would kill it, stated up front

* If a supervised linear probe on activations beats the one-class FIS by a wide
  margin *and* generalises across attack families, the interpretability angle is
  the only thing left — and it has to carry the paper alone.
* If benign-but-unusual prompts (rare languages, code, long context) trip the
  monitor as readily as attacks, the false-positive rate makes it unusable, and
  no amount of AUROC on a clean test set fixes that. **Measure FPR on hard
  benign traffic before claiming anything.**
* If the layer-attribution signatures are not stable across models or seeds,
  Phase 2's distinctive contribution evaporates.

## Honest assessment

This is the most promising of the three directions explored — drafting is
closed, hallucination detection is a correction-to-prior-work story, and this
has a real signal in a regime that suits the FIS and a valuable application. But
the scalar-AUROC edge over Mahalanobis is not there; the case rests on
"unsupervised + interpretable + no attack examples", and Phase 2 is where that
case is won or lost. It is a credible chapter, not a guaranteed one.
EOF
echo "plan written"