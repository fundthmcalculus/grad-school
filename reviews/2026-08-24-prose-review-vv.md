# Prose review — verification & validation, and certifiable AI (2026-08-24)

Scope: all thirteen `research/proposal-defense/prose/*.md` (≈65,500 words), `references.bib`,
`prose/bibliography.md`, `CHECKLIST.md`, and `build_pdf.py`'s validation stage. The review was
asked to check the document's treatment of **verification and validation**, specifically to
include **certifiable AI**. Findings below are marked ✅ fixed in this pass or ⬜ left open.

`build_pdf.py` re-run after every edit: **all cross-references verified, 82 bibliography
entries.** (No PDF — `pandoc` is not installed in
this environment. The assembly and validation stage, which is the part these edits could break,
runs clean.)

---

## The audit, before the edits

The question was what the document already said. The answer was: nothing, in the sense that was
asked about.

| Term | Occurrences across 13 files (before) | In what sense |
|---|---|---|
| `verificat*` | 9 | **all** software-engineering: checking an implementation against a reference |
| `validat*` | 18 | **all** either cross-validation, or "each kernel validates the other" |
| `certif*` | **1** | metaphorical — "a five-seed mean *certified* as stable a model that fails one time in ten" (§6.4) |
| `assuranc*`, `airworth*`, `DO-178`, `DO-333`, `EASA`, `traceab*`, `safety-critical` | **0** | — |
| `regulat*` | 2 | one clause in §1.1, one in Ch 8 |

So: **the systems-engineering sense of V&V was absent from the document, and certifiable AI was
absent entirely** — while the whole interpretability argument, which §2.6 calls "the property I
am unwilling to trade away," leans on it. Two clauses ("regulators and safety engineers
increasingly unwilling to accept a black box"; "in regulated domains, safety engineering … and
the aerospace applications this department cares about") gestured at the framing and neither
carried it.

This was a *tracked* gap, not an oversight: `CHECKLIST` **E10** bundled "(a) how heavily to
invoke the XAI/regulation framing" and "(b) whether to include a formal-methods/verification
subsection (possible Kreinovich nod)" as undecided editorial items. This pass decides both.

---

## Findings

### ✅ V1 — The V&V argument was already being made, in three places, unnamed

The strongest finding of the review, and the reason the addition is a *naming* pass rather than
a new claim. The document repeatedly prefers the configuration that fails **predictably** over
the one with the better mean, and gives the same reason each time:

- §4.3 on normalization: an unbounded input domain "does not make this construction worse on
  average so much as it makes it *less predictable*, which for a component inside a larger
  pipeline is the more expensive property."
- §4.3.2 on output partitioning: "Uniform fails, but *predictably*, which for a component inside
  a larger pipeline is often the more valuable property."
- §4.3.2 again, on why uniform wins: "its failure mode is the better one to own: starvation is
  visible in the bucket occupancy *at fit time*."

That is a V&V argument — bounded, inspectable failure over unbounded average performance — made
three times without the vocabulary that gives it force. §2.6's new subsection names it and
points back at both sections.

### ✅ V2 — §2.6 argued the interpretable-by-construction position while citing only the other side

The section's central move is *interpretable by construction* over *post-hoc explanation*, and
it cited `lundberg2017shap` (the position it argues against) and nothing for the position it
holds. Rudin's "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions"
(*Nature Machine Intelligence* 1(5):206–215, 2019) is the canonical statement of exactly that
position, in exactly the high-stakes framing this chapter needs, and a committee would ask why
it was missing. Added as `rudin2019stop`, metadata verified `[V]`, cited at the end of the
existing SHAP paragraph so the argument now has both sides on the page.

### ✅ V3 — Introducing the systems-engineering sense of V&V collided with the document's own usage

The document's existing 27 uses of "verification"/"validation" are all in the narrower software
and statistical senses (V-audit table above). Dropping the systems-engineering sense in on top
of them, unannounced, would have made "validation" ambiguous in a document that is otherwise
careful about exactly this. The new subsection therefore **opens with a terminology paragraph**
distinguishing the two, names which sense it uses, and says that it is the only place using the
second sense.

### ✅ V4 — "certified" was in use metaphorically, and that became a collision

§6.4's retraction story read "A five-seed mean **certified** as stable a model that fails one
time in ten." Harmless when the document never used the word as a term of art; not harmless once
six passages use "certification" in the regulatory sense. Changed to "**passed** as stable" —
the sentence keeps its force and the term of art keeps its meaning.

### ✅ V5 — The framing needed a boundary, and the boundary needed a tracked home

A V&V/certification framing is the easiest kind of claim to overreach on, because none of it is
measurable and all of it sounds good. Every one of the six passages added therefore carries the
same explicit boundary, and §2.6 states it in bold: **no certification artifact, no DO-178C or
DO-333 objective claimed as satisfied, no assurance case.** §2.6 also names what evidence
*would* be required (operational design domain, hazard analysis, behavioural coverage of the
rule base over that domain, traceability records) so the gap is a stated size rather than a
vague one.

More importantly, §7.4 now carries it **as an exposure with nothing in the runway behind it** —
the same register as that section's existing "two exposures a committee will find" — including
the note that the nearest goal, **G6**, measures the exported partition's *semantic* properties
(coverage, distinguishability, normality, partition-of-unity error) and that those are
fuzzy-design criteria, not certification ones. That distinction is the one a committee member
from the aerospace side would press on, and the document now makes it before they do.

### ✅ V6 — What was deliberately *not* claimed

Two temptations declined, recorded here because the absence is a decision:

1. **Determinism.** The obvious V&V flourish is "a closed-form fit with no stochastic search
   returns the same model from the same data, which is a precondition for repeatable review."
   The document does not support it. Every headline number is `mean ± std` across ten seeds with
   real spread, §6.3.5's antecedent refinement is a search, and Ch 8 already concedes that "the
   best flat figures quoted here include it." Claiming seed-determinism would have been exactly
   the kind of plausible-sounding, unverified sentence `WORKINGDOC.md` §7 is about. The
   subsection uses the *bounded-failure* property instead, which is measured.
2. **A Kreinovich formal-methods subsection**, as a *separate section*. `CHECKLIST` E10(b)
   floated one; the material is folded into §2.6 instead, so the interpretability definition and
   the V&V argument sit together rather than in two places that have to be kept consistent. The
   Kreinovich citation itself is **in** — see V11, which supersedes an earlier version of this
   entry that dropped it on a bad search.

### ✅ V7 — `build_pdf.py` printed a hardcoded, stale bibliography count

`build_pdf.py` printed `references.bib found ({bib_size} bytes, 70 entries)` — a literal, wrong
since the file passed 70 entries (it was at 74 before this pass, 79 after). Now counts `^@`
lines, so the build and `prose/bibliography.md` agree by construction instead of by remembering
to update both. This is the same class of defect as the stale checklist-ID list that commit
`c04836c` fixed.

### ⬜ V8 — Chapter 8 has no numbered sections, and the other chapters do

Caught by nearly writing a `§8.3` cross-reference that would have dangled. Chapters 1–7 and the
appendix all carry `§X.Y` headings; Chapter 8 is a single unbroken run of paragraphs, so
anything referring into it can only say "Chapter 8". `build_pdf.py`'s reference checker would
have caught the dangling `§8.3` — it did not need to, but only because the reference was
rewritten first. Low stakes, and worth a decision rather than drift: either give Chapter 8 three
numbered sections (the retrospective, the limits, the methodological close — the three parts it
already has) or leave it deliberately unnumbered. Not fixed here; restructuring a chapter is not
a V&V review's call.

### ⬜ V9 — Two EASA citations rest on index listings, not title pages

`easa.europa.eu` is blocked by this environment's egress proxy, so `easa2023airoadmap` and
`easa2024mlconcept` are marked `[?]` rather than guessed into `[V]`. The Roadmap 2.0 date
(10 May 2023) is corroborated by two independent listings; the Concept Paper Issue 02's exact
issue date is **not** established, so that entry carries a year and no month. Tracked as
`CHECKLIST` **E11**. Both are cited only for what a standard or roadmap *asks for*, never for a
number, so a metadata correction at proof stage cannot move a result — which is why this is
proof-stage work rather than blocking.

### ⬜ V10 — Worth checking, outside this review's reach

Ch 1 §1.1 claims the pipeline "produces fuzzy models that train in a second or two … **with no
stochastic search in the fit at all**", and quotes $R^2 = 0.861 \pm 0.026$ two sentences
earlier. Ch 8 says "Chapter 6's optional antecedent refinement is a gradient-based solve,
however, and the best flat figures quoted here include it, so the claim is about how the model is
built, not the whole pipeline." Whether the specific 0.861 figure — which §4.4 traces to
`table_hyperparam_normalization.py`, not to the reconciliation generator that carries the
refinement arm — includes refinement cannot be settled from the prose. If it does, §1.1's
absolute and Ch 8's qualification are in tension in the document's most-read paragraph. Settling
it means reading the generator, not the chapters, so it is flagged rather than fixed. It bears on
this review because a repeatability argument is weakened by a search inside the quoted numbers,
which is part of why V6.1 declined the determinism claim.

---

### ✅ V11 — The Kreinovich nod E10(b) asked for is real, and it is load-bearing

A first pass of this review dropped it, reporting that no single Kreinovich paper spans fuzzy
systems and formal verification. That was a bad search, and the author overruled it. A proper one
finds the paper immediately:

> Cohen, K., Bokati, L., Ceberio, M., Kosheleva, O. & Kreinovich, V., **"Why Fuzzy Techniques in
> Explainable AI? Which Fuzzy Techniques in Explainable AI?"** *Explainable AI and Other
> Applications of Fuzzy Techniques* (NAFIPS 2021), LNNS **258**:74–78, Springer, 2022.
> DOI 10.1007/978-3-030-82099-2_7.

It earns a citation twice over rather than serving as a name-drop. The first half is the general
argument §2.6 needs — fuzzy technique as the route to explainable AI. The second half is a
caveat this dissertation should own rather than inherit quietly: **which** fuzzy operations are
the right ones is problem-dependent, not settled once. That lands directly on an open question
already in this document — §4.3.5 runs a Hamacher conorm inherited from `beth-anomaly.py` and
states outright that whether the family matters to detection is untested. So §2.6 cites it as a
debt, not an endorsement. The venue is also this author's own (NAFIPS 2025 Banff, NAFIPS 2026
El Paso, per §1.3).

### ✅ V12 — The same search turned up two things worth more than the nod

**An existence proof.** Arnett, T., Ernest, N., Kunkel, B. & Boronat, H., "Formal Verification of
a Genetic Fuzzy System for Unmanned Aerial Vehicle Navigation and Target Capture in a Safety
Corridor," *Fuzzy Information Processing 2020* (NAFIPS 2020), AISC **1337**:361–372, Springer,
2021, DOI 10.1007/978-3-030-81561-5_30 — in a volume Kreinovich co-edited. This is a fuzzy
controller **actually taken through formal verification** against a behavioural safety
specification, in aerospace, on the analyzability constraint set §2.1 adopts. It changes what
§2.6 can say: the model family is not reviewable only in principle, someone has verified a member
of it. §2.6 now says exactly that, and then the sentence that keeps it honest — "What nobody has
verified is one of mine." The boundary paragraph was tightened accordingly: "an interpretable rule
base makes a review possible, and Arnett et al. show the family admits one, but neither fact is
evidence about a model I built."

**A resolved `[?]`, and an uncited entry.** `arnett2018proposal` has been this file's oldest
unresolved entry — placeholder title *"Dissertation Proposal"*, and a `.bib` header note saying
it was "Cited in section 2.1." It was **not cited anywhere in the prose**, and §2.1's
analyzability constraints were asserted with no citation and no stated reason — so the entry's
recorded purpose ("parallel use of the same FIS constraint set") existed only in the `.bib`.
Both halves fixed: §2.1 now cites it, and gives the constraints their reason. And the proposal now
has a document to be a proposal *for* — Arnett's completed UC dissertation, *"Iteratively
Increasing Complexity During Optimization for Formally Verifiable Fuzzy Systems"* (2019), added
as `arnett2019iteratively`. **E11 says to verify the two together**, since confirming the 2019
title is what would resolve the 2018 placeholder rather than merely re-check it.

This is the finding that justifies the whole detour: the document's link to the formal-methods
literature was already half-present, as a bibliography entry pointing at a section that never
cited it.

## What changed

| File | Change |
|---|---|
| `prose/01-introduction.md` | §1.1: the "regulators and safety engineers" clause now names explainability-as-certification-objective and points to §2.6, with the boundary stated in the same sentence. Light, per E10(a). |
| `prose/02-background.md` | §2.1: the analyzability constraints now state *why* they are made, and credit Arnett's constraint set and the formally-verified UAV controller (V12). §2.6: `rudin2019stop` added to the SHAP paragraph; **new subsection "Verification, validation, and certifiable AI"** — terminology, DO-178C/DO-333, EASA learning assurance, why a rule base is the artifact those frameworks want, the two properties the document already reports that read as V&V arguments, the Kreinovich nod and the Arnett existence proof (V11, V12), and the boundary in bold. |
| `prose/04-fast-fis-synthesis-mog.md` | §4.3.5: one paragraph reading the *none of the above* rule as an operational-design-domain monitor that is part of the model — with §4.4's limits attached (max-membership degeneracy at the shipped θ, level with an isolation forest on Glass, 214 samples supports no monitoring claim). |
| `prose/06-hierarchical-refined-fis.md` | §6.3.4: the Ruspini export named as the one output a V&V process could take as an input, with "no such process has been run on it" and the G6-measures-something-else caveat. §6.4: metaphorical "certified" → "passed" (V4). |
| `prose/07-goals-for-completion.md` | §7.4: new lead risk paragraph (seven invocation sites named) — the framing is motivation with nothing measured behind it, G6 is not it, and if a committee wants it inside this dissertation something in Table 7.1 comes out. |
| `prose/08-conclusion.md` | The "regulated domains, safety engineering, aerospace" sentence now names V&V in the systems sense and carries the same boundary, pointing at §7.4. |
| `references.bib` | +8 entries (82 total): `rudin2019stop` `[V]`; `rtca2011do178c`, `rtca2011do333` `[S]`; `easa2023airoadmap`, `easa2024mlconcept` `[?]`; `cohen2022whyfuzzy`, `arnett2021formal` `[V]` on their SpringerLink indexed records; `arnett2019iteratively` `[?]`. Header count note updated, including the 2026-08-21 `deboor` entry the header had not recorded. |
| `prose/bibliography.md` | 2026-08-24 update block (82 = 53 `[V]` + 25 `[S]` + 4 `[?]`), new "Verification, validation, and certifiable AI" reading-guide group, `rudin2019stop` added to the XAI group, and the `arnett2018proposal` was-never-actually-cited note (V12). |
| `CHECKLIST.md` | **E10** → 🟨: (a) and (b) decided and recorded — (b) with the Kreinovich citation in, plus the two Arnett finds; (c)/(d) still open. New **E11** for the proof-stage verification of five entries (the two EASA, the two Springer, and the Arnett dissertation — the last paired with the long-standing `arnett2018proposal` `[?]`) and for keeping the seven boundary passages consistent. |
| `build_pdf.py` | Bibliography entry count no longer hardcoded (V7). |

## Verdict

The document's V&V treatment was not weak, it was **missing** — and it was missing under a
tracked checklist item, in a document whose central argument needs it. The gap is now closed on
the terms the rest of the proposal is written on: the framing is stated once where the
interpretability definition lives, invoked briefly in five other places, cited to the actual
standards *and* to the fuzzy-systems literature that has been making the same case (Kreinovich et
al. for the argument, Arnett et al. for a fuzzy controller actually verified), bounded explicitly
everywhere it appears, and recorded in §7.4 as an exposure with nothing in the runway behind it. Nothing here is a new
result and nothing here is scheduled as one. The honest one-line position for the defense, and
the one the prose now states: **this work makes a certifiable model family available; it
certifies nothing.**
