# Certified Fuzzy Control by Least Action — paper materials

Write-up of the work in `research/least_action/`. Every table is transcribed
from `research/least_action/results.txt`, which is the recorded output of the
deterministic demo scripts.

| file | what it is |
|---|---|
| `paper.md` | The method paper. Control only — the classifier development is a separate paper. |
| `one-pager.md` | What is proven and why it is valuable. For sharing with people who will not read 18 pages. |
| `roadmap.md` | Prioritized next steps, benchmark control problems, and the methods to compare against. |
| `nafips-split.md` | How to break this into three (possibly four) NAFIPS papers, with readiness assessment. |

## Scope

This is a **control** paper. The classifier results — Mamdani aggregation
reducing to summation iff output supports are disjoint, centroid defuzzification
equalling an order-0 TSK model, and the $\beta$-annealed mean-of-maxima error
bound — are deferred to a companion paper and appear here only as two explicit
pointers (paper §1.2 and the note under the guarantee table). Guarantee IDs
G6–G8 are skipped rather than renumbered so they still match the implementation
notes in `research/least_action/README.md`.

## Format

Markdown, with LaTeX math that renders on GitHub. A LaTeX version was built
earlier and retired when the scope narrowed to control; it is recoverable from
git history at commit `677f983` (`papers/least-action/least_action_fis.tex`) if
a typeset PDF is wanted later. Regenerating it from `paper.md` is
straightforward — the structure is unchanged apart from the removed section.

## Source of each result

| paper section | code | recorded output |
|---|---|---|
| §2–3 action, variable projection, gradient | `fis_action.py` | `demo_regression.py` A–G′ |
| §4 orthogonality, λ* | `fis_action.py` | `demo_regression.py` C, D, H, I |
| §5 control certificates | `fis_control.py` | `demo_control.py` L–P |
| §6 two-mass–spring benchmark | `fis_twocart.py` | `demo_twocart.py`, `demo_policyopt.py`, `demo_holdout.py` |
| §7 SOS certification | `fis_sos.py` | `demo_sos.py`, `demo_certified_policy.py`, `verify_sos_exact.py` |
| §8 certificate-aware optimization | `fis_sos.py`, `fis_twocart.py` | `demo_pareto.py` |
| §9 calibration | `fis_sos.py` | `demo_calibrate.py`, `verify_proxy.py` |
| §11 machine verification | — | `verify_symbolic.py`, `verify_sos_exact.py` |
