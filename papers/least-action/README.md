# Certified Fuzzy Control by Least Action — method paper

Write-up of the work in `research/least_action/`. Every table is transcribed
from `research/least_action/results.txt`, which is the recorded output of the
deterministic demo scripts.

## Build

```bash
pdflatex least_action_fis.tex && pdflatex least_action_fis.tex
```

Needs only `texlive-latex-base` + `texlive-latex-recommended`. `microtype` is
deliberately not loaded, so the source builds on a minimal TeX installation;
add it back if you have scalable fonts.

## For NAFIPS

Swap the class line for `\documentclass[conference]{IEEEtran}` and drop the
`\geometry` line and `\tableofcontents`. The paper is currently ~18 pages in
article class, so it will need cutting for a conference limit — §4
(orthogonality), §5 (classifier) and §§9–10 (certificate-aware optimization and
calibration) are the three most separable stories if it needs to become two
papers.

## Source of each result

| paper section | code | recorded output |
|---|---|---|
| §2–3 action, variable projection, gradient | `fis_action.py` | `demo_regression.py` A–G′ |
| §4 orthogonality, λ* | `fis_action.py` | `demo_regression.py` C, D, H, I |
| §5 classifier | `fis_action.py` | `demo_classifier.py` H–K |
| §6 control certificates | `fis_control.py` | `demo_control.py` L–P |
| §7 two-cart benchmark | `fis_twocart.py` | `demo_twocart.py`, `demo_policyopt.py` |
| §8 SOS certification | `fis_sos.py` | `demo_sos.py`, `verify_sos_exact.py` |
| §9 certificate-aware optimization | `fis_sos.py`, `fis_twocart.py` | `demo_pareto.py` |
| §10 calibration | `fis_sos.py` | `demo_calibrate.py`, `verify_proxy.py` |
| §12 machine verification | — | `verify_symbolic.py`, `verify_sos_exact.py` |
