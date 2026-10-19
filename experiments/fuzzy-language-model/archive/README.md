# Archive: levers that did not work

Moved here on 2026-10-11 (the clean-up asked for by the repo owner, after the GPU session confirmed nothing running imports them). Nothing was deleted; the outcomes are in `../RESULTS.md` and summarised in `../REVIEW.md` section 4.

| file(s) | what it was | outcome |
|---|---|---|
| `flm/run_conseq.py`, `conseq_first.py`, `conseq_first_check.py`, `assoc.py` | consequent-first (TRIBBLE-order) and PPMI/association rule initialisation | no gain over random initialisation |
| `flm/run_cover_sweep.py`, `run_hybrid_cpu.py` | coverage and cluster hybrid initialisations | all worse than random (`v27-cover`, `v26-hybrid`) |
| `flm/run_rep.py`, `run_rep_sweep.py`, `rep_share.py` | repetition-aware loss (unlikelihood, repeat-weighted) | repeats fell 20%, blind coherence fell too (set 9) |
| `flm/run_emb.py`, `run_emb_sweep.py` | embedding initialisation, scale and weight decay | null (`v28-emb`) |
| `flm/rflm_variants.py`, `run_variant.py`, `run_rflm.py`, `test_rflm_variants.py` | RFLM variants (fixed recurrence, consequent propagation) | 4M-character screen only, no clear effect |

The modules import each other as `flm.<name>`. To run one again, move the set back with `git mv archive/flm/<name>.py flm/` (the whole group it imports) and restore `test_rflm_variants.py` if needed. `conftest.py` keeps pytest from collecting this folder.

Still in `flm/` because other tooling needs them: `models.py`, `train.py`, `bpe.py`, `data.py`, `sweep.py`, every `grids_v*.py`, `ngram.py` (the n-gram initialisation flag lives in `train.py`), `run_tribble.py`, `run_tribble2.py`.
