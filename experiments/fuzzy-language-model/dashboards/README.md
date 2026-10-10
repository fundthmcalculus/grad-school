# Dashboards

Two multi-page results dashboards, published as artifacts (private until shared from the page's Share menu):

| dashboard | artifact | pages |
|---|---|---|
| **Tiny Fuzzy Language Models** (CPU phase) | https://claude.ai/artifact/XGJzPWz5T5i2wQvVE6C3Y4 | Overview and claims, size and quality, fuzzy components, tuning and memory, n-gram initialisation |
| **Fuzzy LM Goal Track** (GPU phase) | https://claude.ai/artifact/9kWT8VFUhxs1gKhwLhgqc5 | Overview, where the deficit was, the comparison, bigger models and post-training, subwords, generated text, reading the model (rulebook), open gaps |

Every number on a page comes from a dataset exported from the run records; aggregations are inspectable
calculations on the dashboard. Regenerate the data:

```bash
.venv/bin/python -m flm.dashboard_export     OUT_DIR   # CPU phase: scaling, smallest, gap, h9, tune, state, ablate_runs, screen_runs, screen_curves, ngram_ref, claims
.venv/bin/python -m flm.dashboard_export_gpu OUT_DIR   # GPU phase: gpu_runs, gpu_curves, gpu_samples, gaps
```

`cpu_index.html`, `gpu_index.html`, `lib.js` and `app.css` are the page sources as published (`lib.js` and
`app.css` are shared). The data is a snapshot: after new runs, re-export, re-upload the changed CSVs and update
the dataset documents. `claims.csv` and `gaps.csv` are written by hand in the exporters (they are status
tables, not measurements); update them when a status changes.

The GPU export's `rulebook.csv` needs the BPE1024 tokenizer (`data/tinystories/bpe1024.json`, committed) and a
validation token stream rebuilt from it: `flm.bpe.encode_stream(tok, load('valid'), len(load('valid')), paths(1024)['valid'])`
(2 seconds). Set `GRAD_SCHOOL_DATA` if the data is not under `data/`.
