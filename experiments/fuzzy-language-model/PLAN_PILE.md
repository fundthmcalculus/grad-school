# PLAN_PILE: the Pile sample for the Pythia comparison, scaffolded (2026-10-23; nothing trained beyond a 15-step CPU smoke run)

Owner's instruction (2026-10-23): scaffold the Pythia streaming and Pile-sample work, do not fully train on it, test it on a CPU trainer while the GPU is busy, then open a pull request so it can be tried on another machine (probably a bigger GPU). The idea: take what the earlier rounds learned, meld it with the random allocation search (`PLAN_ROUND3.md` stage 4), and identify the proper training configuration for this corpus. Related: `PLAN_PYTHIA.md`, `ISSUE_STREAMING.md`.

## What exists now
| piece | file | tested |
|---|---|---|
| deterministic sample of `EleutherAI/the_pile_deduplicated` (revision `fcbfcfde…`), streamed, folded to the 98-symbol alphabet, split into train and validation by a hash of shard and row, tokenised, written in the layout every trainer reads, with a manifest of sha256 sums | `flm/prep_pile.py` | `test_stream.py` (7 tests, no network); a real 2-shard stream of the first and the last shard |
| arms fitted to the 99,999-parameter cap for the Pile vocabularies | `flm/pile_arms.py`, `report/pile_arms.json` | parameter counts equal the fit |
| the trainer's CPU smoke path for any corpus | `SHAPE_DEVICE=cpu` in `flm/run_shape.py` | a 15-step run on the Pile sample: 99,767 parameters, 4.52 -> 3.97 bits per character in 90 seconds |
| a mixed TinyStories + SimpleStories corpus (stage 5) | `flm/prep_mix.py`, `flm/eval_mix.py`, `report/mix_arms.json` | `test_mix.py`; a CPU smoke run and a checkpoint re-score that reproduced the training loss (2.8138 against 2.8133) |

Nothing in `flm/train.py` changed: a corpus is a directory `data/<name>/tinystories/` (`bpe<V>.json`, `lens_bpe<V>.npy`, `train_bpe<V>.bin`, `valid_bpe<V>.bin`) plus `outputs/freq_<name>_bpe<V>.npz`, selected with `SHAPE_CORPUS=<name>`.

## Findings from the scaffold
1. **Pythia's 50,277-token vocabulary cannot fit under 100K parameters.** The fuzzy tail table alone puts the model at 437,615 parameters at 50,277 tokens with tail width 8 (head 640, d32, two rules), 238,939 at width 4 and 139,601 at width 2, against a cap of 99,999 (6,144 tokens at width 8: 84,551). The Pile run therefore uses a BPE trained on the Pile sample (6,144 or 8,192 tokens, `--tokenizer bpe:<V>`). Bits per character does not depend on the tokenizer, so the comparison with Pythia is unaffected.
2. **Text is folded to 98 symbols** (tab to space, carriage return dropped, everything else outside printable ASCII to a question mark), as for the other corpora, so the denominator is exact. Pythia has to be scored on the same folded validation text for a like-for-like number; `flm/score_pythia.py` takes the raw corpora now, and a `--text` option for the folded file is the next small change (not done).
3. **A token can span more than 255 characters** (runs of dashes in the Pile), so token lengths are stored as 16-bit. The first version stored them as 8-bit and the length check (token lengths against the characters of the text) caught a 2-character mismatch in 5.7M. The check stays.
4. **Quotas are soft**: a shard stops at the first document past its character quota, and one Pile document can be megabytes (shard 0 gave 3.7M characters against a 2.0M quota in the smoke run). Fine for a sample; cap the document length if equal shard weights matter.
5. The mirror has a `text` column only (no `meta`), so the per-document Pile component is not available from this source.

## Training configuration for the Pile (a procedure for the other machine; not run here)
Registered before anything is run on this corpus.
1. **Corpus.** `python -m flm.prep_pile --name pile6144 --shards 40 --train-chars 330e6 --valid-chars 4e6 --tokenizer bpe:6144` (and a second corpus name with `bpe:8192`). Check the manifest: characters per token, any quota shortfall warning, and that the token-length check passed.
2. **Candidates** come from the random search (`PLAN_ROUND3.md` stage 4): its best three allocations (vocabulary, head, tail, d, mixer rules, rule split), fitted again for the Pile vocabulary with `flm.random_shape.fit_rules`, plus F7 itself as the reference (`P_bpe6144` and `P_bpe8192` in `report/pile_arms.json`).
3. **Screen**: each candidate at 30M characters, three seeds, warm-up 600, lr 6e-2, batch 16, floor 0, wd 0.01 (the recipe that survived 300M; the batch-8 lever did not). A candidate replaces F7 only if it beats F7 by 0.005 on two of three seeds.
4. **Confirm** the winner at 300M (two seeds, warm-up 1000), then score Pythia-14M on the same folded validation text and report bits per character against parameters. With a bigger GPU, re-measure throughput first; a larger batch is its own arm with its own learning-rate screen (the stage-2 lesson).

Predictions for the 30M screen on the Pile sample: F7's bits per character is above its TinyStories value of 1.05 by at least 1.5 (90%: the Pile is far harder than stories); the ranking of the allocations on the Pile differs from the ranking on TinyStories for at least one of the top three (65%); head size matters less on the Pile than on TinyStories (50%).

## Running it (other machine)
```
git fetch && git checkout exp/allfuzzy-round3 && uv sync
python -m flm.prep_pile --name pile6144 --shards 40 --train-chars 330e6 --valid-chars 4e6 --tokenizer bpe:6144
python -m flm.pile_arms --out report/pile_arms.json            # refit if the vocabulary changes
SHAPE_CORPUS=pile6144 SHAPE_ARMS=report/pile_arms.json SHAPE_OUT=outputs/v78-pile SEEDS=0,1,2 python -m flm.run_shape P_bpe6144
pytest experiments/fuzzy-language-model/test_stream.py experiments/fuzzy-language-model/test_mix.py
```
CPU smoke (no GPU): put `SMOKE=1 SHAPE_DEVICE=cpu` in front of the `run_shape` line (0.4M characters, about 90 seconds). One training job at a time. Nothing under `data/` or `outputs/` is committed.
