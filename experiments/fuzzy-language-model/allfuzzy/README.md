# The all-fuzzy language model, end to end in numpy

`allfuzzy_lm.py` is a self-contained re-implementation (numpy + the HF `tokenizers` package; no torch) of the end-to-end all-fuzzy model of `PLAN_FUZZYLOCAL.md`
(`e2e_tail`, seed 0, 300M characters of training, **99,019 parameters**, 1.0007 BPC at 300M characters on average over two seeds against 0.9589 for the hybrid recipe).
Every block component is a fuzzy inference system; the code is written so each one reads as fuzzify, fire (product t-norm), aggregate, defuzzify.

```python
from allfuzzy_lm import generate, write_rulebook
print(generate("Once upon a time"))             # text in, generated continuation out (a str)
write_rulebook("rulebook.txt")                   # every rule of every fuzzy system, as IF-THEN text
```
```
python allfuzzy_lm.py --text "Once upon a time" --trace    # also prints, per word piece, the feed-forward rules that pushed it most (exact direct-path effects)
python allfuzzy_lm.py --rules rulebook.txt                  # 788 lines: 87 feed-forward rules, 512 memory rules, 20 gate rules, 16 projection rules, 25 gain rules, 12 prototypes
```
`generate(text, max_new_tokens=60, temperature=0.5, top_k=10, seed=0, include_prompt=False, trace=None)`; the prompt plus continuation is capped at the 512-piece context.

**Files.** `e2e_tail_300M_s0.npz` (weights, 467 KB), `bpe3072.json` (tokenizer), `rulebook.txt` (the generated rule base), `export_weights.py` (one-off checkpoint to npz export, needs torch),
`test_allfuzzy.py` (parity test against the torch model: max logit difference 1.5e-5 over a 24-piece prompt, identical argmax at every position).

**What the fuzzy systems are** (per block unless stated): two gain controllers (TSK systems on log-rms, replacing RMSNorm; 5 rules each); two rule-gated low-rank maps (8 rules each, replacing the
q/k/v and output projections); two six-rule TSK gates (forgetting, write strength); a fuzzy associative memory (8 heads x 32 Gaussian rules over a 6-number key/query, consequents written in context by the delta rule);
a feed-forward bank of Gaussian rules on the raw input with first-order consequents (52 rules in block 1; 35 in block 0, where the antecedent also reads the previous token, replacing the convolution).
Output side: a gain controller, then scores through the tied table: the 384 most frequent word pieces are crisp rows, the other 2,688 are fuzzy sets over 12 prototypes (weighted average of the prototypes' scores).

**Reading the rules honestly.** The rule texts name the four most selective membership functions of each rule and the word pieces its consequent raises or lowers when it fires positively. Most single-dimension
memberships are wide (widths around 4 against inputs of about unit scale): no rule is sharply selective on one dimension; a rule is selective through the product over all 48 (or 96) dimensions. The
rule base is exact (the script reproduces the trained model), but a rule is a point in a 48-dimensional space, not a short sentence.
