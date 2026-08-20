"""The prompt battery.

Two sources, kept separate because they stress different parts of the
distribution:

* ``dolly`` -- databricks-dolly-15k, eight human-written task categories
  (QA, summarization, brainstorming, classification, extraction, creative
  writing). This is the deployment-realistic set.
* ``synthetic`` -- hand-written probes that deliberately pin the next-token
  distribution to a known regime: near-deterministic (arithmetic, quoting a
  memorized string), near-uniform (open-ended creative continuation), and
  the syntactic-state cases (mid-word subword continuation, post-punctuation)
  that any "shape" model has to get right.

The synthetic set exists so that a failure has a diagnosis. If a model of
distribution shape works on dolly but not on the probes, we know which regime
broke it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    text: str
    source: str
    category: str
    # Completion probes must NOT be wrapped in a chat template. Wrapping "The
    # capital of France is" as a user turn makes the model answer
    # conversationally instead of completing the string, which destroyed the
    # intended regime separation on the first run: low_entropy measured 2.508
    # nats against high_entropy's 2.477 -- no separation at all. The template
    # is a property of the prompt, not of the run.
    use_chat: bool = False


# --------------------------------------------------------------------------
# Synthetic probes. `category` names the *distributional regime* we expect,
# and that expectation is itself checked in experiment 0 -- these are
# hypotheses about entropy, not ground truth.
# --------------------------------------------------------------------------

_LOW_ENTROPY = [
    "The capital of France is",
    "2 + 2 =",
    "One, two, three, four,",
    "Monday, Tuesday, Wednesday,",
    "The first three letters of the alphabet are A, B, and",
    "Roses are red, violets are",
    "H2O is the chemical formula for",
    "10 times 10 equals",
]

_HIGH_ENTROPY = [
    "Write a story about",
    "Here is an interesting fact:",
    "My favorite thing about the world is",
    "In the beginning,",
    "She opened the door and saw",
    "The best advice I ever received was",
]

_MID_WORD = [
    # Deliberately cut inside a word so the model must continue a subword.
    "The astronaut experienced a sudden decompress",
    "This is completely unaccept",
    "The parliamentary representat",
    "An extraordinarily sophist",
]

_STRUCTURED = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    '{\n  "name": "Alice",\n  "age":',
    "| Name | Age |\n|------|-----|\n| Bob  |",
    "import numpy as np\nimport pandas as",
]

SYNTHETIC: list[Prompt] = (
    [Prompt(t, "synthetic", "low_entropy") for t in _LOW_ENTROPY]
    + [Prompt(t, "synthetic", "high_entropy") for t in _HIGH_ENTROPY]
    + [Prompt(t, "synthetic", "mid_word") for t in _MID_WORD]
    + [Prompt(t, "synthetic", "structured") for t in _STRUCTURED]
)


def load_dolly(n: int, seed: int = 0, max_chars: int = 600) -> list[Prompt]:
    """Sample `n` dolly prompts, stratified across its eight categories.

    Stratified rather than uniform because the category mix is imbalanced and
    category is a plausible confounder for entropy -- the anomaly study in
    `experiments/fuzzy-lm-anomaly.md` found prompt family alone reaching ~0.9
    AUROC on a task it had no business predicting. Balancing here means the
    category effect can be measured instead of silently ridden.
    """
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.filter(lambda r: len(r["instruction"]) + len(r["context"]) <= max_chars)

    categories = sorted(set(ds["category"]))
    per_cat = max(1, n // len(categories))

    out: list[Prompt] = []
    for cat in categories:
        sub = ds.filter(lambda r, c=cat: r["category"] == c)
        sub = sub.shuffle(seed=seed).select(range(min(per_cat, len(sub))))
        for row in sub:
            text = row["instruction"]
            if row["context"]:
                text = f"{row['context']}\n\n{text}"
            out.append(Prompt(text, "dolly", cat, use_chat=True))
    return out[:n]


def build_battery(n_dolly: int, seed: int = 0) -> list[Prompt]:
    """Full battery: every synthetic probe plus `n_dolly` sampled dolly rows."""
    return SYNTHETIC + load_dolly(n_dolly, seed=seed)
