"""``GRAD_SCHOOL_DATA=<repo>/data/cosmopedia COH_SEEDS=60,61 python -m flm.coherence_cosmo out ckpt...``: flm.coherence_tiered with expository
(Cosmopedia-style) prompts instead of story prompts; same T 0.5 / top-k 10, 600 characters, stock or tiered checkpoints by their keys.
"""

import runpy

import flm.coherence_set as C

C.PROMPTS = [
    "Photosynthesis is the process by which",
    "To make a simple paper airplane, first",
    "The Roman Empire was one of the largest",
    "In this section, we will learn about fractions.",
]
if __name__ == "__main__":
    runpy.run_module("flm.coherence_tiered", run_name="__main__")
