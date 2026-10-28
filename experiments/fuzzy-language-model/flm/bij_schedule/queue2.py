"""oh40 frozen minimum, drift penalty on oh48, oh48 on the Pile (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/queue2.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
PILE = "C:/personal/grad-school/data/pile_chars"
J = []
for s in (0, 1):
    J.append(("oh40/learned", s, "30", dict(D="40", FFN_R="250", FFN_R0="175")))
    J.append(("oh40/frozen", s, "30", dict(D="40", FFN_R="250", FFN_R0="175", FIXED_C="outputs/embspace/oh40.npz", RUN_MODULE="flm.run_fixed_embedding")))
    J.append(("oh40/init", s, "30", dict(D="40", FFN_R="250", FFN_R0="175", FIXED_C="outputs/embspace/oh40.npz", RUN_MODULE="flm.run_fixed_embedding", FIXED_TRAIN="1")))
for s in (0, 1):
    for k in ("0.001", "0.01"):
        J.append((f"pull/oh48_pull{k}", s, "30", dict(D="48", FFN_R="196", FFN_R0="137", FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding", FIXED_TRAIN="1", FIXED_PULL=k)))
for arm, extra in (("learned", {}), ("frozen", {}), ("init", {"FIXED_TRAIN": "1"})):
    e = dict(D="48", FFN_R="196", FFN_R0="137", DATA_ROOT=PILE)
    if arm != "learned":
        e.update(FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding", **extra)
    J.append((f"oh48_pile/{arm}", 0, "300", e))


def run(j):
    name, s, chars, e = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", chars, f"outputs/{name}_s{s}", "--precision", "fp32", "--seed", str(s)],
        env=dict(os.environ, **e),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
