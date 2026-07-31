"""Registry of every reproducible experiment in the grad-school repositories.

This is the single source of truth the orchestrator (``run.py``) consumes. Each
entry maps an experiment to the exact command that reproduces it, the submodule
it lives in, the datasets and hardware it needs, and the dissertation table or
figure it supports.

Add an entry here and it becomes runnable via ``python reproduce/run.py``.
"""

from dataclasses import dataclass, field


@dataclass
class Experiment:
    id: str                      # stable short id, e.g. "ch4-mog-concrete"
    title: str                   # human description
    chapter: str                 # "Ch3".."Ch6", "App" -- proposal chapter
    produces: str                # "Table 4.1", "Fig 3.2", "SUMMARY_REPORT.md", ...
    repo: str                    # submodule dir (cwd), relative to repo root
    command: list                # argv to run in `repo` (already env-prefixed)
    hardware: str = "any"        # any | cpu-parallel | gpu | big-mem
    datasets: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    notes: str = ""


# `uv run` executes inside a submodule's own locked environment.
def _uv(*args):
    return ["uv", "run", "python", *args]


EXPERIMENTS = [
    # ---- proposal tables (generators live in reproduce/tables, run under a submodule env) ----
    Experiment(
        id="table-3-1-pvat-scaling",
        title="pVAT vs classical VAT reorder time across N",
        chapter="Ch3", produces="Table 3.1",
        repo="tribble-cluster",
        command=_uv("../reproduce/tables/table_3_1_pvat_scaling.py"),
        hardware="any",
        outputs=["reproduce/outputs/table_3_1.md", "reproduce/outputs/table_3_1.csv"],
        notes="Times exact pVAT reorder vs a naive O(N^3) reference; classical capped at small N.",
    ),
    Experiment(
        id="table-4-1-mog-baselines",
        title="MoG FIS vs sklearn baselines (train time + accuracy/R2)",
        chapter="Ch4", produces="Table 4.1",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_1_mog_baselines.py"),
        hardware="any",
        datasets=["Concrete", "PhiUSIIL"],
        outputs=["reproduce/outputs/table_4_1.md", "reproduce/outputs/table_4_1.csv"],
        notes="ANFIS / GA-FIS / M5 columns fill in only if those adapters are available; else N/A.",
    ),
    Experiment(
        id="table-6-1-model-family",
        title="Flat / fuzzy-tree / HME vs CART/M5 baselines",
        chapter="Ch6", produces="Table 6.1",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_6_1_model_family.py"),
        hardware="any",
        datasets=["Concrete", "PhiUSIIL"],
        outputs=["reproduce/outputs/table_6_1.md", "reproduce/outputs/table_6_1.csv"],
    ),

    # ---- Ch4 / Ch6 fuzzy-model benchmarks (existing repo scripts) ----
    Experiment(
        id="ch4-mog-concrete", title="MoG-TSK on UCI Concrete (regression)",
        chapter="Ch4", produces="Fig 4.1 / Ch4 numbers", repo="tribble-fis",
        command=_uv("gaussian_mixture/concrete.py"), datasets=["Concrete"],
    ),
    Experiment(
        id="ch4-mog-phiusiil", title="MoG classifier on PhiUSIIL phishing",
        chapter="Ch4", produces="Ch4 numbers", repo="tribble-fis",
        command=_uv("gaussian_mixture/phiusiil.py"), datasets=["PhiUSIIL"],
    ),
    Experiment(
        id="ch4-mog-iot", title="MoG classifier on RT-IOT2022",
        chapter="Ch4", produces="Ch4 numbers", repo="tribble-fis",
        command=_uv("gaussian_mixture/iot.py"), datasets=["RT-IOT2022"],
        hardware="big-mem", notes="123K x 83, 12 classes; large.",
    ),
    Experiment(
        id="ch6-tree-concrete", title="Fuzzy tree + HME vs flat on Concrete",
        chapter="Ch6", produces="Table 6.1 / Fig 6.1", repo="tribble-fis",
        command=_uv("tribble-tree/demo_concrete.py"), datasets=["Concrete"],
    ),
    Experiment(
        id="ch6-tree-phishing", title="Fuzzy tree + HME on PhiUSIIL",
        chapter="Ch6", produces="Fig 6.1", repo="tribble-fis",
        command=_uv("tribble-tree/demo_phishing.py"), datasets=["PhiUSIIL"],
    ),
    Experiment(
        id="ch6-mimo-pendulum", title="MIMO memory FIS on double pendulum",
        chapter="Ch6", produces="Fig 6.3", repo="tribble-fis",
        command=_uv("tests/test_double_pendulum.py"),
        notes="Confirm exact MIMO-memory entry point; may live under tests/ or gaussian_mixture/.",
    ),

    # ---- Ch5 topological membership generation ----
    Experiment(
        id="ch5-gated-minimax-all", title="Full gated-minimax selection + MF pipeline",
        chapter="Ch5", produces="Tables 5.1/5.2, Figs fig1-fig11", repo="gated-minimax-selection",
        command=["python", "run_all.py"], hardware="any",
        outputs=["gated-minimax-selection/outputs/results.json"],
        notes="Runs on the root .venv (no submodule pyproject). Deterministic.",
    ),

    # ---- Ch3 pVAT / clustering experiments ----
    Experiment(
        id="ch3-adversarial-eval", title="Adversarial clustering-quality eval (ARI grid)",
        chapter="Ch3", produces="Table 3.2", repo="tribble-cluster",
        command=_uv("experiments/adversarial_eval.py"),
    ),
    Experiment(
        id="ch3-autok-eval", title="Auto-k selection eval", chapter="Ch3",
        produces="Ch3 numbers", repo="tribble-cluster",
        command=_uv("experiments/autok_eval.py"),
    ),
    Experiment(
        id="ch3-boruvka-gpu", title="GPU Boruvka MST vs serial Prim",
        chapter="Ch3", produces="Ch3 GPU numbers", repo="tribble-cluster",
        command=_uv("experiments/boruvka_gpu.py"), hardware="gpu",
        notes="Requires a CUDA GPU; skipped on CPU-only hosts.",
    ),

    # ---- Appendix A.3 optimization engine ----
    Experiment(
        id="appA-tsp-compare", title="TSP solver comparison (NN/2-opt/3-opt/LK)",
        chapter="App", produces="App A.3 tables", repo="tribble-opt",
        command=_uv("samples/tsp-demo.py"),
    ),
]


def by_id(exp_id):
    for e in EXPERIMENTS:
        if e.id == exp_id:
            return e
    return None
