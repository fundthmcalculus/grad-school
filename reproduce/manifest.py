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


def _cluster_exp(name):
    """Run a tribble-cluster experiment with its figures redirected into this repo.

    Runs from the repo ROOT, not the submodule, because the runner lives here.
    Left to itself each experiment writes into
    `tribble-cluster/experiments/figures/`, so reproducing a Chapter 3 figure
    dirties a pinned submodule and files the evidence for a grad-school table
    inside a library. The runner redirects to reproduce/outputs/figures/cluster/
    and puts the submodule root on sys.path so the absolute `experiments.*`
    imports resolve. scipy lives in tribble-cluster's `dev` extra, hence --with.
    """
    return ["uv", "run", "--project", "tribble-cluster", "--with", "scipy",
            "python", "reproduce/experiments/run_cluster_experiment.py", name]


def _uvm(module, *args):
    """Run a script as a module (`python -m pkg.mod`) from the submodule root.

    Most of tribble-cluster's experiments do `from experiments.blockwise_vat
    import ...`, which needs the submodule ROOT on sys.path. Invoking them by
    path (`python experiments/foo.py`) puts `experiments/` there instead, and
    every one of them dies with `ModuleNotFoundError: No module named
    'experiments'` before doing any work. The module form is the only one that
    runs, so entries here use it rather than the path.
    """
    return ["uv", "run", "python", "-m", module, *args]


EXPERIMENTS = [
    # ---- proposal tables (generators live in reproduce/tables, run under a submodule env) ----
    Experiment(
        id="table-3-1-reorder-three-arm",
        title="Three-arm reorder timing: classical cubic / stage-one heap / stage-two dense",
        chapter="Ch3", produces="Table 3.1",
        repo="tribble-cluster",
        command=_uv("../reproduce/tables/table_3_1_reorder_three_arm.py"),
        hardware="any",
        outputs=["reproduce/outputs/table_3_1_three_arm.md",
                 "reproduce/outputs/table_3_1_three_arm.csv"],
        notes="VERIFIED RUNNING. All three arms compiled; JIT warmed; every arm's ordering "
              "checked bit-identical to stage two. Also the evidence base for the possible "
              "complexity note (Ch9).",
    ),
    Experiment(
        id="table-3-1-pvat-scaling",
        title="Exact pVAT reorder vs a self-contained classical O(N^3) reference",
        chapter="Ch3", produces="Table 3.1 (grid of N)",
        repo="tribble-cluster",
        command=_uv("../reproduce/tables/table_3_1_pvat_scaling.py"),
        hardware="any",
        outputs=["reproduce/outputs/table_3_1.md", "reproduce/outputs/table_3_1.csv"],
        notes="Needs scipy, which tribble-cluster keeps under its `dev` extra -- "
              "run_all_tables.sh supplies it via EXTRA_DEPS='--with scipy'. The cubic "
              "reference is capped at N<=1024 (REPRO_NAIVE_CAP), so the larger rows are "
              "pVAT-only. Table 3.1's headline 4,096-point pair is NOT from this script -- it "
              "is cited from the NAFIPS work. Raising the cap to 4096 is one flag but "
              "costs ~64x the cubic time for a constant factor the chapter does not "
              "rest on; the scaling claim comes from the swept grid and the three-arm "
              "decomposition. See reproduce/PROVENANCE_MAP.md note 1.",
    ),
    Experiment(
        id="table-concrete-reconciliation",
        title="Concrete under ONE protocol -- makes Ch4 and Ch6 numbers comparable",
        chapter="Ch6", produces="Table 6.1",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_concrete_reconciliation.py"),
        hardware="any", datasets=["Concrete"],
        outputs=["reproduce/outputs/table_concrete_reconciliation.md",
                 "reproduce/outputs/table_concrete_reconciliation.csv"],
        notes="VERIFIED RUNNING at 10 seeds (652s on 8 cores; was 1301s before the arms were "
              "parallelised -- output byte-identical either way, REPRO_JOBS=1 for a "
              "readable log). Every model on identical splits/seeds/preprocessing. "
              "The hierarchy does NOT beat flat under a uniform protocol, and CART/RF beat "
              "all fuzzy models. WATCH THE HME ROW: at log+standardized it reads "
              "R2 = -220.9 +/- 665.0 because seed 9 diverges (predictions to 10,536 MPa on "
              "a <=82 MPa target). The other nine seeds give 0.805 +/- 0.059. A 5-seed run "
              "misses it entirely -- this is the seed-count lesson in one cell.",
    ),
    Experiment(
        id="table-hyperparam-normalization",
        title="Concrete: model x hyperparameters x normalization",
        chapter="Ch4", produces="Table 4.1 (+ the Ch6 hyperparameter caveat)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_hyperparam_normalization.py"),
        hardware="any", datasets=["Concrete"],
        outputs=["reproduce/outputs/table_hyperparam_normalization.md",
                 "reproduce/outputs/table_hyperparam_normalization.csv"],
        notes="VERIFIED RUNNING. Settled the Ch6 confound: the apparent inversion was "
              "mostly library-default hyperparameters, though at 10 seeds the swing is ~0.10 and "
              "most of it is normalization, not tuning. "
              "Also shows normalization helps every fuzzy model and is worth exactly zero "
              "to CART/RF (rank-based splits are transform-invariant).",
    ),
    Experiment(
        id="table-4-4-openset",
        title="Open-set detection: complement rule vs one-class SVM / isolation forest",
        chapter="Ch4", produces="Table 4.7 + Table 4.6 / Fig 4.2 (theta sweep)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_4_openset.py"),
        hardware="any", datasets=["Glass (BETH if present)"],
        outputs=["reproduce/outputs/table_4_4_openset.md",
                 "reproduce/outputs/table_4_4b_theta_sweep.md"],
        notes="VERIFIED RUNNING. Leave-one-class-out open-set protocol; baselines matched to "
              "the complement rule's observed false-alarm rate. BETH is not in the repo, so "
              "the same protocol runs on in-repo Glass. REPRO_THETA_SWEEP is a comma-separated "
              "LIST OF THETAS, not a boolean -- REPRO_THETA_SWEEP=1 silently emits a "
              "one-row table at theta=1.0, where the boost saturates and every cell is "
              "zero. Use REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1 for Fig 4.2.",
    ),
    Experiment(
        id="table-g5-output-partitioning",
        title="G5: uniform vs quantile vs hybrid output partitioning",
        chapter="Ch4", produces="Table 4.2 (Goal G5)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_g5_output_partitioning.py"),
        hardware="any", datasets=["Concrete"],
        outputs=["reproduce/outputs/table_g5_output_partitioning.md"],
        notes="VERIFIED RUNNING at 10 seeds. NO crossover: the 3-seed 'uniform wins at 3, "
              "quantile at 6' reading is retracted -- the largest gap in all 18 configs is "
              "0.012 R2 against sigma 0.02-0.03. The starvation diagnostic IS real (uniform min "
              "bucket 132->75->39 vs quantile 343->257->171); Concrete skew (+0.42) is just too "
              "mild for it to reach the aggregate error. Extreme-pinning was inert pre-#29 and "
              "is now live.",
    ),
    Experiment(
        id="table-g5b-skew-sweep",
        title="G5b: partitioning vs target skew (synthetic, skew isolated)",
        chapter="Ch4", produces="Table 4.3 (Goal G5)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_g5b_skew_sweep.py"),
        hardware="any",
        outputs=["reproduce/outputs/table_g5b_skew_sweep.md"],
        notes="VERIFIED RUNNING at 10 seeds. H2 REFUTED -- the monotone +0.003 -> +0.201 climb was "
              "a 3-seed artifact. Q-U is negative in every row past symmetry (to -11.8), but "
              "read the SPREADS: quantile destabilises (+/-0.99, +/-4.4, +/-21.2) rather than "
              "becoming inaccurate, while uniform decays smoothly. Starvation confirmed (min "
              "occupancy 11 -> 0). H3 still reversed: quantile holds the TAILS better. G5 is "
              "REOPENED -- 'quantile by default' is withdrawn.",
    ),
    Experiment(
        id="table-norm-conorm-matrix",
        title="Norm/conorm sweep: the five De Morgan pairs x model x dataset",
        chapter="Ch4", produces="No numbered prose table -- backs TNORM_REEVALUATION_RESULTS.md",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_norm_conorm_matrix.py"),
        hardware="any", datasets=["Concrete", "PhiUSIIL"],
        outputs=["reproduce/outputs/table_norm_conorm_matrix.md",
                 "reproduce/outputs/table_norm_conorm_matrix.csv"],
        notes="Answers whether the fuzzy operator choice matters at all -- previously "
              "an unexamined default. Needs tribble-fis#32: before it, regression could "
              "not select an operator (tsk_firing_strengths read it off the anomaly "
              "parameters, which regression never supplies) and every regressor "
              "silently ran min/max. Columns differ by model: flat MoG uses both "
              "operators, the fuzzy tree the t-norm only, and the HME row its experts "
              "only -- the HME gate is a product by construction. Mixed (non-De Morgan) "
              "pairs are an opt-in advanced setting and are deliberately not swept.",
    ),
    Experiment(
        id="table-4-1-mog-baselines",
        title="MoG FIS vs sklearn baselines (train time + accuracy/R2)",
        chapter="Ch4", produces="Tables 4.4 and 4.5",
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
        chapter="Ch6", produces="Table 6.2",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_6_1_model_family.py"),
        notes="Runs the fuzzy arms at RAW preprocessing and library defaults, so its "
              "flat R2 (0.644) is NOT comparable to Table 6.1's uniform-protocol 0.868. "
              "Kept as the external-baseline (CART/RF/M5) source only; the file name "
              "predates the prose renumbering.",
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
        chapter="Ch6", produces="Fig 6.1 (and the tuned config Table 6.1 replicates)",
        repo="tribble-fis",
        command=_uv("tribble-tree/demo_concrete.py"), datasets=["Concrete"],
    ),
    Experiment(
        id="ch6-tree-phishing", title="Fuzzy tree + HME on PhiUSIIL",
        chapter="Ch6", produces="Fig 6.1", repo="tribble-fis",
        command=_uv("tribble-tree/demo_phishing.py"), datasets=["PhiUSIIL"],
    ),
    Experiment(
        id="ch6-mimo-pendulum", title="MIMO memory FIS on double pendulum",
        chapter="Ch6", produces="Table 6.4 / Fig 6.3", repo="tribble-fis",
        command=_uv("tests/test_double_pendulum.py"),
        notes="Confirm exact MIMO-memory entry point; may live under tests/ or gaussian_mixture/.",
    ),

    # ---- Ch5 topological membership generation ----
    Experiment(
        id="ch5-gated-minimax-all", title="Full gated-minimax selection + MF pipeline",
        chapter="Ch5", produces="Tables 5.1/5.2/5.3, Figs fig1-fig11", repo="gated-minimax-selection",
        command=["python", "run_all.py"], hardware="any",
        outputs=["gated-minimax-selection/outputs/results.json"],
        notes="Runs on the root .venv (no submodule pyproject). Deterministic.",
    ),

    Experiment(
        id="table-5-x-ch5-selection",
        title="Ch5 Tables 5.1/5.2/5.3 rendered from the gated-minimax results of record",
        chapter="Ch5", produces="Tables 5.1, 5.2, 5.3",
        repo=".",
        command=["python3", "reproduce/tables/table_5_x_ch5_selection.py"],
        hardware="any",
        outputs=["reproduce/outputs/table_5_1_battery.md",
                 "reproduce/outputs/table_5_2_multiscale.md",
                 "reproduce/outputs/table_5_3_selection.md"],
        notes="Pure renderer -- reads gated-minimax-selection/outputs/results.json and "
              "does no computation, so run ch5-gated-minimax-all first if the JSON needs "
              "regenerating. Stdlib only; no submodule environment required. Exists so "
              "the Ch5 tables stop being hand-transcribed: drift now shows as a diff.",
    ),

    # ---- Ch3 pVAT / clustering experiments ----
    Experiment(
        id="ch3-adversarial-eval", title="Adversarial clustering-quality eval (ARI grid)",
        chapter="Ch3", produces="Table 3.4", repo=".",
        command=_cluster_exp("adversarial_eval"),
        outputs=["reproduce/outputs/figures/cluster/adversarial_eval.png",
                 "tribble-cluster/experiments/findings/ADVERSARIAL_EVAL_FINDINGS.md"],
    ),
    Experiment(
        id="ch3-principled-stitch",
        title="Stitch ablation on two moons: fps reps x top-m cross-edges",
        chapter="Ch3", produces="Table 3.5", repo=".",
        command=_cluster_exp("principled_stitch"),
        outputs=["reproduce/outputs/figures/cluster/principled_stitch_two_moons.png",
                 "reproduce/outputs/figures/cluster/principled_stitch_circles.png",
                 "tribble-cluster/experiments/findings/GAPS_FINDINGS.md"],
        notes="Table 3.5's four rows are the ablation grid. Numbers currently quoted in "
              "the prose match GAPS_FINDINGS.md; not yet re-run under this harness.",
    ),
    Experiment(
        id="ch3-hardening-eval",
        title="Agreement with exact single-linkage under non-metric dissimilarities",
        chapter="Ch3", produces="Table 3.6", repo=".",
        command=_cluster_exp("hardening_eval"),
        outputs=["reproduce/outputs/figures/cluster/hardening_partition_robustness.png",
                 "tribble-cluster/experiments/findings/HARDENING_FINDINGS.md"],
        notes="Fractional Minkowski p=0.5 (14.1% triangle violations), cosine, and "
              "kNN-geodesic all reproduce the exact ordering (agreement 1.0).",
    ),
    Experiment(
        id="ch3-autok-eval", title="Auto-k selection eval", chapter="Ch3",
        produces="Ch3 numbers", repo="tribble-cluster",
        command=_uvm("experiments.autok_eval"),
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
