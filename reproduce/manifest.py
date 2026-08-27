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
    id: str  # stable short id, e.g. "ch4-mog-concrete"
    title: str  # human description
    chapter: str  # "Ch3".."Ch6", "App" -- proposal chapter
    produces: str  # "Table 4.1", "Fig 3.2", "SUMMARY_REPORT.md", ...
    repo: str  # submodule dir (cwd), relative to repo root
    command: list  # argv to run in `repo` (already env-prefixed)
    hardware: str = "any"  # any | cpu-parallel | gpu | big-mem
    datasets: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    notes: str = ""


# `uv run` executes inside a submodule's own locked environment.
def _uv(*args):
    return ["uv", "run", "python", *args]


def _cluster_exp(name):
    """Run a ClusteringExperiments script with figures redirected into reproduce/.

    Runs from the repo ROOT. grad-school #26 moved these out of the
    tribble-cluster submodule into ClusteringExperiments/, so they are now plain
    sibling modules here; the runner puts that directory on sys.path and rebinds
    FIG_DIR to reproduce/outputs/figures/cluster/ so a regenerated Chapter 3
    figure lands with the rest of the evidence. tribble-cluster is still needed
    for the `tribbleclustering` library itself, and scipy lives in its `dev`
    extra, hence --with.
    """
    return [
        "uv",
        "run",
        "--project",
        "tribble-cluster",
        "--with",
        "scipy",
        "python",
        "reproduce/experiments/run_cluster_experiment.py",
        name,
    ]


def _uvm(module, *args):
    """Run a script as a module (`python -m pkg.mod`) from a repo root.

    Kept for entries that genuinely need package semantics. The Chapter 3
    clustering experiments no longer do: grad-school #26 moved them out of the
    tribble-cluster submodule into ClusteringExperiments/ as plain sibling
    modules, and their `from experiments.foo import ...` imports were rewritten
    to match, so they run by path.
    """
    return ["uv", "run", "python", "-m", module, *args]


EXPERIMENTS = [
    # ---- proposal tables (generators live in reproduce/tables, run under a submodule env) ----
    Experiment(
        id="table-3-1-reorder-three-arm",
        title="Three-arm reorder timing: classical cubic / stage-one heap / stage-two dense",
        chapter="Ch3",
        produces="Table 3.1",
        repo="tribble-cluster",
        command=_uv("../reproduce/tables/table_3_1_reorder_three_arm.py"),
        hardware="any",
        outputs=[
            "reproduce/outputs/table_3_1_three_arm.md",
            "reproduce/outputs/table_3_1_three_arm.csv",
        ],
        notes="VERIFIED RUNNING. All three arms compiled; JIT warmed; every arm's ordering "
        "checked bit-identical to stage two. Also the evidence base for the possible "
        "complexity note (Ch9).",
    ),
    Experiment(
        id="table-3-1-pvat-scaling",
        title="Exact pVAT reorder vs a self-contained classical O(N^3) reference",
        chapter="Ch3",
        produces="Table 3.1 (grid of N)",
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
        chapter="Ch6",
        produces="Table 6.1",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_concrete_reconciliation.py"),
        hardware="any",
        datasets=["Concrete"],
        outputs=[
            "reproduce/outputs/table_concrete_reconciliation.md",
            "reproduce/outputs/table_concrete_reconciliation.csv",
        ],
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
        chapter="Ch4",
        produces="Table 4.1 (+ the Ch6 hyperparameter caveat)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_hyperparam_normalization.py"),
        hardware="any",
        datasets=["Concrete"],
        outputs=[
            "reproduce/outputs/table_hyperparam_normalization.md",
            "reproduce/outputs/table_hyperparam_normalization.csv",
        ],
        notes="VERIFIED RUNNING. Settled the Ch6 confound: the apparent inversion was "
        "mostly library-default hyperparameters, though at 10 seeds the swing is ~0.10 and "
        "most of it is normalization, not tuning. "
        "Also shows normalization helps every fuzzy model and is worth exactly zero "
        "to CART/RF (rank-based splits are transform-invariant).",
    ),
    Experiment(
        id="table-4-4-openset",
        title="Open-set detection: complement rule vs one-class SVM / isolation forest",
        chapter="Ch4",
        produces="Table 4.7 + Table 4.6 / Fig 4.2 (theta sweep)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_4_openset.py"),
        hardware="any",
        datasets=["Glass (BETH if present)"],
        outputs=[
            "reproduce/outputs/table_4_4_openset.md",
            "reproduce/outputs/table_4_4b_theta_sweep.md",
        ],
        notes="VERIFIED RUNNING. Leave-one-class-out open-set protocol; baselines matched to "
        "the complement rule's observed false-alarm rate. BETH is not in the repo, so "
        "the same protocol runs on in-repo Glass. REPRO_THETA_SWEEP is a comma-separated "
        "LIST OF THETAS, not a boolean -- REPRO_THETA_SWEEP=1 silently emits a "
        "one-row table at theta=1.0, where the boost saturates and every cell is "
        "zero. Use REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1 for Fig 4.2.",
    ),
    Experiment(
        id="table-4-11-beth-anomaly",
        title="BETH host telemetry: one-class anomaly detection on the full 763k training split",
        chapter="Ch4",
        produces="Table 4.11 + Table 4.11(b) (BETH false-alarm operating curve)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_11_beth_anomaly.py"),
        hardware="any",
        datasets=["BETH (data/beth/, gitignored -- see data/.gitignore)"],
        outputs=[
            "reproduce/outputs/table_4_11_beth_anomaly.md",
            "reproduce/outputs/table_4_11_beth_fa_sweep.md",
        ],
        notes="VERIFIED RUNNING (10 seeds, 1m39s). This is the BETH experiment "
        "table_4_4_openset.py could not run: leave-one-class-out needs >=3 classes and "
        "BETH is binary. ONE-CLASS BY NECESSITY -- all 158,432 positives are in the "
        "test split, train (763,144) and val (188,967) are 100% benign, so a supervised "
        "RF fits without error and predicts constant 0. NO supervised rows are emitted; "
        "ISOLATION FOREST IS THE RF-FAMILY ONE-CLASS DETECTOR and is the RF-shaped arm "
        "this task admits. The fuzzy arms use the library API -- "
        "tribblefis.one_class.TribbleOneClassDetector -- NOT a hand-assembly of the "
        "multi-class gauss_math path (an earlier revision did that; both give the same "
        "operating point, but the hand-rolled one could only emit a hard label). "
        "TWO SCORE MODES ARE REPORTED AND ONLY ONE IS READABLE: score=surprisal gives "
        "AUC 0.990, score=complement (the library default, and Chapter 4's "
        "formulation) gives 0.928 on the SAME model -- they are monotone transforms in "
        "exact arithmetic, so the gap is pure float64 resolution loss. BETH resolves "
        "4,002 distinct feature vectors; surprisal recovers 3,997 of them, complement "
        "only 1,508. The library docstring puts complement saturation past ~60 "
        "features; BETH hits it at 8 because the log-scaled pid/tid columns are "
        "heavy-tailed. At a 0.1% budget the complement collapses to det=0.000 while "
        "surprisal holds 0.993. Two columns are dropped before any fit: `sus` is BETH's "
        "second LABEL (1 for 100% of evil rows) and `timestamp` separates the files, "
        "not the behaviour -- the drop is here, not in load_beth(), so "
        "table_4_4_openset's archived numbers do not move. The threshold is the "
        "(1-budget) quantile of benign-VALIDATION scores (REPRO_BETH_FA_BUDGET, default "
        "0.01), and the finding is that it DOES NOT TRANSFER: 0.0100 val false alarm "
        "becomes 0.1500 on test, 15x. Table 4.11(b) sweeps the budget and shows the "
        "TIGHTEST budget is the best operating point (J +0.870 at 0.1% vs +0.843 at "
        "1%), so the default is not optimal. n_jobs capped at 8 (REPRO_BETH_N_JOBS) and "
        "BLAS threads at 8 (REPRO_BLAS_THREADS, set before the numpy import) because "
        "n_jobs=-1 on a 32-core host hung the machine and segfaulted the process; that "
        "was thread oversubscription, never memory (peak RSS 521 MB of 95.6 GB).",
    ),
    Experiment(
        id="table-g5-output-partitioning",
        title="G5: uniform vs quantile vs hybrid output partitioning",
        chapter="Ch4",
        produces="Table 4.2 (Goal G5)",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_g5_output_partitioning.py"),
        hardware="any",
        datasets=["Concrete"],
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
        chapter="Ch4",
        produces="Table 4.3 (Goal G5)",
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
        chapter="Ch4",
        produces="No numbered prose table -- backs TNORM_REEVALUATION_RESULTS.md",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_norm_conorm_matrix.py"),
        hardware="any",
        datasets=["Concrete", "PhiUSIIL"],
        outputs=[
            "reproduce/outputs/table_norm_conorm_matrix.md",
            "reproduce/outputs/table_norm_conorm_matrix.csv",
        ],
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
        id="table-4-8-mf-dedup",
        title="MF deduplication: reduction vs. tolerance across six problems, "
        "plus the correction-rule pass quantified (Glass)",
        chapter="Ch4",
        produces="Tables 4.8 and 4.9, Fig 4.3",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_8_mf_dedup.py"),
        hardware="any",
        datasets=["Glass", "Wine", "BreastCancer", "Digits", "Concrete", "Diabetes"],
        outputs=[
            "reproduce/outputs/table_4_8_mf_dedup.md",
            "reproduce/outputs/table_4_8_mf_dedup_sweep.md",
            "reproduce/outputs/table_4_9_correction_pass.md",
        ],
        notes="Wine/BreastCancer/Digits/Diabetes are scikit-learn-bundled (no network, no "
        "missing-file risk) rather than the dissertation's other named datasets "
        "(PhiUSIIL, RT-IOT2022), which are not in this repository. Max-lossless "
        "tolerance is dataset-dependent: 2x (Diabetes) to 10x (Wine, Concrete); "
        "reduction at that boundary ranges 0% (BreastCancer) to 44.2% (Digits). "
        "Table 4.9's cascade-flatten row isolates the mechanism cost (dropping the "
        "gating logic) from the dedup-tolerance cost by running at rtol=atol=0. "
        "Filed upstream as tribble-fis#85 (dedup tolerance is a hardcoded module "
        "constant with no parameter to sweep it).",
    ),
    Experiment(
        id="table-4-1-mog-baselines",
        title="MoG FIS vs sklearn baselines (train time + accuracy/R2)",
        chapter="Ch4",
        produces="Tables 4.4 and 4.5",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_4_1_mog_baselines.py"),
        hardware="any",
        datasets=["Concrete", "PhiUSIIL"],
        outputs=["reproduce/outputs/table_4_1.md", "reproduce/outputs/table_4_1.csv"],
        notes="ANFIS / GA-FIS / M5 columns fill in only if those adapters are available; else N/A.",
    ),
    Experiment(
        id="table-a7-regression-scale",
        title="Large-scale regression benchmark: model family on California Housing / Superconductivity",
        chapter="App",
        produces="Appendix A.7.1",
        repo="tribble-fis",
        command=_uv("../reproduce/tables/table_a7_regression_scale.py"),
        hardware="any",
        datasets=["California Housing", "Superconductivity"],
        outputs=[
            "reproduce/outputs/table_a7_regression_scale.md",
            "reproduce/outputs/table_a7_regression_scale.csv",
        ],
        notes="Supersedes the single-seed pilot in reproduce/regression_scale/ "
        "(CHECKLIST C13) -- ten seeds, canonically sourced (California Housing via "
        "sklearn.fetch_california_housing(), Superconductivity via UCI id 464 "
        "direct download, both resolved 2026-08-12). VERIFIED RUNNING. Random "
        "Forest wins both datasets cleanly. Flat MoG and HME are unstable on "
        "Superconductivity even after the FeatureAgglomeration decorrelation the "
        "pilot found necessary -- occasionally catastrophic negative R2, echoing "
        "the seed-9 HME divergence table_concrete_reconciliation already documents "
        "on Concrete.",
    ),
    Experiment(
        id="table-6-1-model-family",
        title="Flat / fuzzy-tree / HME vs CART/M5 baselines",
        chapter="Ch6",
        produces="Table 6.2",
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
        id="ch4-mog-concrete",
        title="MoG-TSK on UCI Concrete (regression)",
        chapter="Ch4",
        produces="Fig 4.1 / Ch4 numbers",
        repo="tribble-fis",
        command=_uv("../FuzzySystemsExperiments/concrete.py"),
        datasets=["Concrete"],
    ),
    Experiment(
        id="ch4-mog-phiusiil",
        title="MoG classifier on PhiUSIIL phishing",
        chapter="Ch4",
        produces="Ch4 numbers",
        repo="tribble-fis",
        command=_uv("../FuzzySystemsExperiments/phiusiil.py"),
        datasets=["PhiUSIIL"],
    ),
    Experiment(
        id="ch4-mog-iot",
        title="MoG classifier on RT-IOT2022",
        chapter="Ch4",
        produces="Ch4 numbers",
        repo="tribble-fis",
        command=_uv("../FuzzySystemsExperiments/iot.py"),
        datasets=["RT-IOT2022"],
        hardware="big-mem",
        notes="123K x 83, 12 classes; large.",
    ),
    Experiment(
        id="ch6-tree-concrete",
        title="Fuzzy tree + HME vs flat on Concrete",
        chapter="Ch6",
        produces="Fig 6.1 (and the tuned config Table 6.1 replicates)",
        repo="tribble-fis",
        command=_uv("tribble-tree/demo_concrete.py"),
        datasets=["Concrete"],
    ),
    Experiment(
        id="ch6-tree-phishing",
        title="Fuzzy tree + HME on PhiUSIIL",
        chapter="Ch6",
        produces="Fig 6.1",
        repo="tribble-fis",
        command=_uv("tribble-tree/demo_phishing.py"),
        datasets=["PhiUSIIL"],
    ),
    Experiment(
        id="ch6-mimo-pendulum",
        title="MIMO memory FIS on double pendulum",
        chapter="Ch6",
        produces="Table 6.4 / Fig 6.3",
        repo="tribble-fis",
        command=_uv("../AnalyticalDynamics/test_double_pendulum.py"),
        notes="Moved to AnalyticalDynamics/ by grad-school #26. Entry point still unconfirmed.",
    ),
    # ---- Ch5 topological membership generation ----
    Experiment(
        id="ch5-gated-minimax-all",
        title="Full gated-minimax selection + MF pipeline",
        chapter="Ch5",
        produces="Tables 5.1/5.2/5.3, Figs fig1-fig11",
        repo="gated-minimax-selection",
        command=["python", "run_all.py"],
        hardware="any",
        outputs=["gated-minimax-selection/outputs/results.json"],
        notes="Runs on the root .venv (no submodule pyproject). Deterministic.",
    ),
    Experiment(
        id="table-5-x-ch5-selection",
        title="Ch5 Tables 5.1/5.2/5.3 rendered from the gated-minimax results of record",
        chapter="Ch5",
        produces="Tables 5.1, 5.2, 5.3",
        repo=".",
        command=["python3", "reproduce/tables/table_5_1_3_ch5_tables.py"],
        hardware="any",
        outputs=[
            "reproduce/outputs/table_5_1_battery.md",
            "reproduce/outputs/table_5_2_multiscale.md",
            "reproduce/outputs/table_5_3_selection.md",
        ],
        notes="Pure renderer -- reads gated-minimax-selection/outputs/results.json and "
        "does no computation, so run ch5-gated-minimax-all first if the JSON needs "
        "regenerating. Stdlib only; no submodule environment required. Exists so "
        "the Ch5 tables stop being hand-transcribed: drift now shows as a diff.",
    ),
    Experiment(
        id="table-5-4-ch5-g1-scaling",
        title="Goal G1 scaling decision rule: two-stage selector vs. flat "
        "set-cover, n=100..5000, ten seeds (one-pass arm not yet built)",
        chapter="Ch5",
        produces="Table 5.4",
        repo=".",
        command=["python", "reproduce/tables/table_5_4_ch5_g1_scaling.py"],
        hardware="any",
        outputs=[
            "reproduce/outputs/table_5_4_ch5_g1_scaling.md",
            "reproduce/outputs/table_5_4_ch5_g1_scaling.csv",
            "reproduce/outputs/table_5_4_ch5_g1_scaling_raw.csv",
        ],
        notes="Does its OWN computation (unlike table-5-x above) -- imports "
        "ivat_mf/selection/multiscale_persistence/battery_hierarchical directly "
        "from gated-minimax-selection/ on the root .venv. Runs the two arms of "
        "Goal G1's decision rule that exist today (07-goals-for-completion.md: "
        "'Phase five, the one-pass refactor, is plumbing and unattempted') at the "
        "full size grid the decision rule names, extending "
        "gated-minimax-selection/notes/SCALING_STUDY.md (single seed, two-stage "
        "only) with the flat baseline, partition-of-unity error, and a ten-seed "
        "spread. Takes ~3 minutes on the 2026-08 workstation.",
    ),
    # ---- Ch3 pVAT / clustering experiments ----
    Experiment(
        id="ch3-adversarial-eval",
        title="Adversarial clustering-quality eval (ARI grid)",
        chapter="Ch3",
        produces="Table 3.4",
        repo=".",
        command=_cluster_exp("adversarial_eval"),
        outputs=[
            "reproduce/outputs/figures/cluster/adversarial_eval.png",
            "ClusteringExperiments/findings/ADVERSARIAL_EVAL_FINDINGS.md",
        ],
    ),
    Experiment(
        id="ch3-principled-stitch",
        title="Stitch ablation on two moons: fps reps x top-m cross-edges",
        chapter="Ch3",
        produces="Table 3.5",
        repo=".",
        command=_cluster_exp("principled_stitch"),
        outputs=[
            "reproduce/outputs/figures/cluster/principled_stitch_two_moons.png",
            "reproduce/outputs/figures/cluster/principled_stitch_circles.png",
            "ClusteringExperiments/findings/GAPS_FINDINGS.md",
        ],
        notes="Table 3.5's four rows are the ablation grid. Numbers currently quoted in "
        "the prose match GAPS_FINDINGS.md; not yet re-run under this harness.",
    ),
    Experiment(
        id="ch3-hardening-eval",
        title="Agreement with exact single-linkage under non-metric dissimilarities",
        chapter="Ch3",
        produces="Table 3.6",
        repo=".",
        command=_cluster_exp("hardening_eval"),
        outputs=[
            "reproduce/outputs/figures/cluster/hardening_partition_robustness.png",
            "ClusteringExperiments/findings/HARDENING_FINDINGS.md",
        ],
        notes="Fractional Minkowski p=0.5 (14.1% triangle violations), cosine, and "
        "kNN-geodesic all reproduce the exact ordering (agreement 1.0).",
    ),
    Experiment(
        id="table-3-7-g2-dtw-nonmetric",
        title="Goal G2: exact reorder + triangle-inequality rate on real DTW dissimilarity matrices",
        chapter="Ch3",
        produces="Table 3.7 (last row)",
        repo="tribble-cluster",
        command=[
            "uv",
            "run",
            "--with",
            "aeon",
            "python",
            "../reproduce/tables/table_3_7_g2_dtw_nonmetric.py",
        ],
        hardware="any",
        datasets=["ECG5000, FordA, Crop (UCR/UEA via aeon)"],
        outputs=[
            "reproduce/outputs/table_3_7_g2_dtw_nonmetric.md",
            "reproduce/outputs/table_3_7_g2_dtw_nonmetric.csv",
        ],
        notes="Fills Table 3.7's 'not run -- no non-coordinate dataset in the harness "
        "(Goal G2)' row, the single most important credibility gap per Chapter 7's "
        "Goal G2. VERIFIED RUNNING on all three datasets (2026-08-12): exactness 1.000 "
        "on ECG5000/FordA/Crop; triangle-inequality violations 20.9%/0.4%/23.6% "
        "(FordA below the 14% synthetic proxy, the other two above it). Covers "
        "decision-rule items 1 (exactness), 2 (triangle-inequality rate) and 4 (Crop "
        "at 24,000 points, the scale target, ~4.6 GB, 1597s matrix + 4.7s reorder). "
        "Item 3 (downstream usefulness) is a separate script, see "
        "table-3-7-g2-downstream below. `uv pip install aeon` does not persist under "
        "this project's lockfile resync, hence --with aeon on the invocation. "
        "REPRO_G2_DATASETS selects which datasets run (default ECG5000 only); FordA "
        "(~2h matrix) and Crop (~27min matrix) both took a separate, flagged, "
        "explicitly-approved run given their cost.",
    ),
    Experiment(
        id="table-3-7-g2-downstream",
        title="Goal G2 decision-rule item 3: set-cover vs. NERFCM-given-k on real DTW matrices",
        chapter="Ch3",
        produces="Table 3.7 companion (downstream usefulness)",
        repo="tribble-cluster",
        command=[
            "uv",
            "run",
            "--with",
            "aeon",
            "python",
            "../reproduce/tables/table_3_7_g2_downstream.py",
        ],
        hardware="any",
        datasets=["ECG5000, Crop, FordA (UCR/UEA via aeon)"],
        outputs=[
            "reproduce/outputs/table_3_7_g2_downstream.md",
            "reproduce/outputs/table_3_7_g2_downstream.csv",
        ],
        notes="Reuses NERFCM and Chapter 5's select_coverage_cover/select_multiscale "
        "unmodified from gated-minimax-selection/ -- both already matrix-only and "
        "already discover k, so this is a new caller, not new algorithm code. "
        "VERIFIED RUNNING on all three datasets (2026-08-12). Result is mixed, not a "
        "clean pass: on ECG5000 (the one dataset with real recoverable structure) the "
        "set-cover BEATS NERFCM-given-k by 0.122 ARI (0.715 vs 0.593), which fails the "
        "decision rule's +/-0.05 tolerance in the favorable direction; on Crop and "
        "FordA both methods score low in absolute terms and land within 0.05 of each "
        "other, a low-information pass. Only 2 of 3 tested sets meet the criterion "
        "literally, so the 'at least three of five' threshold is not yet satisfied. "
        "ConiVAT and bottleneck-bootstrap are N/A -- genuine implementation gaps, not "
        "missing call sites (see the script's docstring). REPRO_G2_DOWNSTREAM_DATASETS "
        "selects which dataset(s) run; each rebuilds its own DTW matrix rather than "
        "sharing one with the sibling script.",
    ),
    Experiment(
        id="ch3-autok-eval",
        title="Auto-k selection eval",
        chapter="Ch3",
        produces="Ch3 numbers",
        repo="tribble-cluster",
        command=["python", "ClusteringExperiments/autok_eval.py"],
    ),
    Experiment(
        id="ch3-boruvka-gpu",
        title="GPU Boruvka MST vs serial Prim",
        chapter="Ch3",
        produces="Ch3 GPU numbers",
        repo="tribble-cluster",
        command=["python", "ClusteringExperiments/boruvka_gpu.py"],
        hardware="gpu",
        notes="Requires a CUDA GPU; skipped on CPU-only hosts.",
    ),
    # ---- Appendix A.3 optimization engine ----
    Experiment(
        id="appA-tsp-compare",
        title="TSP solver comparison (NN/2-opt/3-opt/LK)",
        chapter="App",
        produces="App A.3 tables",
        repo="tribble-opt",
        command=_uv("samples/tsp-demo.py"),
    ),
]


def by_id(exp_id):
    for e in EXPERIMENTS:
        if e.id == exp_id:
            return e
    return None
