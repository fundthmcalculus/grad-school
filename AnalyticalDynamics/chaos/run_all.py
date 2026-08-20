"""One-command reproduction of every number and figure paper.md cites.

Runs seven stages in order, each writing one JSON file under results/ plus
whatever PNGs belong to it. A stage is skipped if its cached JSON already
matches a hash of that stage's declared inputs (see pipeline_cache.py) --
fits are deterministic (random_state is fixed throughout), so re-running an
unchanged pipeline reproduces the same hash and does no work.

    data            pendulum_data.generate_all() + provenance checks -> data/*.npz
    sweep           main hyperparameter grid, n in {2,3,5} x friction         (~2.5 h)
    lowcap          low-capacity follow-up grid                               (~10 min)
    select          refit each dataset's winner, Table 1                    (~20 min)
    bracket         bracket-decorrelation diagnostic, Tables 2-3               (~10 s)
    capacity        capacity/extrapolation comparison, Table 4                (~5 min)
    representation  wrap / hysteresis / sin-cos study, Tables 6-8            (~90 min)

Run:
    python run_all.py                       # reproduce everything, cached
    python run_all.py --fresh               # ignore every cache
    python run_all.py --fresh sweep lowcap  # force just these; downstream stages
                                             # only redo their own work if these
                                             # stages' *output* actually changed
    python run_all.py --stage bracket       # run one stage in isolation (its
                                             # upstream JSON must already exist)

sweep, lowcap, and representation are the expensive stages (~2.5 h, ~10 min, and
~90 min respectively) -- data/select/bracket/capacity are seconds to a few
minutes. Timings are for the fixed n in {2, 3, 5} grid this pipeline always
requests; sweep.py's own `--n`/`--out` flags remain for exploring an additional
chain length outside the pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np

import bracket_diagnostic
import compare_families
import gen_n2_rollout_comparison
import gen_rollout_error_vs_n
import paper_results as pr
import pendulum_data as pdata
import pipeline_cache
import plots
import sweep
import wrap_sweep
from fis_timestep import (
    FisConfig,
    RESULT_DIR,
    baseline_bracket_midpoint,
    baseline_nearest,
    load,
    predictions_for,
    run,
)

STAGE_ORDER = [
    "data",
    "sweep",
    "lowcap",
    "select",
    "bracket",
    "capacity",
    "representation",
]

DATA_PATH = RESULT_DIR / "data.json"
SWEEP_PATH = RESULT_DIR / "sweep.json"
LOWCAP_PATH = RESULT_DIR / "sweep_lowcap.json"
SELECT_PATH = RESULT_DIR / "select.json"
BRACKET_PATH = RESULT_DIR / "bracket.json"
CAPACITY_PATH = RESULT_DIR / "capacity.json"
REPRESENTATION_PATH = RESULT_DIR / "representation.json"

DATASETS = [(n, f) for n in pdata.N_LINKS for f in (False, True)]


# ---------------------------------------------------------------------------
# Stage 1: data
# ---------------------------------------------------------------------------
def stage_data(fresh):
    hash_of = {
        "theta1_deg": pdata.THETA1_DEG,
        "train_theta2_grid": pdata.TRAIN_THETA2_DEG.tolist(),
        "test_theta2_deg": pdata.TEST_THETA2_DEG,
        "damping": pdata.DAMPING,
        "t_end": pdata.T_END,
        "n_steps": pdata.N_STEPS,
        "test_t_end": pdata.TEST_T_END,
        "n_links": list(pdata.N_LINKS),
    }
    h = pipeline_cache.stage_hash(hash_of)
    if "data" not in fresh:
        cached = pipeline_cache.load_if_fresh(DATA_PATH, h)
        if cached is not None:
            print("[data] cached")
            return cached, h

    print("[data] generating datasets from scratch, with provenance checks ...")
    provenance = pdata.collect_provenance()
    datasets = pdata.generate_all()
    payload = {"provenance": provenance, "datasets": datasets}
    pipeline_cache.write_stage(DATA_PATH, "data", h, hash_of, payload)
    print(f"[data] wrote {DATA_PATH.name}: {len(datasets)} datasets")
    return payload, h


# ---------------------------------------------------------------------------
# Stages 2-3: sweep, lowcap
# ---------------------------------------------------------------------------
def _run_sweep_stage(name, path, lowcap, data_hash, fresh):
    cfgs = sweep.configs(lowcap=lowcap)
    hash_of = {
        "data_hash": data_hash,
        "n_links": list(pdata.N_LINKS),
        "configs": [c.key() for c in cfgs],
    }
    h = pipeline_cache.stage_hash(hash_of)
    if name not in fresh:
        cached = pipeline_cache.load_if_fresh(path, h)
        if cached is not None:
            print(f"[{name}] cached ({len(cached['rows'])} rows)")
            return cached, h

    print(f"[{name}] scoring {len(cfgs)} configs x {len(DATASETS)} datasets ...")
    rows = sweep.run_sweep(cfgs, DATASETS, log=lambda m: print(f"  {m}", flush=True))
    payload = {"rows": rows}
    pipeline_cache.write_stage(path, name, h, hash_of, payload)
    print(f"[{name}] wrote {path.name} ({len(rows)} rows)")
    return payload, h


def stage_sweep(data_hash, fresh):
    return _run_sweep_stage("sweep", SWEEP_PATH, False, data_hash, fresh)


def stage_lowcap(data_hash, fresh):
    return _run_sweep_stage("lowcap", LOWCAP_PATH, True, data_hash, fresh)


# ---------------------------------------------------------------------------
# Stage 4: select -- refit each dataset's winner (Table 1)
# ---------------------------------------------------------------------------
def config_from_key(key):
    """Rebuild a FisConfig from the key sweep.py wrote.

    Round-tripping the key rather than re-deriving it keeps the refit provably
    identical to the swept run; a mismatch would silently report a model that was
    never actually scored.
    """
    parts = key.split("_")
    cfg = FisConfig(
        n_output_buckets=int(parts[0][2:]),
        tsk_order=parts[1],
        n_gaussians=int(parts[2][1:]),
        output_partition=parts[3],
        consequent_basis=parts[4],
        norm_conorm=parts[5],
    )
    for tail in parts[6:]:
        if tail.startswith("harmonic"):
            cfg.encoding, cfg.n_harmonics = "harmonic", int(tail[len("harmonic") :])
        elif tail.startswith("l2"):
            cfg.l2_reg = float(tail[2:])
    assert cfg.key() == key, f"config round-trip failed: {cfg.key()!r} != {key!r}"
    return cfg


def pick(rows, label, metric, lower_is_better=True):
    cand = [r for r in rows if r["dataset"] == label and metric in r]
    if not cand:
        return None
    return (min if lower_is_better else max)(cand, key=lambda r: r[metric])


def draw_dataset(split, system, friction, base, trained, holdout):
    """Every figure for one dataset, each drawn with that cell's winner.

    Returns the held-out prediction dict so the caller can build the
    error-against-time figure across all datasets.
    """
    bars = {
        "bracket midpoint": (
            base["bracket midpoint (no learning)"]["rmse"],
            base["bracket midpoint (no learning)"]["r2"],
        ),
        "nearest trained IC": (
            base["nearest trained IC (no learning)"]["rmse"],
            base["nearest trained IC (no learning)"]["r2"],
        ),
    }
    holdout_pred = None
    for setting, (model, cfg, res) in (("trained", trained), ("holdout", holdout)):
        pred = predictions_for(split, model, cfg, which=setting)
        if setting == "holdout":
            holdout_pred = pred
        plots.angles_overlay(pred, system, friction, setting)
        plots.trajectory_overlay(pred, system, friction, setting)
        metrics = res.trained_ic if setting == "trained" else res.holdout_ic
        # The no-learning baselines only exist for the held-out IC: they are
        # defined by interpolating *between* trained ICs, so on a trained IC they
        # would just be that trajectory itself.
        plots.compare_cell(
            system,
            friction,
            setting,
            metrics["rmse"],
            metrics["r2"],
            baselines=bars if setting == "holdout" else None,
        )
    return holdout_pred


def _compute_select(sweep_payload, lowcap_payload):
    rows = [
        r for r in sweep_payload["rows"] + lowcap_payload["rows"] if not r.get("error")
    ]
    print(f"  {len(rows)} scored configs (sweep + lowcap, error rows dropped)")

    datasets_out = {}
    selection_rows = []
    fis_holdout_rmse = {}
    capacity = {}
    holdout_preds = {}

    for n_links, friction in DATASETS:
        split = load(n_links, friction)
        label = split.label
        system = pdata.system_name(n_links)

        by_hold = pick(rows, label, "holdout_rmse", True)
        by_train = pick(rows, label, "trained_rmse", True)

        # Capacity curve: 1st-order rows only, so rule count is the sole variable.
        pts = [
            (r["n_rules"] // n_links, r["holdout_r2"])
            for r in rows
            if r["dataset"] == label
            and r["config"].endswith("_1st_g0_uniform_raw_probability")
        ]
        if pts:
            capacity[label] = ([p[0] for p in pts], [p[1] for p in pts])

        for tag, r in (("best_holdout", by_hold), ("best_trained", by_train)):
            if r is None:
                continue
            selection_rows.append({"dataset": label, "selected_by": tag, **r})

        # Two winners per dataset, not one -- see the module docstring in the
        # original run_all.py history: the trained-IC and held-out-IC optima sit
        # at opposite ends of the capacity range on the frictionless problems.
        cfg_h = config_from_key(by_hold["config"])
        res_h, model_h = run(split, cfg_h)
        assert (
            abs(res_h.holdout_ic["rmse"] - by_hold["holdout_rmse"]) < 1e-9
        ), "refit did not reproduce the swept holdout score"
        if by_train["config"] == by_hold["config"]:
            cfg_t, res_t, model_t = cfg_h, res_h, model_h
        else:
            cfg_t = config_from_key(by_train["config"])
            res_t, model_t = run(split, cfg_t)
            assert (
                abs(res_t.trained_ic["rmse"] - by_train["trained_rmse"]) < 1e-9
            ), "refit did not reproduce the swept trained-IC score"

        baselines = {
            "bracket midpoint (no learning)": baseline_bracket_midpoint(split),
            "nearest trained IC (no learning)": baseline_nearest(split),
        }

        holdout_preds[label] = draw_dataset(
            split,
            system,
            friction,
            baselines,
            trained=(model_t, cfg_t, res_t),
            holdout=(model_h, cfg_h, res_h),
        )

        fis_holdout_rmse[system] = res_h.holdout_ic["rmse"]
        datasets_out[label] = {
            "system": system,
            "friction": friction,
            "holdout_winner": {"config": cfg_h.key(), "metrics": res_h.holdout_ic},
            "trained_winner": {"config": cfg_t.key(), "metrics": res_t.trained_ic},
            "baselines": baselines,
            "capacity_curve": (
                {
                    "rules_per_output": capacity[label][0],
                    "holdout_r2": capacity[label][1],
                }
                if label in capacity
                else None
            ),
            "extrapolation": {
                "config": cfg_h.key(),
                "train_t_end_s": split.train_t_end,
                "test_t_end_s": round(
                    float(split.holdout_t[-1] + (split.t[1] - split.t[0])), 3
                ),
                "in_window_rmse": res_h.holdout_ic["rmse"],
                "in_window_r2": res_h.holdout_ic["r2"],
                "in_window_rmse_deg": res_h.holdout_ic["rmse_deg"],
                "extrap_rmse": res_h.extrap_ic["rmse"],
                "extrap_r2": res_h.extrap_ic["r2"],
                "extrap_rmse_deg": res_h.extrap_ic["rmse_deg"],
                "t_break_s": res_h.t_break,
            },
        }

        print(
            f"{label}:\n"
            f"    holdout-IC winner {cfg_h.key():48s} "
            f"RMSE={res_h.holdout_ic['rmse']:.4e} R2={res_h.holdout_ic['r2']:+.4f} "
            f"({res_h.holdout_ic['rmse_deg']:.2f} deg)\n"
            f"    trained-IC winner {cfg_t.key():48s} "
            f"RMSE={res_t.trained_ic['rmse']:.4e} R2={res_t.trained_ic['r2']:+.4f} "
            f"({res_t.trained_ic['rmse_deg']:.2f} deg)"
        )

    plots.error_vs_time(holdout_preds, t_end=pdata.T_END)
    plots.rmse_heatmap(
        fis_holdout_rmse,
        setting="holdout",
        friction=True,
        systems=[pdata.system_name(n) for n in pdata.N_LINKS],
    )
    plots.capacity_curve(capacity, best_paper=pr.best("double", True, "holdout")[2])

    print("\nPast the training window (held-out IC, holdout-winning config):")
    for label, d in datasets_out.items():
        e = d["extrapolation"]
        print(
            f"  {label:24s} 0-{e['train_t_end_s']:.0f}s R2={e['in_window_r2']:+.4f}"
            f"  {e['train_t_end_s']:.0f}-{e['test_t_end_s']:.0f}s R2={e['extrap_r2']:+.3e}"
            f"  RMSE={e['extrap_rmse']:.4g}  breaks at {e['t_break_s']:.2f}s"
        )

    return {"datasets": datasets_out, "selection": selection_rows}


def _write_comparison_md(payload):
    """results/comparison.md: FIS and no-learning baselines against the paper.

    Regenerated from `payload` on every run regardless of caching -- it is pure
    formatting over already-computed numbers, effectively free.
    """
    lines = [
        "# FIS vs the models reported in arXiv:2504.13453",
        "",
        "RMSE is in the paper's scaled units (per-trajectory min-max to [0, 1]), so it",
        "is a fraction of each trajectory's own angular range, not degrees. FIS rows",
        "also carry the degree figure.",
        "",
        "Two no-learning baselines appear in the holdout tables. They have no",
        "parameters and fit nothing: the holdout IC 2.05 deg sits exactly between the",
        "trained ICs 2.0 and 2.1 deg, so `bracket midpoint` averages those two scaled",
        "trajectories and `nearest trained IC` copies one of them. Any learned model",
        "that does not beat them has demonstrated grid interpolation, not dynamics.",
        "",
        "Each cell is scored with the FIS configuration that wins *that* cell; the",
        "configuration is named under the table. The trained-IC and held-out-IC optima",
        "are at opposite ends of the rule-count range, so no single configuration is",
        "best in both.",
        "",
    ]
    for label, d in payload["datasets"].items():
        system, friction = d["system"], d["friction"]
        for setting in ("trained", "holdout"):
            winner = d[f"{setting}_winner"]
            metrics, cfg_key = winner["metrics"], winner["config"]
            cell = {
                k: v for k, v in pr.RESULTS[(system, friction, setting)].items() if v
            }
            fric = "friction" if friction else "frictionless"
            rows = [(f"{k} (paper)", v) for k, v in cell.items()]
            rows.append(("**FIS (ours)**", (metrics["rmse"], metrics["r2"])))
            if setting == "holdout":
                for bname, bm in d["baselines"].items():
                    rows.append((f"_{bname}_", (bm["rmse"], bm["r2"])))
            merged = sorted(rows, key=lambda kv: kv[1][0])

            lines += [
                f"## {system} pendulum, {fric}, {setting} IC",
                "",
                "| Rank | Model | RMSE | R^2 |",
                "|---|---|---|---|",
            ]
            for i, (name, (rm, r2)) in enumerate(merged, 1):
                lines.append(f"| {i} | {name} | {rm:.4g} | {r2:.6f} |")
            rank = [n for n, _ in merged].index("**FIS (ours)**") + 1
            note = f"FIS configuration: `{cfg_key}`. "
            if not cell:
                note = "The paper ran no model in this cell. " + note
            # "Rank 1 of 1" is noise: with no paper models and no baselines there
            # is nothing to rank against, and printing a rank implies there was.
            if len(merged) > 1:
                note += f"Rank by RMSE: **{rank} of {len(merged)}**. "
            note += f"RMSE in degrees: {metrics['rmse_deg']:.2f}."
            lines += ["", note, ""]
    (RESULT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def stage_select(sweep_payload, lowcap_payload, sweep_hash, lowcap_hash, fresh):
    hash_of = {"sweep_hash": sweep_hash, "lowcap_hash": lowcap_hash}
    h = pipeline_cache.stage_hash(hash_of)
    if "select" not in fresh:
        cached = pipeline_cache.load_if_fresh(SELECT_PATH, h)
        if cached is not None:
            print("[select] cached")
            _write_comparison_md(cached)
            return cached, h

    print("[select] refitting each dataset's winner and drawing Table 1 figures ...")
    payload = _compute_select(sweep_payload, lowcap_payload)
    pipeline_cache.write_stage(SELECT_PATH, "select", h, hash_of, payload)
    _write_comparison_md(payload)
    print(f"[select] wrote {SELECT_PATH.name} and comparison.md")
    return payload, h


# ---------------------------------------------------------------------------
# Stage 5: bracket -- decorrelation diagnostic (Tables 2-3)
# ---------------------------------------------------------------------------
def stage_bracket(data_hash, fresh):
    hash_of = {
        "data_hash": data_hash,
        "lower_deg": bracket_diagnostic.LOWER_DEG,
        "upper_deg": bracket_diagnostic.UPPER_DEG,
    }
    h = pipeline_cache.stage_hash(hash_of)
    if "bracket" not in fresh:
        cached = pipeline_cache.load_if_fresh(BRACKET_PATH, h)
        if cached is not None:
            print("[bracket] cached")
            return cached, h

    print("[bracket] measuring bracket decorrelation ...")
    rows, curves = bracket_diagnostic.measure_all(log=lambda m: print(f"  {m}"))
    bracket_diagnostic.draw_fig_bracket(curves)
    bracket_diagnostic.draw_bracket_separation()
    payload = {"rows": rows}
    pipeline_cache.write_stage(BRACKET_PATH, "bracket", h, hash_of, payload)
    print(
        f"[bracket] wrote {BRACKET_PATH.name}, fig_bracket.png, trajectory_snapshots.png"
    )
    return payload, h


# ---------------------------------------------------------------------------
# Stage 6: capacity -- capacity/extrapolation comparison (Table 4)
# ---------------------------------------------------------------------------
def stage_capacity(data_hash, fresh):
    hash_of = {
        "data_hash": data_hash,
        "configs": [cfg.key() for _, cfg in compare_families.WITH_TIME],
    }
    h = pipeline_cache.stage_hash(hash_of)
    if "capacity" not in fresh:
        cached = pipeline_cache.load_if_fresh(CAPACITY_PATH, h)
        if cached is not None:
            print("[capacity] cached")
            return cached, h

    print("[capacity] fitting Table 4's capacity/extrapolation comparison ...")
    rows = gen_n2_rollout_comparison.main()
    payload = {"rows": rows}
    pipeline_cache.write_stage(CAPACITY_PATH, "capacity", h, hash_of, payload)
    print(f"[capacity] wrote {CAPACITY_PATH.name}, n2_rollout_comparison_all.png")
    return payload, h


# ---------------------------------------------------------------------------
# Stage 7: representation -- wrap / hysteresis / sin-cos study (Tables 6-8)
# ---------------------------------------------------------------------------
def stage_representation(data_hash, fresh):
    hash_of = {
        "data_hash": data_hash,
        "wrap_limits": wrap_sweep.WRAP_LIMITS,
        "sincos_buckets": [40, 120, 300],
    }
    h = pipeline_cache.stage_hash(hash_of)
    if "representation" not in fresh:
        cached = pipeline_cache.load_if_fresh(REPRESENTATION_PATH, h)
        if cached is not None:
            print("[representation] cached")
            return cached, h

    print("[representation] scoring wrap/representation schemes (Tables 6-8) ...")
    wrap_rows = wrap_sweep.build_wrap_sweep_rows(log=lambda m: print(f"  {m}"))
    repr_rows = wrap_sweep.build_representation_rows()
    sincos_rows = wrap_sweep.main_sincos_capacity(log=lambda m: print(f"  {m}"))
    payload = {
        "wrap_sweep": wrap_rows,
        "representations": repr_rows,
        "sincos_capacity": sincos_rows,
    }
    # gen_rollout_error_vs_n.py reads results/representation.json off disk, so the
    # file has to exist before it runs.
    pipeline_cache.write_stage(
        REPRESENTATION_PATH, "representation", h, hash_of, payload
    )
    gen_rollout_error_vs_n.main()
    print(
        f"[representation] wrote {REPRESENTATION_PATH.name}, "
        f"rollout_error_vs_n.png, rollout_error_vs_n_lines.png"
    )
    return payload, h


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _require(path, stage_name):
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist -- run `python run_all.py --stage {stage_name}` "
            f"(or a full `python run_all.py`) first"
        )
    return pipeline_cache.load_payload(path)


def _require_hash(path, stage_name):
    import json as _json

    _require(path, stage_name)
    return _json.loads(path.read_text(encoding="utf-8"))["hash"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fresh",
        nargs="*",
        default=None,
        metavar="STAGE",
        help="force recompute. Bare --fresh forces every stage; "
        "--fresh sweep lowcap forces only those (downstream stages still "
        "recompute only if what they depend on actually changed).",
    )
    ap.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="run only this stage; its upstream JSON must already exist",
    )
    args = ap.parse_args()

    if args.fresh is None:
        fresh = set()
    elif len(args.fresh) == 0:
        fresh = set(STAGE_ORDER)
    else:
        fresh = set(args.fresh)

    if args.stage:
        _run_one(args.stage, fresh)
    else:
        _run_all(fresh)


def _run_one(stage, fresh):
    """Run a single stage, loading (never recomputing) whatever it depends on."""
    if stage == "data":
        stage_data(fresh)
        return

    data_hash = _require_hash(DATA_PATH, "data")
    if stage == "sweep":
        stage_sweep(data_hash, fresh)
    elif stage == "lowcap":
        stage_lowcap(data_hash, fresh)
    elif stage == "select":
        sweep_payload = _require(SWEEP_PATH, "sweep")
        lowcap_payload = _require(LOWCAP_PATH, "lowcap")
        sweep_hash = _require_hash(SWEEP_PATH, "sweep")
        lowcap_hash = _require_hash(LOWCAP_PATH, "lowcap")
        stage_select(sweep_payload, lowcap_payload, sweep_hash, lowcap_hash, fresh)
    elif stage == "bracket":
        stage_bracket(data_hash, fresh)
    elif stage == "capacity":
        stage_capacity(data_hash, fresh)
    elif stage == "representation":
        stage_representation(data_hash, fresh)


def _run_all(fresh):
    _, data_hash = stage_data(fresh)
    sweep_payload, sweep_hash = stage_sweep(data_hash, fresh)
    lowcap_payload, lowcap_hash = stage_lowcap(data_hash, fresh)
    stage_select(sweep_payload, lowcap_payload, sweep_hash, lowcap_hash, fresh)
    stage_bracket(data_hash, fresh)
    stage_capacity(data_hash, fresh)
    stage_representation(data_hash, fresh)
    print(
        f"\nDone. results/*.json + comparison.md in {RESULT_DIR}, figures in {plots.FIG_DIR}"
    )


if __name__ == "__main__":
    main()
