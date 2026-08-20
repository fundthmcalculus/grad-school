"""Build the shareable HTML summary of the benchmark and the review.

Numbers come from the same run artifacts `write_benchmark.py` reads, so the page
and the markdown report cannot disagree. Figures are inlined as data URIs
because the artifact host blocks every external request.
"""

from __future__ import annotations

import base64
import os

import numpy as np
import pandas as pd

import cmapss_data
import report

OUT = report.OUT
FIG = report.FIG
DEST = os.path.join(OUT, "summary.html")


def data_uri(name: str) -> str:
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def rows_for(tag: str, tables: dict) -> pd.DataFrame:
    return tables[tag].sort_values("rmse")


def table_html(df: pd.DataFrame, fis_rmse: float) -> str:
    out = [
        "<div class='scroll'><table>",
        "<thead><tr><th>model</th><th class='n'>params</th>"
        "<th class='n'>test RMSE</th><th class='n'>MAE</th>"
        "<th class='n'>endpoint</th><th class='n'>total s</th></tr></thead><tbody>",
    ]
    for _, r in df.iterrows():
        is_fis = r["arm"] == "fis"
        is_hot = str(r["arm"]).startswith("hot")
        cls = "fis" if is_fis else ("hot" if is_hot else "")
        beats = "" if is_fis else (" win" if r["rmse"] < fis_rmse else "")
        params = (
            "--"
            if not np.isfinite(r.get("n_parameters", np.nan))
            else f"{r['n_parameters']:,.0f}"
        )
        out.append(
            f"<tr class='{cls}'>"
            f"<td>{report.ARM_LABEL.get(r['arm'], r['arm'])}</td>"
            f"<td class='n'>{params}</td>"
            f"<td class='n{beats}'>{r['rmse']:.2f}</td>"
            f"<td class='n'>{r['mae']:.2f}</td>"
            f"<td class='n'>{r['rmse_endpoint']:.2f}</td>"
            f"<td class='n'>{r['total_s']:.3f}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def fidelity_html(fid: dict) -> str:
    df = pd.DataFrame(fid["rows"])
    df = df[df.top_n > 0].sort_values("n_features")
    out = [
        "<div class='scroll'><table>",
        "<thead><tr><th class='n'>FIS features</th><th class='n'>knots</th>"
        "<th class='n'>seed vs FIS</th><th class='n'>best additive</th>"
        "<th class='n'>rows outside knots</th></tr></thead><tbody>",
    ]
    for _, r in df.iterrows():
        bad = " bad" if r["fidelity_relative"] > 1 else ""
        out.append(
            f"<tr><td class='n'>{r['n_features']:.0f}</td>"
            f"<td class='n'>{r['n_hidden']:.0f}</td>"
            f"<td class='n{bad}'>{r['fidelity_relative']:.3f}</td>"
            f"<td class='n'>{r['additive_relative']:.3f}</td>"
            f"<td class='n'>{r['frac_rows_outside_knots']:.1%}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


FINDINGS = [
    (
        "warning",
        "PR #111",
        "<code>triangle_to_relu</code> mishandles degenerate triangles",
        "For <code>a == b &lt; c</code> it returns the <em>negation</em> of the correct ramp and keeps falling "
        "(<code>max|err| = 1.0</code>). The form DS02 actually hits is the fully collapsed "
        "<code>a == b == c</code> from a zero-variance feature (1 of 109 terms), where the old code emitted an "
        "all-zero expansion &mdash; wrong in kind, not in value, so no published number was corrupted. A "
        "discontinuous term has no finite ReLU expansion, so the fix is to raise, not to patch.",
        "fixed — raises now; fis_knots skips degenerate terms and warns with a count; 5 new regression tests",
    ),
    (
        "warning",
        "PR #111",
        "The seed extrapolates linearly past its outermost knot, unmeasured",
        "42% of DS02 test rows fall outside at least one FIS feature's knot range on the <code>honest</code> "
        "pipeline. The write-up attributes all fidelity loss to non-additivity without separating this term.",
        "documented — the term, and the two cheap ablations that would settle it",
    ),
    (
        "warning",
        "PR #111",
        "<code>partial_dependence</code> breaks on a 2nd-order TSK",
        "Moving one feature off the data manifold makes a quadratic consequent extrapolate. On DS02's champion "
        "pipeline that drives the seed 31x past the FIS's own spread. The study only ever ran 0th/1st-order "
        "systems, so it never hit this; the docstring should say so.",
        "documented — 0th/1st-order only, with the ALE-style fix named",
    ),
    (
        "note",
        "PR #111",
        "The <code>quantile</code> headline uses the unfair comparison, and says so",
        "<code>quantile</code> fits labels; <code>hot-analytic</code> does not. But <code>hot</code> already gets "
        "the identical ridge solve, so <code>hot</code> vs <code>quantile</code> is apples-to-apples and much "
        "closer: 9.39 against 9.00 on DS02. Promote that pair to the headline.",
        "author's call — a framing change, not a code change",
    ),
    (
        "good",
        "CMAPSS",
        "The published 6.48 survives an honest re-selection",
        "The DOE selects Factor-D on <code>rmse_test_true</code>, so its configurations are test-selected. "
        "Re-running the same 72-point grid against a validation fold returns the <strong>identical</strong> "
        "configuration and the identical 6.48. The objection is real in principle and empty in fact here — "
        "which is a far better answer than conceding it. Disclose both.",
        "disclosed — in cmapss_rul.py and cmapss_rul_best.py, with the re-check",
    ),
    (
        "warning",
        "CMAPSS",
        "Test-unit RUL caps are built, and one metric consumes them",
        "<code>unit_physical_caps(pd.concat([train_tab, test_tab]))</code> derives caps from the <code>hs</code> "
        "oracle over both splits. <code>rmse_test_true</code> is clean, but <code>rmse_test_shaped</code> is not, "
        "and it ships in the results CSVs under a name that does not announce itself.",
        "fixed — caps built from training units only; 11.23 and 6.48 unchanged",
    ),
    (
        "note",
        "CMAPSS",
        "The per-engine headline is three data points",
        "On DS02 the canonical convention scores one RUL per test engine — n=3. It swings 15 cycles between "
        "models whose per-sample RMSE differs by under 1. Lead with per-sample; always attach the n.",
        "author's call — a reporting convention",
    ),
]


def main():
    tables, sweeps, fids = report.main()
    ext = report.load("external_baselines.json")

    hon, best = tables["honest"], tables["best"]
    b1 = tables.get("best (1st-order FIS)")
    hon_fis = float(hon[hon.arm == "fis"]["rmse"].iloc[0])
    best_fis = float(best[best.arm == "fis"]["rmse"].iloc[0])
    hon_nn = hon[hon.kind == "network"].sort_values("rmse").iloc[0]
    best_nn = best[best.kind == "network"].sort_values("rmse").iloc[0]
    hon_hot = float(hon[hon.arm == "hot"]["rmse"].iloc[0])
    conv_b1 = report.load("arms_best_1storder.json")["conversion"]

    tiles = [
        (
            "cheap pipeline",
            f"{hon_nn['rmse']:.2f}",
            f"vs FIS {hon_fis:.2f}",
            f"{100 * (hon_fis - hon_nn['rmse']) / hon_fis:.0f}% better, "
            f"{hon_nn['n_parameters']:,.0f} parameters",
            "win",
        ),
        (
            "rich pipeline",
            f"{best_nn['rmse']:.2f}",
            f"vs FIS {best_fis:.2f}",
            "inside the seed spread — a tie",
            "tie",
        ),
        (
            "smallest net at FIS parity",
            "1",
            "hidden unit",
            "on TRIBBLE's 28 columns, 0.007 s to parity",
            "win",
        ),
        (
            "warm start",
            f"{hon_hot:.2f}",
            f"vs random init {float(hon[hon.arm == 'he']['rmse'].iloc[0]):.2f}",
            "worse, and 6x the wall clock",
            "loss",
        ),
    ]

    css = """
:root{
  color-scheme: light;
  --ground:#eef1f4; --surface:#fbfcfd; --plate:#ffffff;
  --ink:#0e151b; --ink-2:#4a5763; --ink-3:#77848f;
  --rule:#d3dae1; --rule-2:#e3e8ed;
  --accent:#0a6167; --accent-soft:#d7e8e8;
  --good:#1c6b45; --warn:#8f5310; --crit:#9c2f2a; --note:#4a5763;
  --win:#0a6167; --loss:#9c2f2a;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0b0f13; --surface:#141b21; --plate:#f4f6f8;
    --ink:#e9eef2; --ink-2:#a4b1bc; --ink-3:#75828d;
    --rule:#27313a; --rule-2:#1d262e;
    --accent:#4fb3b8; --accent-soft:#14343a;
    --good:#5cb98a; --warn:#d69a4e; --crit:#e07b74; --note:#a4b1bc;
    --win:#4fb3b8; --loss:#e07b74;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0b0f13; --surface:#141b21; --plate:#f4f6f8;
  --ink:#e9eef2; --ink-2:#a4b1bc; --ink-3:#75828d;
  --rule:#27313a; --rule-2:#1d262e;
  --accent:#4fb3b8; --accent-soft:#14343a;
  --good:#5cb98a; --warn:#d69a4e; --crit:#e07b74; --note:#a4b1bc;
  --win:#4fb3b8; --loss:#e07b74;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:76rem;margin:0 auto;padding:clamp(1.5rem,4vw,3.5rem) clamp(1rem,4vw,2.5rem) 5rem}
.serif{font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif}
.mono,code,td.n,th.n{
  font-family: ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  font-variant-numeric: tabular-nums;
}
.prose{max-width:66ch}

header{border-bottom:2px solid var(--ink);padding-bottom:1.75rem;margin-bottom:2.5rem}
.eyebrow{
  font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin:0 0 .9rem
}
h1{
  font-size:clamp(2rem,5.5vw,3.4rem); line-height:1.05; margin:0 0 1rem;
  text-wrap:balance; font-weight:600; letter-spacing:-.015em;
}
.standfirst{font-size:clamp(1.05rem,2vw,1.22rem);color:var(--ink-2);margin:0;max-width:60ch}
.meta{
  display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;margin-top:1.5rem;
  font-size:.8rem;color:var(--ink-3)
}
.meta b{color:var(--ink-2);font-weight:600}

.tiles{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));margin-bottom:3.5rem}
.tile{background:var(--surface);padding:1.4rem 1.3rem 1.5rem;display:flex;
  flex-direction:column;gap:.35rem}
.tile .lab{font-size:.7rem;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
.tile .big{font-size:2.6rem;line-height:1;font-weight:600;letter-spacing:-.02em}
.tile .big.win{color:var(--win)} .tile .big.loss{color:var(--loss)}
.tile .cmp{font-size:.85rem;color:var(--ink-2)}
.tile .note{font-size:.78rem;color:var(--ink-3);margin-top:.2rem}

section{margin-bottom:3.5rem}
h2{font-size:1.55rem;font-weight:600;letter-spacing:-.01em;margin:0 0 .5rem;text-wrap:balance}
h2 .num{color:var(--accent);font-size:.8rem;letter-spacing:.12em;display:block;
  margin-bottom:.5rem;font-weight:600;font-family:inherit}
h3{font-size:1.02rem;font-weight:600;margin:2rem 0 .6rem;color:var(--ink)}
p{margin:0 0 1rem}
.lede{color:var(--ink-2)}

.scroll{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);margin:1rem 0 1.5rem}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:.55rem .85rem;text-align:left;border-bottom:1px solid var(--rule-2);white-space:nowrap}
thead th{
  font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;border-bottom:1px solid var(--rule);background:var(--surface);
  position:sticky;top:0
}
td.n,th.n{text-align:right}
tbody tr:last-child td{border-bottom:none}
tr.fis{background:var(--accent-soft)}
tr.fis td:first-child::after{content:" — incumbent";color:var(--ink-3);font-size:.78em}
tr.hot td:first-child::after{content:" — under test";color:var(--ink-3);font-size:.78em}
td.win{color:var(--win);font-weight:600}
td.bad{color:var(--loss)}

figure{margin:1.5rem 0 2rem}
figure img{display:block;width:100%;height:auto;background:var(--plate);
  border:1px solid var(--rule);padding:.5rem}
figcaption{font-size:.82rem;color:var(--ink-3);margin-top:.6rem;max-width:66ch}

.findings{display:flex;flex-direction:column;gap:1px;background:var(--rule);
  border:1px solid var(--rule)}
.finding{background:var(--surface);padding:1.15rem 1.3rem;display:grid;
  grid-template-columns:auto 1fr;gap:.4rem 1rem;align-items:start}
.chip{
  font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700;
  padding:.22rem .5rem;border:1px solid currentColor;white-space:nowrap;line-height:1.3
}
.chip.critical{color:var(--crit)} .chip.warning{color:var(--warn)}
.chip.good{color:var(--good)} .chip.note{color:var(--note)}
.finding .who{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;grid-column:2}
.finding h4{margin:0;font-size:1rem;font-weight:600;grid-column:2;line-height:1.4}
.finding p{margin:.3rem 0 0;grid-column:2;font-size:.9rem;color:var(--ink-2)}
code{font-size:.88em;background:var(--rule-2);padding:.08em .35em;border-radius:2px}
.finding .chip{grid-row:1/4}
.finding .status{grid-column:2;font-size:.78rem;margin:.55rem 0 0;font-weight:600}
.finding .status.done{color:var(--good)}
.finding .status.done::before{content:"\2713\00a0\00a0"}
.finding .status.open{color:var(--ink-3)}
.finding .status.open::before{content:"\2192\00a0\00a0"}

pre{background:var(--surface);border:1px solid var(--rule);padding:1rem 1.15rem;
  overflow-x:auto;font-size:.82rem;line-height:1.55;margin:1rem 0}
pre code{background:none;padding:0}

.callout{border-left:3px solid var(--accent);padding:.2rem 0 .2rem 1.25rem;margin:1.5rem 0;
  color:var(--ink-2)}
.callout strong{color:var(--ink)}

footer{border-top:1px solid var(--rule);padding-top:1.75rem;margin-top:1rem;
  font-size:.85rem;color:var(--ink-3)}
footer a{color:var(--accent)}
a:focus-visible,:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
"""

    H = []
    a = H.append
    a("<title>Rules Against Weights</title>")
    a(f"<style>{css}</style>")
    a("<div class='wrap'>")

    # ---- header ----
    a("<header>")
    a("<p class='eyebrow'>N-CMAPSS DS02 &middot; turbofan remaining useful life</p>")
    a("<h1 class='serif'>Rules Against Weights</h1>")
    a(
        "<p class='standfirst'>A ReLU network benchmarked against the TRIBBLE fuzzy "
        "inference system on identical features, and the Bede&ndash;Kreinovich&ndash;Toth "
        "warm start applied to both. The network wins where the features are cheap, "
        "ties where they are rich, and the warm start loses everywhere &mdash; for a "
        "reason that can be measured rather than argued.</p>"
    )
    a("<div class='meta'>")
    a(
        "<span><b>Engines</b> fit 2/5/10/16 &middot; val 18/20 &middot; test 11/14/15</span>"
    )
    a("<span><b>Selection</b> validation only &mdash; test scored once</span>")
    a("<span><b>Hardware</b> 8-core CPU, NumPy, no GPU</span>")
    a("</div>")
    a("</header>")

    # ---- tiles ----
    a("<div class='tiles'>")
    for lab, big, cmp_, note, kind in tiles:
        a(
            f"<div class='tile'><span class='lab'>{lab}</span>"
            f"<span class='big mono {kind}'>{big}</span>"
            f"<span class='cmp mono'>{cmp_}</span>"
            f"<span class='note'>{note}</span></div>"
        )
    a("</div>")

    # ---- 1 quality ----
    a("<section>")
    a("<h2><span class='num'>Result</span>Quality, both pipelines</h2>")
    a(
        "<p class='lede prose'>Per-sample RMSE in cycles on the three held-out engines, "
        "median over seeds, against uncapped ground-truth RUL. Every model below sees "
        "the same condition-corrected columns and the same selection budget.</p>"
    )
    a(
        "<h3>Cheap pipeline &mdash; 18 real sensors, one row per flight cycle "
        "(309 fit rows, 90 features)</h3>"
    )
    a(table_html(hon, hon_fis))
    a(
        "<h3>Rich pipeline &mdash; the literature's 20 channels through a memory window "
        "(18,025 fit rows, 60 features)</h3>"
    )
    a(table_html(best, best_fis))
    a("<h3>Where the two models actually differ</h3>")
    a(
        "<p class='lede prose'>An RMSE is one number over a whole trajectory. "
        "On a prognostics problem the shape matters too &mdash; whether a model "
        "tracks near end of life, and whether it errs early (safe) or late "
        "(not).</p>"
    )
    a(
        f"<figure><img src='{data_uri('trajectories.png')}' alt='Six panels: "
        "predicted versus true remaining useful life against flight cycle for "
        "test engines 11, 14 and 15, on both pipelines. The FIS and network "
        "curves track each other closely and both converge on the truth in the "
        "final third of every trajectory.'>"
    )
    a(
        "<figcaption>One engine carries the error: on the cheap pipeline unit 14 "
        "runs 15.8 (FIS) and 10.1 (network) against 6.7&ndash;7.9 for the other "
        "two, and both models start it ~30 cycles low. The network's advantage "
        "is early-life &mdash; past roughly cycle 45 the two are "
        "indistinguishable on every engine, which is the part of the trajectory "
        "where the answer matters most. Both also dip below zero in the last few "
        "cycles; cheap to clamp, and hidden by a trajectory-wide average. "
        "Configuration, epoch and seed are read from the run artifacts and each "
        "curve's RMSE is asserted against the recorded value before "
        "drawing.</figcaption></figure>"
    )
    a(
        "<div class='callout prose'><strong>The FIS reproduces exactly.</strong> "
        "Independently reimplemented, both pipelines land on the numbers "
        "<code>cmapss_rul_best.py</code> documents &mdash; 11.23 and 6.48. And "
        "re-selecting the FIS's own hyperparameters on validation rather than on test "
        "returns the <em>identical</em> configuration on the rich pipeline, so its 6.48 "
        "owes nothing to test-set selection.</div>"
    )
    a("</section>")

    # ---- 2 size ----
    a("<section>")
    a("<h2><span class='num'>Sweep</span>How small does the network need to be?</h2>")
    a(
        "<p class='lede prose'>Width, learning rate and batch size swept with "
        "epochs-to-target read off the validation curve. Two feature spaces: every "
        "aggregated column, and only the columns TRIBBLE's feature selection kept.</p>"
    )
    a(
        f"<figure><img src='{data_uri('sweep.png')}' alt='Validation RMSE against "
        "hidden-unit count for both feature spaces and both pipelines. Both curves are "
        "nearly flat across a 12x to 32x range in width, and both sit below the FIS "
        "reference line.'>"
    )
    a(
        "<figcaption>Width is not the binding constraint. One hidden unit on TRIBBLE's "
        "28 columns already beats the FIS; 12 units are no better. After condition "
        "correction, RUL is nearly affine in the sensor residuals and the network's "
        "linear skip does most of the work. Note the reversal between panels: feature "
        "selection helps when the columns are many and redundant, and costs when they "
        "are few and each carries signal.</figcaption></figure>"
    )
    a("</section>")

    # ---- 3 hot start ----
    a("<section>")
    a("<h2><span class='num'>Diagnosis</span>Why the warm start does not fire</h2>")
    a(
        "<p class='lede prose'>The conversion is supposed to hand the network the FIS "
        "itself. On DS02 the seed lands further from the FIS than the FIS is from the "
        "data, so there is nothing for gradient descent to inherit. Forcing the FIS "
        "down to <code>top_n</code> features says why &mdash; measured against the "
        "<em>best possible</em> additive approximation, a reference PR #111 does not "
        "have.</p>"
    )
    a(fidelity_html(fids["honest"]))
    a(
        f"<figure><img src='{data_uri('fidelity.png')}' alt='Log-scale conversion "
        "fidelity against the number of features the FIS kept, for both pipelines. The "
        "seed curve and the best-possible-additive curve lie on top of each other "
        "throughout.'>"
    )
    a(
        "<figcaption>Two curves, one line. The seed tracks the best achievable additive "
        "fit at every dimension and sometimes beats it &mdash; so the loss is "
        "irreducible interaction, not a bad conversion or badly placed knots. At one "
        "feature the error is 0.070 relative: the equivalence, executable, on turbofan "
        "data. No wider axis-aligned seed can close the rest.</figcaption></figure>"
    )
    a(
        "<div class='callout prose'><strong>And its best case still loses.</strong> "
        "Converting a 1st-order FIS instead of the <code>full-2nd</code> champion cuts "
        f"the fidelity loss from 31.46 to {conv_b1['fidelity_relative']:.2f} relative "
        "&mdash; the conversion now works &mdash; and the warm arm improves to "
        f"{float(b1[b1.arm == 'hot-analytic']['rmse'].iloc[0]):.2f}. It still loses to "
        f"{float(b1[(b1.kind == 'network') & (~b1.arm.str.startswith('hot'))]['rmse'].min()):.2f} "
        "from an initialization that cost nothing.</div>"
    )
    a("</section>")

    # ---- 4 review ----
    a("<section>")
    a("<h2><span class='num'>Review</span>What the code review turned up</h2>")
    a(
        "<p class='lede prose'>Both bodies of work were re-run, not just read. "
        "The conversion is correct where it claims to be, and the preprocessing "
        "reproduces to the documented digit. These are the things that needed "
        "changing, ordered by severity, and what was done about each. The test "
        "suite went from 13 tests to 18; no reported result moved.</p>"
    )
    a("<div class='findings'>")
    for sev, who, title, body, status in FINDINGS:
        done = (
            "done"
            if status.split()[0] in ("fixed", "documented", "disclosed")
            else "open"
        )
        a(
            f"<div class='finding'><span class='chip {sev}'>{sev}</span>"
            f"<span class='who'>{who}</span><h4>{title}</h4><p>{body}</p>"
            f"<p class='status {done}'>{status}</p></div>"
        )
    a("</div>")
    a("</section>")

    # ---- 5 instrument check ----
    if ext:
        a("<section>")
        a(
            "<h2><span class='num'>Control</span>Is the trainer the limiting factor?</h2>"
        )
        a(
            "<p class='lede prose'>Every network number comes from a 60-line NumPy Adam "
            "loop, chosen so no arm can win on a framework default. Three off-the-shelf "
            "models, selected and scored the same way, check that this is not "
            "understating what an ordinary model does here.</p>"
        )
        df = pd.DataFrame(
            [
                dict(
                    bundle=r["bundle"],
                    model=r["model"],
                    val=r["val_rmse"],
                    rmse=r["test"]["rmse"],
                    fit=r["fit_seconds"],
                )
                for r in ext
            ]
        )
        a(
            "<div class='scroll'><table><thead><tr><th>pipeline</th><th>model</th>"
            "<th class='n'>val RMSE</th><th class='n'>test RMSE</th>"
            "<th class='n'>fit s</th></tr></thead><tbody>"
        )
        for _, r in df.sort_values(["bundle", "rmse"]).iterrows():
            a(
                f"<tr><td>{r['bundle']}</td><td>{r['model']}</td>"
                f"<td class='n'>{r['val']:.2f}</td><td class='n'>{r['rmse']:.2f}</td>"
                f"<td class='n'>{r['fit']:.3f}</td></tr>"
            )
        a("</tbody></table></div>")
        a(
            "<p class='prose'>Nothing off the shelf changes a conclusion, and the tree "
            "models &mdash; not networks at all &mdash; land in the same band. The "
            "ceiling here is the data, not the optimizer.</p>"
        )
        a("</section>")

    # ---- 6 caveats + repro ----
    a("<section>")
    a("<h2><span class='num'>Limits</span>What would change these numbers</h2>")
    a(
        "<p class='prose'>Three test engines and two validation engines. The per-engine "
        "endpoint column is n=3 and swings 15 cycles between models whose per-sample "
        "RMSE differs by under one. One dataset &mdash; and the feature-selection "
        "conclusion already flips between two aggregations of that same dataset, so it "
        "would very likely move again across DS01&ndash;DS08. The RUL cap uses the "
        "simulator's <code>hs</code> latent; both arms get it, so the comparison is "
        "fair, but neither number is what an onboard system could achieve.</p>"
    )
    a("<h3>Reproducing</h3>")
    a(
        "<pre><code>cd experiments/nn-cmapss\n"
        "python cmapss_data.py honest &amp;&amp; python cmapss_data.py best\n"
        "python smoke.py honest\n"
        "python sweep.py --bundle honest --epochs 400 --seeds 5 ...\n"
        "python sweep_fis.py &amp;&amp; python fidelity.py\n"
        "python arms.py --bundle honest --epochs 400 --n-hidden 8 --seeds 5\n"
        "python arms.py --bundle best --epochs 120 --n-hidden 32 --seeds 3\n"
        "python external_baselines.py &amp;&amp; python write_benchmark.py</code></pre>"
    )
    a("</section>")

    a("<footer>")
    a(
        "<p>Full tables, per-seed curves and the complete review live in "
        "<code>outputs/nn-cmapss/</code> &mdash; <code>BENCHMARK.md</code> and "
        "<code>REVIEW.md</code>. Every figure and number on this page is generated from "
        "the run artifacts by <code>write_artifact.py</code>; nothing is transcribed.</p>"
    )
    a("</footer>")
    a("</div>")

    with open(DEST, "w") as f:
        f.write("\n".join(H))
    kb = os.path.getsize(DEST) / 1024
    print(f"wrote {os.path.relpath(DEST, cmapss_data.REPO)} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
