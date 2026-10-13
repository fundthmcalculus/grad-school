"""Verdicts for the GPU track, registered in RESULTS.md (D1-D3, Q1).

    .venv/bin/python -m flm.device          # reads outputs/device-check and outputs/scaling-gpu

D1: |val BPC (cuda) - val BPC (cpu tune cell)| < 0.01 for each of the eight arms.
D2: the CUDA self-repeat difference is reported per arm (the noise floor D1 is read against).
D3: CUDA chars/s >= 3x the CPU tune cell's chars/s, per arm, at d = 32.
Q1: every scaling-gpu run is finite, not diverged, and complete (144 jobs).

Exits non-zero if any pass/fail criterion fails, so the verdict cannot be read off a
silent table. A missing or incomplete sweep is reported as such, not as a pass.
"""

from __future__ import annotations

import json
import math
import sys

import flm.grids  # noqa: F401  (registers the grids; best_tune / scaling-gpu live there)
from flm.grids import ARMS, best_tune
from flm.sweep import GRIDS, OUT

D1_TOL = 0.01
D3_MIN = 3.0


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def device_check():
    rows, ok = [], True
    for arm in ARMS:
        lr, sc = best_tune(arm)[1], best_tune(arm)[3]
        cpu = _load(OUT / "tune" / f"{arm}_lr{lr}_sc{sc}.json")
        gpu = _load(OUT / "device-check" / f"{arm}_cuda.json")
        rep = _load(OUT / "device-check" / f"{arm}_cuda_rep.json")
        ctl = _load(OUT / "device-check-cpu" / f"{arm}_cpu.json")
        if cpu is None or gpu is None or rep is None:
            rows.append((arm, "MISSING", "", "", "", "", "", "", ""))
            ok = False
            continue
        d1 = abs(gpu["val_bpc"] - cpu["val_bpc"])
        d2 = abs(gpu["val_bpc"] - rep["val_bpc"])
        d3 = gpu["chars_per_sec"] / cpu["chars_per_sec"]
        p1 = math.isfinite(d1) and d1 < D1_TOL
        p3 = d3 >= D3_MIN
        ok &= p1 and p3
        rows.append(
            (
                arm,
                f"{cpu['val_bpc']:.5f}",
                f"{gpu['val_bpc']:.5f}",
                f"{d1:.5f} {'PASS' if p1 else 'FAIL'}",
                f"{d2:.5f}",
                f"{d3:.1f}x {'PASS' if p3 else 'FAIL'}",
                f"{ctl['val_bpc']:.5f}" if ctl else "not run",
                f"{abs(gpu['val_bpc'] - ctl['val_bpc']):.5f}" if ctl else "",
                f"{gpu.get('gpu')}",
            )
        )
    print(
        "| arm | CPU tune BPC (archived, cpu wheel) | CUDA BPC | D1 abs diff | D2 repeat diff | D3 speed | CPU BPC, cu130 build | D1s same-build diff | GPU |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    return ok


def same_build_control():
    """Amendment control: CPU under the same CUDA torch build as the GPU track (D1s)."""
    for arm in ARMS:
        ctl = _load(OUT / "device-check-cpu" / f"{arm}_cpu.json")
        gpu = _load(OUT / "device-check" / f"{arm}_cuda.json")
        if ctl is None or gpu is None:
            print(f"D1s {arm}: MISSING")
            continue
        d = abs(gpu["val_bpc"] - ctl["val_bpc"])
        print(
            f"D1s {arm}: cpu(cu130) {ctl['val_bpc']:.5f} vs cuda {gpu['val_bpc']:.5f}"
            f" -> {d:.5f} {'PASS' if d < D1_TOL else 'FAIL'}"
        )


def qualification():
    runs = sorted((OUT / "scaling-gpu").glob("*.json"))
    bad = []
    for f in runs:
        r = json.loads(f.read_text())
        if not math.isfinite(r["val_bpc"]) or r["diverged_step"] is not None:
            bad.append(f.name)
    n_grid = len(GRIDS["scaling-gpu"])
    print(
        f"Q1 scaling-gpu: {len(runs)}/{n_grid} runs, {len(bad)} non-finite or diverged"
    )
    for b in bad:
        print("  bad:", b)
    return len(runs) == n_grid and not bad


def main():
    print("## D1-D3 device parity (seed 0, d=32, 10M chars)")
    ok_d = device_check()
    print()
    same_build_control()
    print()
    ok_q = qualification()
    print()
    print("verdict:", "PASS" if ok_d and ok_q else "FAIL")
    sys.exit(0 if ok_d and ok_q else 1)


if __name__ == "__main__":
    main()
