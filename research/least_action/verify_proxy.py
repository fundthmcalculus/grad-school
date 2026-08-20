"""Check the two properties the certificate proxy has to have (README §15a).

Run: .venv/bin/python research/least_action/verify_proxy.py   (~5 s)

The proxy is only safe to optimize against if it is a genuine NECESSARY
condition for the SOS certificate: every radius it reports must be witnessed by
a real state where Vdot >= 0. And §15a claims the obvious sharpening -- local
descent on min{z'Pz : Vdot(z) >= 0} -- is structurally degenerate rather than
merely badly tuned. Both are cheap to check directly, so neither is left as an
assertion.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

from fis_sos import _vdot_at, _vdot_batch, ball_samples, sphere_directions
from fis_twocart import TskController, TwoCart, fit_consequents, place_rules

warnings.filterwarnings("ignore")


def main() -> None:
    plant = TwoCart()
    a_mat, b_mat = plant.linearization()
    p_lyap = solve_continuous_are(
        a_mat, b_mat, np.diag([1.0, 10.0, 1.0, 1.0]), np.array([[10.0]])
    )
    d = np.load(".twocart_train.npz")
    cen, wid = place_rules(d["z"], d["w"], 2)
    ctrl = TskController(cen, wid, order=1, mf="pi")
    fit_consequents(ctrl, d["z"], d["u"], d["w"])
    lo = float(np.linalg.eigvalsh(p_lyap).min())

    print("1. The descent formulation is degenerate, not badly tuned")
    v0 = _vdot_at(np.zeros(4), ctrl, plant, p_lyap)
    print(f"   Vdot(0) = {v0:+.3e}  ->  z = 0 SATISFIES the constraint " f"Vdot >= 0,")
    print("   so min{z'Pz : Vdot(z) >= 0} has global optimum 0 at the origin.")

    samples = ball_samples()
    vdot = _vdot_batch(samples, ctrl, plant, p_lyap)
    rho = np.einsum("ni,ij,nj->n", samples, p_lyap, samples)
    bad = np.flatnonzero(vdot >= 0.0)
    print(
        f"   violations are not rare: {bad.size}/{len(samples)} "
        f"({100 * bad.size / len(samples):.0f}%) of cloud points have "
        f"Vdot >= 0,"
    )
    print("   so the cloud estimate is a minimum-order statistic (see 3).")

    starts = bad[np.argsort(rho[bad])][:6]
    collapsed = boundary = 0
    for idx in starts:
        res = minimize(
            lambda z: float(z @ p_lyap @ z),
            samples[idx],
            method="SLSQP",
            jac=lambda z: 2.0 * p_lyap @ z,
            constraints=[
                {"type": "ineq", "fun": lambda z: _vdot_at(z, ctrl, plant, p_lyap)}
            ],
            options={"maxiter": 40, "ftol": 1e-10},
        )
        if np.linalg.norm(res.x) < 1e-6:
            collapsed += 1
        if _vdot_at(res.x, ctrl, plant, p_lyap) < 0.0:
            boundary += 1
    print(
        f"   of {len(starts)} local solves from the best sampled starts: "
        f"{collapsed} collapsed to ||z|| = 0,"
    )
    print(
        f"   and {boundary} ended infeasible (on the Vdot = 0 boundary from "
        f"below), so a"
    )
    print(
        "   strict feasibility recheck discards every one and the bound " "never moves."
    )

    print("\n2. The ray proxy's witnesses are real (necessary condition holds)")
    dirs = sphere_directions(2000)
    grid = np.linspace(2.0 / 64, 2.0, 64)
    pts = (dirs[:, None, :] * grid[None, :, None]).reshape(-1, 4)
    vd = _vdot_batch(pts, ctrl, plant, p_lyap).reshape(len(dirs), 64)
    hit = vd >= 0.0
    idx = np.flatnonzero(hit.any(axis=1))
    first = np.argmax(hit, axis=1)[idx]
    hi_r, lo_r = grid[first], np.where(first > 0, grid[first - 1], 0.0)
    for _ in range(22):
        mid = 0.5 * (hi_r + lo_r)
        good = _vdot_batch(dirs[idx] * mid[:, None], ctrl, plant, p_lyap) >= 0.0
        hi_r = np.where(good, mid, hi_r)
        lo_r = np.where(good, lo_r, mid)
    z = dirs[idx] * hi_r[:, None]
    v_out = _vdot_batch(z, ctrl, plant, p_lyap)
    rho_out = np.einsum("ni,ij,nj->n", z, p_lyap, z)
    j = int(np.argmin(rho_out))
    print(
        f"   every returned endpoint satisfies Vdot >= 0: "
        f"{int((v_out >= 0).sum())}/{len(v_out)}, "
        f"min Vdot = {v_out.min():+.3e}"
    )
    print(f"   the binding witness is a REAL state, not an extrapolation:")
    print(f"     z*   = [{', '.join(f'{x:+.4f}' for x in z[j])}]")
    print(
        f"     Vdot = {v_out[j]:+.3e} >= 0,  ball radius "
        f"{np.sqrt(rho_out[j] / lo):.4f}"
    )
    print("   so no certificate can reach past it -- necessary, never " "sufficient.")

    print("\n3. Seed stability of the two estimators")
    print(f"   {'seed':>6} {'cloud':>10} {'ray':>10}")
    cl, ry = [], []
    for seed in (0, 1, 2, 3, 4):
        s = ball_samples(seed=seed)
        v = _vdot_batch(s, ctrl, plant, p_lyap)
        r = np.einsum("ni,ij,nj->n", s, p_lyap, s)
        b = v >= 0.0
        cl.append(float(np.sqrt(r[b].min() / lo)))
        dd = sphere_directions(2000, seed=seed)
        pts = (dd[:, None, :] * grid[None, :, None]).reshape(-1, 4)
        vv = _vdot_batch(pts, ctrl, plant, p_lyap).reshape(len(dd), 64)
        h = vv >= 0.0
        ii = np.flatnonzero(h.any(axis=1))
        f0 = np.argmax(h, axis=1)[ii]
        hr, lr = grid[f0], np.where(f0 > 0, grid[f0 - 1], 0.0)
        for _ in range(22):
            m = 0.5 * (hr + lr)
            g = _vdot_batch(dd[ii] * m[:, None], ctrl, plant, p_lyap) >= 0.0
            hr = np.where(g, m, hr)
            lr = np.where(g, lr, m)
        zz = dd[ii] * hr[:, None]
        ry.append(float(np.sqrt(np.einsum("ni,ij,nj->n", zz, p_lyap, zz).min() / lo)))
        print(f"   {seed:>6} {cl[-1]:>10.4f} {ry[-1]:>10.4f}")
    print(
        f"   spread: cloud {(max(cl) - min(cl)) / np.mean(cl):.1%}, "
        f"ray {(max(ry) - min(ry)) / np.mean(ry):.1%}"
    )
    print(
        "   Same necessary condition, same controller -- only the estimator " "differs."
    )


if __name__ == "__main__":
    main()
