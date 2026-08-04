"""
Checks whether n_output_buckets=3 (used by every black-box ablation in this
project, matching the original n=2 study's default) was quietly under-
serving some of those comparisons -- see N3_N5_FUZZY_REGRESSION_REPORT.md
SS4.7 for the full write-up.

MimoGaussianPredictor's n_output_buckets partitions the TARGET y into that
many value ranges, then fits antecedent memberships and a local TSK
consequent per bucket from whatever rows land in that y-range (see
gaussian_regressor.py fit()). For a target that's genuinely linear across
its whole domain (exactly what n2_physics_informed_v2_rational.py's known-
denominator features produce), low bucket counts give each local fit a
restricted, biased slice of x-space; it converges to the correct global fit
as bucket count rises. For a target that's missing information no local
fit can supply (angle-only inputs, which never see velocity), no bucket
count fixes that.

Three things this script checks, printed as it goes:
  1. Physics-basis single-step R^2 vs. bucket count (should converge to the
     plain-OLS ~0.99 as buckets rise -- confirms the diagnosis, not a
     library bug).
  2. Rollout time-to-0.5rad-error vs. bucket count, for every n=2 black-box
     ablation and the n=3 baseline (does raising resolution change the
     "black-box fails fast" conclusion?).
  3. Rollout time-to-0.5rad-error vs. bucket count for the physics-informed
     consequent refit through genuine multi-rule MimoGaussianPredictor
     (does it approach the plain-regression ceiling?).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from test_fuzzy_ode import initialize_model
from n_pendulum_fuzzy_regression import (generate_family, load_mimo_data, train_mimo,
                                          run_iterative_prediction)
from n2_moving_average_fuzzy import (generate_training_set as ma_generate_training_set,
                                      build_dataset as ma_build_dataset,
                                      rollout as ma_rollout, INPUT_COLS as MA_INPUT_COLS,
                                      FEATURE_NAMES as MA_FEATURE_NAMES)
from n2_physics_informed_v2_rational import (build_dataset as phys2_build_dataset,
                                              rational_basis_features, ALPHA1_COLS, ALPHA2_COLS,
                                              RATIONAL_BASIS_COLS)
from n_pendulum_physics_basis import derive_physics_basis, compute_features, state_cols

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tribble-fis" / "src"))
from tribblefis.gaussian_regressor import MimoGaussianPredictor

DT = 0.01
BUCKET_COUNTS = [3, 10, 20, 40]


def time_to_threshold(t, err, threshold=0.5):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


def part1_physics_basis_r2_vs_buckets():
    print("=" * 70)
    print("PART 1: n=2 physics-basis single-step R^2 vs. bucket count")
    print("=" * 70)
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]
    pendulum = train_results.model
    l1, l2, m1, m2 = pendulum.l1, pendulum.l2, pendulum.m1, pendulum.m2

    X_train, y_train = phys2_build_dataset(train_results.trajectories, l1, l2, m1, m2)
    X_test, y_test = phys2_build_dataset([tst], l1, l2, m1, m2)
    X_train_df = pd.DataFrame(X_train, columns=RATIONAL_BASIS_COLS)[ALPHA1_COLS]
    X_test_df = pd.DataFrame(X_test, columns=RATIONAL_BASIS_COLS)[ALPHA1_COLS]

    for nb in [2, 3, 5] + BUCKET_COUNTS[1:]:
        reg = MimoGaussianPredictor(n_output_buckets=nb, tsk_order="1st",
                                     optimize_coefficients=True, random_state=42, top_p=1.0)
        reg.fit(X_train_df, pd.DataFrame(y_train[:, 0], columns=['omega_1']))
        pred = reg.predict(X_test_df)
        print(f"  n_output_buckets={nb}: R2={r2_score(y_test[:, 0], pred['omega_1']):.4f}")


def part2_blackbox_sweep():
    print("\n" + "=" * 70)
    print("PART 2: black-box baselines, rollout time-to-0.5rad vs. bucket count")
    print("=" * 70)
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]

    def sweep(feature_names, label):
        Xtr, ytr = load_mimo_data(train_results.trajectories, feature_names, window_size=1)
        Xte, yte = load_mimo_data([tst], feature_names, window_size=1)
        seed = tst[feature_names].iloc[:1].reset_index(drop=True)
        print(f"\n  {label}:")
        for nb in BUCKET_COUNTS:
            res = train_mimo(feature_names, Xtr, ytr, Xte, yte, window_size=1, n_bins=nb)
            pred = run_iterative_prediction(res['regressor'], seed, feature_names, len(tst) - 1, window_size=1)
            t = np.arange(len(pred)) * DT
            err = np.abs(pred['theta_1'].values - tst['theta_1'].values[:len(pred)])
            print(f"    n_bins={nb}: time-to-0.5rad={time_to_threshold(t, err)}")

    sweep(['theta_1', 'theta_2'], 'n=2 angle-only')
    sweep(['theta_1', 'omega_1', 'theta_2', 'omega_2'], 'n=2 angle+velocity')

    print("\n  n=2 moving-average (101 traj):")
    pendulum, trajectories, test_trajectory, _grid = ma_generate_training_set()
    X_train, y_train = ma_build_dataset(trajectories)
    X_test, y_test = ma_build_dataset([test_trajectory])
    for nb in BUCKET_COUNTS:
        regr = MimoGaussianPredictor(n_output_buckets=nb, tsk_order="1st",
                                      optimize_coefficients=True, random_state=42)
        regr.fit(pd.DataFrame(X_train, columns=MA_INPUT_COLS), pd.DataFrame(y_train, columns=MA_FEATURE_NAMES))
        pred = ma_rollout(regr, test_trajectory, dt=DT)
        t = np.arange(len(pred)) * DT
        err = np.abs(pred['theta_1'].values - test_trajectory['theta_1'].values[:len(pred)])
        print(f"    n_bins={nb}: time-to-0.5rad={time_to_threshold(t, err)}")

    print("\n  n=3 angle-only (fan configuration):")
    family = generate_family(3, [120.0, 60.0, 0.0], 2, np.arange(1.5, 3.00001, 0.1),
                              test_delta_deg=2.05, dt=DT, duration=30.0)
    feature_names = ['theta_1', 'theta_2', 'theta_3']
    Xtr, ytr = load_mimo_data(family.train_trajectories, feature_names, window_size=1)
    Xte, yte = load_mimo_data([family.test_trajectory], feature_names, window_size=1)
    seed = family.test_trajectory[feature_names].iloc[:1].reset_index(drop=True)
    for nb in BUCKET_COUNTS:
        res = train_mimo(feature_names, Xtr, ytr, Xte, yte, window_size=1, n_bins=nb)
        pred = run_iterative_prediction(res['regressor'], seed, feature_names,
                                         len(family.test_trajectory) - 1, window_size=1)
        t = np.arange(len(pred)) * DT
        err = np.abs(pred['theta_1'].values - family.test_trajectory['theta_1'].values[:len(pred)])
        print(f"    n_bins={nb}: time-to-0.5rad={time_to_threshold(t, err)}")


def part3_physics_informed_fuzzy_sweep():
    print("\n" + "=" * 70)
    print("PART 3: physics-informed consequent via genuine MimoGaussianPredictor")
    print("=" * 70)

    print("\n  n=2:")
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]
    pendulum = train_results.model
    l1, l2, m1, m2 = pendulum.l1, pendulum.l2, pendulum.m1, pendulum.m2
    X_train, y_train = phys2_build_dataset(train_results.trajectories, l1, l2, m1, m2)
    X_train_df = pd.DataFrame(X_train, columns=RATIONAL_BASIS_COLS)
    y_train_df = pd.DataFrame(y_train, columns=['omega_1', 'omega_2'])

    class FuzzyPhysicsRegressor2:
        def __init__(self, nb):
            self.r1 = MimoGaussianPredictor(n_output_buckets=nb, tsk_order="1st",
                                             optimize_coefficients=True, random_state=42, top_p=1.0)
            self.r2 = MimoGaussianPredictor(n_output_buckets=nb, tsk_order="1st",
                                             optimize_coefficients=True, random_state=42, top_p=1.0)

        def fit(self, Xdf, ydf):
            self.r1.fit(Xdf[ALPHA1_COLS], ydf[['omega_1']])
            self.r2.fit(Xdf[ALPHA2_COLS], ydf[['omega_2']])
            return self

        def predict(self, Xdf):
            p1 = self.r1.predict(Xdf[ALPHA1_COLS])
            p2 = self.r2.predict(Xdf[ALPHA2_COLS])
            return pd.DataFrame({'omega_1': p1['omega_1'].values, 'omega_2': p2['omega_2'].values})

    def rollout2(regressor, test_trajectory, n_steps=None):
        state_cols_ = ['theta_1', 'omega_1', 'theta_2', 'omega_2']
        state = test_trajectory[state_cols_].iloc[0].copy()
        total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
        rows = [state.to_dict()]
        for step in range(total_steps):
            x_now = rational_basis_features(pd.DataFrame([state.to_dict()]), l1, l2, m1, m2)
            delta_omega = regressor.predict(x_now).iloc[0]
            omega1_new = state['omega_1'] + delta_omega['omega_1']
            omega2_new = state['omega_2'] + delta_omega['omega_2']
            theta1_new = state['theta_1'] + omega1_new * DT
            theta2_new = state['theta_2'] + omega2_new * DT
            new_state = {'theta_1': theta1_new, 'omega_1': omega1_new,
                         'theta_2': theta2_new, 'omega_2': omega2_new}
            if not np.isfinite(list(new_state.values())).all() or any(abs(v) > 1e4 for v in new_state.values()):
                for _ in range(total_steps - step):
                    rows.append({c: np.nan for c in state_cols_})
                break
            state = pd.Series(new_state)
            rows.append(new_state)
        return pd.DataFrame(rows)

    for nb in [20, 40]:
        reg = FuzzyPhysicsRegressor2(nb).fit(X_train_df, y_train_df)
        pred = rollout2(reg, tst)
        t = np.arange(len(pred)) * DT
        err = np.abs(pred['theta_1'].values - tst['theta_1'].values[:len(pred)])
        print(f"    n_output_buckets={nb}: time-to-0.5rad={time_to_threshold(t, err)}")

    print("\n  n=3:")
    n = 3
    theta_cols, omega_cols = state_cols(n)
    basis = derive_physics_basis(n, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 9.81)
    family = generate_family(n, [120.0, 60.0, 0.0], 2, np.arange(1.5, 3.00001, 0.1),
                              test_delta_deg=2.05, dt=DT, duration=30.0)

    def build3(trajectories):
        all_X = [[] for _ in range(n)]
        all_y = []
        for df in trajectories:
            theta_arrs = [df[c].values[:-1] for c in theta_cols]
            omega_arrs = [df[c].values[:-1] for c in omega_cols]
            _denom, feats = compute_features(basis, theta_arrs, omega_arrs)
            for i in range(n):
                all_X[i].append(feats[i])
            all_y.append(np.diff(df[omega_cols].values, axis=0))
        return [np.vstack(x) for x in all_X], np.vstack(all_y)

    X_train3, y_train3 = build3(family.train_trajectories)

    class FuzzyPhysicsRegressorN:
        def __init__(self, n, nb):
            self.n = n
            self.models = [MimoGaussianPredictor(n_output_buckets=nb, tsk_order="1st",
                                                  optimize_coefficients=True, random_state=42, top_p=1.0)
                           for _ in range(n)]

        def fit(self, Xs, y):
            for i in range(self.n):
                cols = [f'f{j}' for j in range(Xs[i].shape[1])]
                self.models[i].fit(pd.DataFrame(Xs[i], columns=cols), pd.DataFrame(y[:, i], columns=['omega']))
            return self

        def predict_row(self, theta_row, omega_row):
            theta_arrs = [np.array([v]) for v in theta_row]
            omega_arrs = [np.array([v]) for v in omega_row]
            _denom, feats = compute_features(basis, theta_arrs, omega_arrs)
            out = []
            for i in range(self.n):
                cols = [f'f{j}' for j in range(feats[i].shape[1])]
                p = self.models[i].predict(pd.DataFrame(feats[i], columns=cols))
                out.append(p['omega'].values[0])
            return np.array(out)

    def rollout3(reg, test_trajectory, n_steps=None):
        theta = test_trajectory[theta_cols].iloc[0].values.astype(float)
        omega = test_trajectory[omega_cols].iloc[0].values.astype(float)
        total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
        rows = [dict(zip(theta_cols + omega_cols, np.concatenate([theta, omega])))]
        for step in range(total_steps):
            delta_omega = reg.predict_row(theta, omega)
            omega_new = omega + delta_omega
            theta_new = theta + omega_new * DT
            state = np.concatenate([theta_new, omega_new])
            if not np.isfinite(state).all() or np.any(np.abs(state) > 1e4):
                for _ in range(total_steps - step):
                    rows.append({c: np.nan for c in theta_cols + omega_cols})
                break
            theta, omega = theta_new, omega_new
            rows.append(dict(zip(theta_cols + omega_cols, state)))
        return pd.DataFrame(rows)

    for nb in [10, 20, 40]:
        reg = FuzzyPhysicsRegressorN(n, nb).fit(X_train3, y_train3)
        pred = rollout3(reg, family.test_trajectory)
        t = np.arange(len(pred)) * DT
        err = np.abs(pred['theta_1'].values - family.test_trajectory['theta_1'].values[:len(pred)])
        print(f"    n_output_buckets={nb}: time-to-0.5rad={time_to_threshold(t, err)}")


if __name__ == '__main__':
    t0 = time.perf_counter()
    part1_physics_basis_r2_vs_buckets()
    part2_blackbox_sweep()
    part3_physics_informed_fuzzy_sweep()
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
