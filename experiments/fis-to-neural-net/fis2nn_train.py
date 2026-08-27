"""Training loop and metrics shared by every arm of the experiment.

Written against numpy only, deliberately: the point is that the converted
network is an ordinary MLP that any framework can consume, and a 60-line Adam
loop keeps the comparison between initializations free of framework defaults
that would otherwise differ between arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fis2nn_network import ReLUNet


@dataclass
class TrainHistory:
    epochs: list[int]
    train_rmse: list[float]
    test_rmse: list[float]
    val_rmse: list[float]
    seconds: list[float]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, used for the binary-classification arms."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse of :func:`sigmoid`, clipped away from the asymptotes.

    The conversion seeds a *logit*, not a probability: the network's scalar
    output is what a sigmoid is applied to, so backing the FIS out in
    probability space and then squashing it again would compose two sigmoids
    and misplace every weight. A FIS that returns a hard 0 or 1 -- which
    `TribbleClassifier` does routinely, since a normalized firing strength can
    saturate -- would otherwise map to an infinite target.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def error_rate(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Misclassification rate at a 0 logit threshold."""
    return float(
        np.mean(
            (np.asarray(logits).ravel() > 0.0) != (np.asarray(y_true).ravel() > 0.5)
        )
    )


def log_loss(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy, computed from logits without forming p."""
    z = np.asarray(logits, dtype=float).ravel()
    t = np.asarray(y_true, dtype=float).ravel()
    return float(np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = (
        np.asarray(y_true, dtype=float).ravel()
        - np.asarray(y_pred, dtype=float).ravel()
    )
    return float(np.sqrt(np.mean(d * d)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_adam(
    net: ReLUNet,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 3e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int = 0,
    eval_every: int = 1,
    eval_batches: int | None = None,
    track_train: bool = True,
    y_scale: float = 1.0,
    y_center: float = 0.0,
    loss: str = "mse",
    metric_fn=None,
) -> tuple[ReLUNet, TrainHistory]:
    """Minibatch Adam on the MSE, identical for every arm.

    ``y_scale``/``y_center`` map the network's (standardized) output back to the
    target's own units for reporting, so an arm is never scored in a frame of
    its own choosing.

    ``loss="bce"`` treats the network's scalar output as a **logit** and
    optimizes binary cross-entropy instead. Only the output-layer gradient
    changes -- ``sigmoid(pred) - y`` in place of ``2 * (pred - y)`` -- because
    everything below it is the same network; keeping one training loop for both
    is what stops the regression and classification arms differing by an
    optimizer detail nobody meant to introduce. ``metric_fn(y_true, raw_pred)``
    overrides the reported curve (error rate rather than RMSE, for instance).

    ``eval_batches`` records the curve every N *minibatches* instead of every
    epoch, and the recorded "epoch" becomes fractional. At 160k rows an epoch is
    313 updates and a network can cross every quality target inside the first
    one, which makes an epoch-resolution time-to-target table read as a row of
    ties. This is the knob that makes the comparison measurable at scale rather
    than a statement about the granularity of the ruler.
    """
    import time

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    net = net.copy()

    params = ["W1", "b1", "w2", "v", "c"]
    m = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    vv = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    t = 0

    hist = TrainHistory([], [], [], [], [])

    if loss not in ("mse", "bce"):
        raise ValueError(f"loss must be 'mse' or 'bce', got {loss!r}")

    def _score(Xe, ye):
        if Xe is None or ye is None:
            return float("nan")
        if metric_fn is not None:
            return float(metric_fn(np.asarray(ye).ravel(), net.predict(Xe)))
        return rmse(
            np.asarray(ye).ravel() * y_scale + y_center,
            net.predict(Xe) * y_scale + y_center,
        )

    # Wall clock spent *measuring* rather than training. `hist.seconds` used to
    # be `perf_counter() - start`, which charged every prior evaluation pass
    # over X_test/X_val to the training time it was supposed to be reporting.
    # That cancels between arms of equal width and does not cancel otherwise --
    # and the comparison this module exists for puts a hot arm whose width is
    # fixed by the FIS's knot count (264 units on N-CMAPSS DS02) against an `he`
    # arm free to be narrow (8), a 33x difference in per-evaluation cost billed
    # to the wrong column. Subtracting it makes every recorded second a second
    # of gradient descent.
    eval_seconds = 0.0

    def record(epoch: float) -> None:
        nonlocal eval_seconds
        t_rec = time.perf_counter()
        hist.epochs.append(epoch)
        hist.seconds.append(t_rec - start - eval_seconds)
        # Scoring the training set is the most expensive part of a record, and
        # at sub-epoch cadence on 160k rows it costs more than the training it
        # is measuring. Nothing in this experiment selects on the train curve.
        hist.train_rmse.append(_score(X, y) if track_train else float("nan"))
        hist.test_rmse.append(_score(X_test, y_test))
        hist.val_rmse.append(_score(X_val, y_val))
        eval_seconds += time.perf_counter() - t_rec

    start = time.perf_counter()
    record(0.0)
    n_batches = max(1, int(np.ceil(n / batch_size)))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        # `bi` and not `b`: `b` is the batch *size* three lines down, and
        # letting the index share the name silently disabled sub-epoch eval.
        for bi, lo in enumerate(range(0, n, batch_size)):
            idx = order[lo : lo + batch_size]
            Xb, yb = X[idx], y[idx]
            b = Xb.shape[0]

            z = Xb @ net.W1 + net.b1
            h = np.maximum(z, 0.0)
            pred = h @ net.w2 + Xb @ net.v + net.c
            if loss == "bce":
                g_out = (sigmoid(pred) - yb) / b
            else:
                g_out = (2.0 / b) * (pred - yb)

            grads = {
                "w2": h.T @ g_out,
                "v": Xb.T @ g_out,
                "c": np.atleast_1d(float(g_out.sum())),
            }
            g_h = np.outer(g_out, net.w2) * (z > 0.0)
            grads["W1"] = Xb.T @ g_h
            grads["b1"] = g_h.sum(axis=0)

            t += 1
            for p in params:
                g = grads[p]
                m[p] = beta1 * m[p] + (1 - beta1) * g
                vv[p] = beta2 * vv[p] + (1 - beta2) * (g * g)
                m_hat = m[p] / (1 - beta1**t)
                v_hat = vv[p] / (1 - beta2**t)
                step = lr * m_hat / (np.sqrt(v_hat) + eps)
                if p == "c":
                    net.c = float(net.c - step[0])
                else:
                    setattr(net, p, getattr(net, p) - step)

            if eval_batches and (bi + 1) % eval_batches == 0:
                record(epoch - 1 + (bi + 1) / n_batches)

        if eval_batches is None and (epoch % eval_every == 0 or epoch == epochs):
            record(float(epoch))
    if eval_batches:
        record(float(epochs))

    return net, hist
