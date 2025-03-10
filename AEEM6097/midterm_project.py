import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from AEEM6097.aco_solver import solve_aco, AcoContinuousVariable, AcoDiscreteVariable


def plot_all_the_things(
    x, y, Z_true, Z_approx: np.ndarray, x_mu: np.ndarray, y_mu: np.ndarray
):
    # Create a 2x2 subplot grid with 3D subplots in the top row
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "surface"}],
        ],
        subplot_titles=(
            "Membership Functions-X",
            "Membership Functions-Y",
            "True Function",
            "Approximated Function",
        ),
    )

    # Sample 2D functions - replace with your true and approximated functions
    x_2d = x[0, :].flatten()
    y_2d = y[:, 0].flatten()

    # Add 2D line plots to the top row
    for idx_mf in range(x_mu.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=x_2d,
                y=x_mu[:, idx_mf],
                mode="lines",
                name=f"X Membership Function-{idx_mf}",
            ),
            row=1,
            col=1,
        )
    for idx_mf in range(y_mu.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=y_2d,
                y=y_mu[:, idx_mf],
                mode="lines",
                name=f"Y Membership Function-{idx_mf}",
            ),
            row=1,
            col=2,
        )

    # Add 3D surface plots to the bottom row
    fig.add_trace(
        go.Surface(z=Z_true, x=x, y=y, colorscale="Viridis", showscale=False),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Surface(z=Z_approx, x=x, y=y, colorscale="Viridis", showscale=False),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text="Membership Functions and Function Approximation",
        height=800,
        width=1000,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Membership Value",
            camera=dict(
                eye=dict(x=0, y=0, z=2.5),  # Top-down view (high z, x and y centered)
                up=dict(x=0, y=1, z=0),
            ),
        ),
        scene2=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Membership Value",
            camera=dict(
                eye=dict(x=0, y=0, z=2.5),  # Top-down view (high z, x and y centered)
                up=dict(x=0, y=1, z=0),
            ),
        ),
    )

    # Update 2D plot axes
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_yaxes(title_text="Y", row=2, col=1)
    fig.update_xaxes(title_text="X", row=2, col=2)
    fig.update_yaxes(title_text="Y", row=2, col=2)

    # Show the plot
    fig.show()


def f_true(x, y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y)


def mu_poly(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + ((x - a) / b) ** 2)


def mu_poly_set(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.sort(p)
    mu_x = np.zeros((len(x), len(p)))
    for i in range(len(p)):
        if i < len(p) - 1:
            mu_x[:, i] = mu_poly(x, p[i], np.diff(p[i : i + 2]) / 2.0)
        else:
            mu_x[:, i] = mu_poly(x, p[i], np.diff(p[i - 1 : i + 1]) / 2.0)
    return mu_x


def mu_tri(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    if a == b:
        return np.maximum(np.minimum(1.0, (c - x) / b), 0.0)
    if b == c:
        np.maximum(np.minimum((x - a) / b, 1.0), 0.0)
    return np.maximum(np.minimum((x - a) / b, (c - x) / b), 0.0)


def mu_tri_set(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.sort(p)
    mu_x = np.zeros((len(x), len(p)))
    for i in range(len(p)):
        mu_x[:, i] = mu_tri(x, p[max(0, i - 1)], p[i], p[min(i + 1, len(p) - 1)])
    return mu_x


def tsk_rule(s: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    # Zeroth order TSK, or Mamdani
    if len(coeff) == 1:
        return coeff[0]
    first_order_size = s.shape[1] + 1
    if len(coeff) == first_order_size:
        X = np.ones((len(s), first_order_size))
        X[:, 1:] = s[:, :]
        return np.dot(X, coeff)
    # TODO - 2nd order TSK?
    else:
        raise ValueError(f"TSK rule requires {first_order_size} coefficients")


# TODO - Generate derivatives as well?
def fuzzy_or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.max([x, y], axis=0)
    # return x + y - x * y


def fuzzy_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.min([x, y], axis=0)
    # return x * y


RULE_AND = 1
RULE_OR = 0


def eval_rule(rule: np.ndarray, mu_x: np.ndarray, mu_y: np.ndarray, s: np.ndarray):
    op = rule[1]
    op1_selector = int(rule[0])
    op2_selector = int(rule[2])
    if op == RULE_AND:
        r_eval = fuzzy_and(mu_x[:, op1_selector], mu_y[:, op2_selector])
        z_eval = tsk_rule(s, rule[3:])
    elif op == RULE_OR:
        r_eval = fuzzy_or(mu_x[:, op1_selector], mu_y[:, op2_selector])
        z_eval = tsk_rule(s, rule[3:])
    else:
        raise ValueError(f"Unknown rule operator {op}")
    return r_eval, z_eval


def eval_rules(
    rules: np.ndarray, mu_x: np.ndarray, mu_y: np.ndarray, s: np.ndarray
) -> np.ndarray:
    n_rules = rules.shape[0]
    sum_R = 0.0
    sum_ZR = 0.0
    for i in range(n_rules):
        R, Z = eval_rule(rules[i, :], mu_x, mu_y, s)
        sum_R += R
        sum_ZR += Z * R
    return sum_ZR / sum_R


def compute_fuzzy_system(
    x: np.ndarray, pts: np.ndarray, n_mu: int, n_rule_idxs: int = 4
):
    mu_xp = np.sort(x[0:n_mu])
    mu_yp = np.sort(x[n_mu : 2 * n_mu])
    mu_x = mu_poly_set(pts[:, 0], mu_xp)
    mu_y = mu_poly_set(pts[:, 1], mu_yp)
    # Create the fuzzy rules, 0th order TSK
    rules = np.reshape(x[2 * n_mu :], (-1, n_rule_idxs))
    Z_defuzzy = eval_rules(rules, mu_x, mu_y, pts[:, 0:2])
    # Compute the RMS error
    rms_error = np.sqrt(np.mean((Z_defuzzy - pts[:, 2]) ** 2))
    return rms_error, Z_defuzzy


def aco_optimize(pts: np.ndarray, n_mu: int, n_rules: int, num_rule_idxs: int = 4):
    y_min = -np.pi
    y_max = np.pi
    x_min = -np.pi
    x_max = np.pi
    dy = (y_max - y_min) / n_mu
    dx = (x_max - x_min) / n_mu
    rule_options = [RULE_AND]
    mu_options = np.r_[0:n_mu]

    # Here is the goal-seeking function
    def fuzzy_test2(x: np.ndarray) -> float:
        rms_err, _ = compute_fuzzy_system(x, pts, n_mu, n_rule_idxs=num_rule_idxs)
        return rms_err

    aco_variables = []
    for i in range(n_mu):
        aco_variables.append(
            AcoContinuousVariable(
                f"mu_x{i}", x_min + i * dx, x_min + (i + 1) * dx, x_min + i * dx / 2.0
            )
        )
    for j in range(n_mu):
        aco_variables.append(
            AcoContinuousVariable(
                f"mu_y{j}", y_min + j * dy, y_min + (j + 1) * dy, y_min + j * dy / 2.0
            )
        )

    for k in range(n_rules):
        # Programmatically add the rules!
        aco_variables.extend(
            [
                AcoDiscreteVariable(f"rule{k}_A", mu_options),
                AcoDiscreteVariable(f"rule{k}_op", rule_options),
                AcoDiscreteVariable(f"rule{k}_B", mu_options),
                AcoContinuousVariable(f"rule{k}_c0", -1, 1),
            ]
        )
        # TODO - Allow more rules?
    best_soln, soln_history = solve_aco(
        fuzzy_test2, aco_variables, num_generations=100, num_ants=75
    )

    print(f"Best solution: {best_soln} with value: {soln_history[-1]}")
    # Plot solution history
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(soln_history)),
            y=soln_history,
            mode="lines",
            name="RMS Error",
        )
    )
    fig.update_layout(
        title="RMS Error vs. Generation",
        xaxis_title="Generation",
        yaxis_title="RMS Error",
    )
    fig.show()
    return best_soln, soln_history


def main():
    print("Test 2!")
    # 1600 datapoints, so 40 in each direction
    n_steps = 40
    # TODO - Randomly sample the domain
    # TODO - Allow X and Y to vary domain!
    s_min = -np.pi
    s_max = np.pi
    x = np.linspace(s_min, s_max, n_steps)
    y = np.linspace(s_min, s_max, n_steps)
    X, Y = np.meshgrid(x, y)
    # Compute the true function
    Z = f_true(X, Y)
    # Create pairs of points
    pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    rand_pts = pts.copy()
    # Randomly reorder the points
    np.random.shuffle(rand_pts)
    # Get the training dataset, and the testing dataset
    n_train = Z.size // 4 * 3
    pts_train = rand_pts[:n_train, :]
    pts_test = rand_pts[n_train:, :]

    # Solve for the optimal system!
    n_mu = 5
    num_rule_ = 4
    n_rules = 6
    best_soln, soln_history = aco_optimize(
        pts_train, n_mu, n_rules, num_rule_idxs=num_rule_
    )
    rms_err, _ = compute_fuzzy_system(best_soln, pts_test, n_mu, n_rule_idxs=num_rule_)
    print(f"Train error={soln_history[-1]}, Test error={rms_err}")

    rms_error, Z_defuzzy = compute_fuzzy_system(
        best_soln, pts, n_mu, n_rule_idxs=num_rule_
    )
    print(f"RMS error: {rms_error}")
    # Reshape into the 2D array
    Z_defuzzy = Z_defuzzy.reshape(X.shape)
    # TODO - Train the fuzzy model!
    # TODO - Test the fuzzy model!
    # TODO - Plot the model!
    # Plot using plotly
    mu_xp = best_soln[0:n_mu]
    mu_yp = best_soln[n_mu : 2 * n_mu]
    mu_x_plot = mu_poly_set(x, mu_xp)
    mu_y_plot = mu_poly_set(y, mu_yp)
    plot_all_the_things(X, Y, Z, Z_defuzzy, mu_x_plot, mu_y_plot)


if __name__ == "__main__":
    main()
