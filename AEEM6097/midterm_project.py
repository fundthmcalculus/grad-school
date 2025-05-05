import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from AEEM6097.aco_solver import AcoContinuousVariable, solve_gradiant

# Some typing information shorthand!
f64 = np.float64
af64 = NDArray[f64]


def plot_all_the_things(
    x, y, z_true: af64, z_approx: af64, x_mu: af64, y_mu: af64
):
    # Plot the membership functions
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    for ij in range(x_mu.shape[1]):
        plt.plot(x[0,:],x_mu[:,ij], label=f'Mu-X-{ij}')
    plt.title("Membership Functions-X")
    plt.ylabel("mu")
    plt.xlabel("x")
    plt.subplot(2, 2, 2)
    for ij in range(y_mu.shape[1]):
        plt.plot(y[:,0],y_mu[:,ij], label=f'Mu-Y-{ij}')
    plt.title("Membership Functions-Y")
    plt.ylabel("mu")
    plt.xlabel("y")
    plt.subplot(2, 2, 3)
    im_extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    plt.imshow(z_true, extent=im_extents)
    plt.title("True Z")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.subplot(2, 2, 4)
    plt.imshow(z_approx, extent=im_extents)
    plt.title("Approx Z")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.tight_layout()
    plt.show()


def f_true(x, y: af64) -> af64:
    return np.sin(x) * np.cos(y)


def mu_poly(x: af64, a: float, b: float) -> af64:
    return 1.0 / (1.0 + ((x - a) / b) ** 2)


def expsq(x: af64, a: f64, b: f64) -> af64:
    """exponential quadratic membership function"""
    return np.exp(-((x-a)/b)**2.0)


def mu_poly_set(s: af64, mu_a: af64, mu_b: af64) -> af64:
    mu_s = np.zeros((len(s), len(mu_a)))
    for i in range(len(mu_a)):
        mu_s[:, i] = mu_poly(s, mu_a[i], mu_b[i])
    return mu_s


def tsk_rule(s: af64, coeff: af64) -> af64:
    # Zeroth order TSK, or Mamdani
    if len(coeff) == 1:
        return coeff[0]
    first_order_size = s.shape[1] + 1
    if len(coeff) == first_order_size:
        fuzzy_s = np.ones((len(s), first_order_size))
        fuzzy_s[:, 1:] = s[:, :]
        return np.dot(fuzzy_s, coeff)
    # TODO - 2nd order TSK?
    else:
        raise ValueError(f"TSK rule requires {first_order_size} coefficients")


def fuzzy_or(x: af64, y: af64) -> af64:
    # return np.max([x, y], axis=0)
    return x + y - x * y

def fuzzy_and(x: af64, y: af64) -> af64:
    # return np.min([x, y], axis=0)
    return x * y


RULE_AND = 1
RULE_OR = 0


def eval_rule(rule: af64, mu_x: af64, mu_y: af64, s: af64):
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
    rules: af64, mu_x: af64, mu_y: af64, s: af64
) -> af64:
    n_rules = rules.shape[0]
    sum_R = 0.0
    sum_ZR = 0.0
    for i in range(n_rules):
        R, Z = eval_rule(rules[i, :], mu_x, mu_y, s)
        sum_R += R
        sum_ZR += Z * R
    return sum_ZR / sum_R


def compute_fuzzy_system(x: af64, pts: af64, n_mu: int)-> tuple[f64, af64]:
    mu_x, mu_y = extract_mu_from_args(n_mu, pts, x)
    # Do the rules in order
    rule_idx = 0
    sum_R = 0.0
    sum_ZR = 0.0
    for ix in range(n_mu):
        for iy in range(n_mu):
            arg_rule_idx = 4 * n_mu + 4 * rule_idx
            and_c = x[arg_rule_idx]
            # f_a = x[arg_rule_idx+1]
            # f_b = x[arg_rule_idx+2]
            # f_c = x[arg_rule_idx+3]
            # NOTE - This is magic!
            r_eval_and = fuzzy_and(mu_x[:, ix], mu_y[:, iy])
            # Exclude the third column we don't include the output.
            z_eval = tsk_rule(pts[:,0:2], x[arg_rule_idx+1:arg_rule_idx+4])
            r_eval_or = fuzzy_or(mu_x[:, ix], mu_y[:, iy])
            r_eval = (1-and_c)*r_eval_or+and_c*r_eval_and
            # Fuzzify!
            sum_R += r_eval
            sum_ZR += r_eval*z_eval

            rule_idx+=1

    # Weighted defuzzy!
    z_defuzzy = sum_ZR / sum_R
    # Compute the RMS error
    rms_error = np.sqrt(np.mean((z_defuzzy - pts[:, 2]) ** 2))
    return rms_error, z_defuzzy


def extract_mu_from_args(n_mu, pts, x):
    mu_x_a = x[0:2 * n_mu:2]
    mu_x_b = x[1:2 * n_mu:2]
    mu_y_a = x[2 * n_mu:4 * n_mu:2]
    mu_y_b = x[2 * n_mu + 1:4 * n_mu:2]
    mu_x = mu_poly_set(pts[:, 0], mu_x_a, mu_x_b)
    mu_y = mu_poly_set(pts[:, 1], mu_y_a, mu_y_b)
    return mu_x, mu_y


def aco_optimize(pts: af64, n_mu: int, n_rules: int):
    y_min = -np.pi
    y_max = np.pi
    x_min = -np.pi
    x_max = np.pi
    dy = (y_max - y_min) / n_mu
    dx = (x_max - x_min) / n_mu

    aco_variables = []
    for i in range(n_mu):
        aco_variables.extend([
            AcoContinuousVariable(
                f"mu_x{i}-a", x_min, x_max, x_min + i * dx / 2.0
            ),
            AcoContinuousVariable(
                f"mu_x{i}-b", 0.01, 2*dx,0.05
            )]
        )
    for i in range(n_mu):
        aco_variables.extend([
            AcoContinuousVariable(
                f"mu_y{i}-a", y_min, y_max, y_min + i * dy / 2.0
            ),
            AcoContinuousVariable(
                f"mu_y{i}-b", 0.01, 2*dy,0.05
            )]
        )

    for k in range(n_rules):
        aco_variables.extend(
            [
                # NOTE - This is the exploratory magic of degree of or/and
                AcoContinuousVariable(f"rule{k}-and-op", 0, 1, 1.0),
                AcoContinuousVariable(f"rule{k}-f-a", -3, 3),
                AcoContinuousVariable(f"rule{k}-f-b", -3, 3),
                AcoContinuousVariable(f"rule{k}-f-c", -3, 3, 0.0),
            ]
        )

    # Here is the goal-seeking function
    def fuzzy_test2(x: af64) -> f64:
        rms_err, _ = compute_fuzzy_system(x, pts, n_mu)
        return rms_err

    best_soln, soln_history = solve_gradiant(fuzzy_test2, aco_variables)
    return best_soln, soln_history


def main():
    print("Mid-Term Project!")
    X, Y, Z, pts, pts_test, pts_train, x, y = setup_dataset()
    # Solve for the optimal system!
    n_mu = 2
    n_vars = 2
    n_rules = n_mu ** n_vars
    # NOTE - full-factorial on rules is bad in general!

    best_soln, soln_history = aco_optimize(pts_train, n_mu, n_rules)
    rms_err, _ = compute_fuzzy_system(best_soln, pts_test, n_mu)
    print(f"Train error={soln_history[-1]}, Test error={rms_err}")

    rms_error, z_defuzzy = compute_fuzzy_system(best_soln, pts, n_mu)
    print(f"Overall RMS error: {rms_error}")
    # Reshape into the 2D array
    z_defuzzy = z_defuzzy.reshape(X.shape)
    # Plot using plotly
    mu_x_plot, mu_y_plot = extract_mu_from_args(n_mu, np.hstack((np.reshape(x,(len(X),1)), np.reshape(y,(len(X),1)))), best_soln)
    plot_all_the_things(X, Y, Z, z_defuzzy, mu_x_plot, mu_y_plot)


def setup_dataset():
    # 1600 datapoints, so 40 in each direction
    n_steps = 40
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
    return X, Y, Z, pts, pts_test, pts_train, x, y


if __name__ == "__main__":
    main()
