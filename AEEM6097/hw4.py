import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fuzzy_and(a, b):
    return np.minimum(a, b)


def fuzzy_or(a, b):
    return np.maximum(a, b)


def fuzzy_not(a):
    return 1 - a


def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)


def linzmf(x, a, b):
    return np.maximum(np.minimum((b - x) / (b - a), 1), 0)


def linsmf(x, a, b):
    return np.maximum(np.minimum((x - a) / (b - a), 1), 0)


def trapmf(x, a, b, c, d):
    return np.maximum(np.minimum((x - a) / (b - a), 1, (d - x) / (d - c)), 0)


# Key points for membership functions
x0 = 0
x1 = 2.5  # ADDED for part 2.1
x2 = 5
x3 = 7.5  # ADDED for part 2.1
x4 = 10

y0 = x0**2
y1 = x1**2
y2 = x2**2
y3 = x3**2
y4 = x4**2


def mu_N(y):
    return linzmf(y, y0, y1)


def mu_V(y):
    return linsmf(y, y3, y4)


def mu_W(y):
    return trimf(y, y0, y1, y2)


def mu_S(y):
    return trimf(y, y1, y2, y3)


def mu_T(y):
    return trimf(y, y2, y3, y4)


def lerp(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# Define the membership functions
def mu_Z(x):
    return trimf(x, -x1, x0, x1)


def mu_NZ2(x):
    return trimf(x, -x2, -x1, x0)


def mu_PZ2(x):
    return trimf(x, x0, x1, x2)


def mu_NB(x):
    return linzmf(x, -x4, -x3)


def mu_NS(x):
    return trimf(x, -x3, -x2, -x1)


def mu_NT(x):
    return trimf(x, -x4, -x3, -x2)


def mu_PS(x):
    return trimf(x, x1, x2, x3)


def mu_PT(x):
    return trimf(x, x2, x3, x4)


def mu_PB(x):
    return linsmf(x, x3, x4)


# IF x = Z, THEN y = N
def rule_1(x):
    return mu_Z(x)


# IF x = NS or PS, THEN y = S
def rule_2(x):
    return fuzzy_or(mu_NS(x), mu_PS(x))


# IF x = NB or PB, THEN y = V
def rule_3(x):
    return fuzzy_or(mu_NB(x), mu_PB(x))


# IF x = NZ2 or PZ2, THEN y = W
def rule_4(x):
    return fuzzy_or(mu_NZ2(x), mu_PZ2(x))


# IF x = NT or PT, THEN y = W
def rule_5(x):
    return fuzzy_or(mu_NT(x), mu_PT(x))


def problem2_1():
    # Develop Mamdani fuzzy inference system for the following problem:
    print("Part 2.1")
    x_freq = np.linspace(-10, 10, 2001, dtype=np.float64)
    y_true = x_freq**2
    y_defuzzy = np.zeros_like(x_freq, dtype=np.float64)

    # TODO - Vectorize this!
    for ij, x in enumerate(x_freq):
        # Fire all the rules
        mu_N_ij = rule_1(x)
        mu_S_ij = rule_2(x)
        mu_V_ij = rule_3(x)
        mu_W_ij = rule_4(x)
        mu_T_ij = rule_5(x)
        # Approximate the centroid of the composite shape
        mu_rule_1 = np.minimum.reduce([mu_N_ij * np.ones_like(y_true), mu_N(y_true)])
        mu_rule_2 = np.minimum.reduce([mu_S_ij * np.ones_like(y_true), mu_S(y_true)])
        mu_rule_3 = np.minimum.reduce([mu_V_ij * np.ones_like(y_true), mu_V(y_true)])
        mu_rule_4 = np.minimum.reduce([mu_W_ij * np.ones_like(y_true), mu_W(y_true)])
        mu_rule_5 = np.minimum.reduce([mu_T_ij * np.ones_like(y_true), mu_T(y_true)])
        mu_all_rules = np.maximum.reduce(
            [mu_rule_1, mu_rule_2, mu_rule_3, mu_rule_4, mu_rule_5]
        )
        y_defuzzy[ij] = np.sum(y_true * mu_all_rules) / np.sum(mu_all_rules)

    # Create subplots with 1 row and 3 columns
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Membership Functions of x",
            "",
            "Membership Functions of y",
            "Overlay of y-partitions, y, and y = x^2",
        ),
    )

    # Add traces for mu_ functions in the first column
    fig.add_trace(
        go.Scatter(x=x_freq, y=mu_Z(x_freq), mode="lines", name="Z"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_freq, y=mu_NB(x_freq), mode="lines", name="NB"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_freq, y=mu_NS(x_freq), mode="lines", name="NS"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_freq, y=mu_PS(x_freq), mode="lines", name="PS"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_freq, y=mu_PB(x_freq), mode="lines", name="PB"), row=1, col=1
    )

    # Add traces for y_ functions in the second column
    fig.add_trace(
        go.Scatter(x=y_true, y=rule_1(x_freq), mode="lines", name="rule-1"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=y_true, y=rule_2(x_freq), mode="lines", name="rule-2"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=y_true, y=rule_3(x_freq), mode="lines", name="rule-3"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=y_true, y=rule_4(x_freq), mode="lines", name="rule-4"),
        row=2,
        col=1,
    )

    # Add traces for the overlay of y and y = x^2 in the third column
    fig.add_trace(
        go.Scatter(x=x_freq, y=y_defuzzy, mode="lines", name="y-fuzzy"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x_freq, y=y_true, mode="lines", name="y = x^2"), row=2, col=2
    )

    # Calculate RMS error between `y` and `y = x^2`
    rms_error = np.sqrt(np.mean((np.array(y_true) - np.array(y_defuzzy)) ** 2))
    fig.add_annotation(
        text=f"RMS Error: {rms_error:.4f}",
        xref="paper",
        yref="paper",
        x=2,
        y=50,
        showarrow=False,
        align="center",
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(title="Problem2.1", showlegend=True)

    # Show the plot
    fig.show()


def mu_prob2_2(x, n_fcns=7, x_min=-10, x_max=10):
    # TODO - Vectorize this
    # Define the membership functions
    mu = np.array([0.0] * n_fcns)
    fcn_domain = (x_max - x_min) / n_fcns
    for i in range(1, n_fcns - 1):
        mu[i] = trimf(
            x,
            x_min + (i - 1) * fcn_domain,
            x_min + i * fcn_domain,
            x_min + (i + 1) * fcn_domain,
        )
    mu[0] = linzmf(x, x_min, x_min + fcn_domain)
    mu[-1] = linsmf(x, x_max - fcn_domain, x_max)

    return mu


def takagi_sugeno_functions(x, n_fcns=7, x_min=-10, x_max=10):
    # Utilizing Ross solution manual as initial guess to demonstrate
    x_arr = np.linspace(x_min, x_max, n_fcns + 1, dtype=np.float64)
    y_arr = x_arr**2
    # TODO - Handle more functions?
    ts_y = np.array([0.0] * (len(x_arr) - 1))
    for i in range(1, len(x_arr)):
        ts_y[i - 1] = lerp(x, x_arr[i - 1], x_arr[i], y_arr[i - 1], y_arr[i])

    return ts_y


def problem2_2():
    # Develop Mamdani fuzzy inference system for the following problem:
    print("Part 2.2")
    n_fcns = 7
    x_min = -10
    x_max = 10
    x_freq = np.linspace(-10, 10, 2001, dtype=np.float64)
    y_true = x_freq**2
    y_defuzzy = np.zeros_like(x_freq, dtype=np.float64)

    # TODO - Vectorize this!
    for ij, x in enumerate(x_freq):
        mu_x = mu_prob2_2(x, n_fcns, x_min, x_max)
        # The generated membership functions match to output, so no need for rules, ha.
        mu_rule = np.copy(mu_x)
        mu_all_rules = mu_rule
        y_fcns = takagi_sugeno_functions(x, n_fcns, x_min, x_max)
        y_defuzzy[ij] = np.sum(y_fcns * mu_all_rules) / np.sum(mu_all_rules)

    # Create subplots with 1 row and 3 columns
    fig = go.Figure()

    # Add traces for the overlay of y and y = x^2 in the third column
    fig.add_trace(
        go.Scatter(x=x_freq, y=y_defuzzy, mode="markers", name="y-fuzzy"),
    )
    fig.add_trace(go.Scatter(x=x_freq, y=y_true, mode="lines", name="y = x^2"))

    # Calculate RMS error between `y` and `y = x^2`
    rms_error = np.sqrt(np.mean((np.array(y_true) - np.array(y_defuzzy)) ** 2))
    fig.add_annotation(
        text=f"RMS Error: {rms_error:.4f}",
        x=2,
        y=50,
        showarrow=False,
        align="center",
    )

    # Update layout
    fig.update_layout(title="Problem2.2", showlegend=True)

    # Show the plot
    fig.show()


def main():
    problem2_1()
    problem2_2()


if __name__ == "__main__":
    main()
