import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_all_the_things(x,y,Z_true,Z_approx: np.ndarray, x_mu: np.ndarray, y_mu: np.ndarray):
    # Create a 2x2 subplot grid with 3D subplots in the top row
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'xy'}],[{'type': 'surface'}, {'type': 'surface'}],
               ],
        subplot_titles=('Membership Functions-X', 'Membership Functions-Y',
                        'True Function', 'Approximated Function')
    )

    # Sample 2D functions - replace with your true and approximated functions
    x_2d = x[0, :].flatten()
    y_2d = y[:, 0].flatten()

    # Add 2D line plots to the top row
    for idx_mf in range(x_mu.shape[1]):
        fig.add_trace(
            go.Scatter(x=x_2d, y=x_mu[:,idx_mf], mode='lines', name=f'X Membership Function-{idx_mf}', line=dict(color='blue', width=2)),
            row=1, col=1
        )
    for idx_mf in range(y_mu.shape[1]):
        fig.add_trace(
            go.Scatter(x=y_2d, y=y_mu[:,idx_mf], mode='lines', name=f'Y Membership Function-{idx_mf}', line=dict(color='red', width=2)),
            row=1, col=2
        )

    # Add 3D surface plots to the bottom row
    fig.add_trace(
        go.Surface(z=Z_true, x=x, y=y, colorscale='Viridis', showscale=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Surface(z=Z_approx, x=x, y=y, colorscale='Plasma', showscale=False),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text='Membership Functions and Function Approximation',
        height=800,
        width=1000,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Function Value'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Approximated Value'
        )
    )

    # Update 2D plot axes
    fig.update_xaxes(title_text='X', row=2, col=1)
    fig.update_yaxes(title_text='Y', row=2, col=1)
    fig.update_xaxes(title_text='X', row=2, col=2)
    fig.update_yaxes(title_text='Y', row=2, col=2)

    # Show the plot
    fig.show()


def f_true(x,y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y)


def mu_poly(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + ((x - a) / b) ** 2)

def mu_poly_set(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    mu_x = np.zeros((len(x),len(p)))
    for i in range(len(p)):
        if i < len(p) - 1:
            mu_x[:,i] = mu_poly(x, p[i], np.diff(p[i:i+2])/2.0)
        else:
            mu_x[:,i] = mu_poly(x, p[i], np.diff(p[i-1:i+1])/2.0)
    return mu_x


def tsk_rule(s: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    # Zeroth order TSK, or Mamdani
    if len(coeff) == 1:
        return coeff[0]
    first_order_size = s.shape[1] + 1
    if len(coeff) == first_order_size:
        X = np.ones((first_order_size, len(s)))
        X[1:,:] = s[:,:]
        return np.dot(X, coeff)
    # TODO - 2nd order TSK?
    else:
        raise ValueError(f"TSK rule requires {first_order_size} coefficients")

# TODO - Generate derivatives as well?
def fuzzy_or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y - x * y
def fuzzy_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y

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
    s = pts[:,0:2]
    # Create the default membership functions
    n_mus = 2 # Per variable
    mu_xp = np.linspace(s_min, s_max, n_mus)
    mu_yp = np.linspace(s_min, s_max, n_mus)
    mu_x = mu_poly_set(pts[:,0], mu_xp)
    mu_y = mu_poly_set(pts[:,1], mu_yp)
    # Create the fuzzy rules, 0th order TSK
    n_rules = 4 # TODO - Change this?
    # Rule 1, X is small and Y is small
    rule1 = np.array([0,RULE_AND,0,np.min(Z)])
    rule2 = np.array([0,RULE_AND,1,np.max(Z)])
    rule3 = np.array([1,RULE_AND,0,np.max(Z)])
    rule4 = np.array([1,RULE_AND,1,np.min(Z)])
    r1_eval, z1_eval = eval_rule(rule1, mu_x, mu_y, s)
    r2_eval, z2_eval = eval_rule(rule2, mu_x, mu_y, s)
    r3_eval, z3_eval = eval_rule(rule3, mu_x, mu_y, s)
    r4_eval, z4_eval = eval_rule(rule4, mu_x, mu_y, s)
    # Evaluate the model on the entire dataset and plot it
    z_arr = np.array([z1_eval, z2_eval, z3_eval, z4_eval])
    r_arr = np.array([r1_eval, r2_eval, r3_eval, r4_eval])
    Z_defuzzy = np.dot(r_arr.T, z_arr) / np.sum(r_arr,axis=0)
    # Compute the RMS error
    rms_error = np.sqrt(np.mean((Z_defuzzy - pts[:,2]) ** 2))
    print(f"RMS error: {rms_error}")
    # Reshape into the 2D array
    Z_defuzzy = Z_defuzzy.reshape(X.shape)
    # Randomly reorder the points
    np.random.shuffle(pts)
    # Get the training dataset, and the testing dataset
    n_train = 1280
    pts_train = pts[:n_train, :]
    pts_test = pts[n_train:, :]
    # TODO - Train the fuzzy model!

    # Plot using plotly
    plot_all_the_things(X,Y,Z,Z_defuzzy, mu_x, mu_y)



if __name__ == '__main__':
    main()