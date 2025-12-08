import numpy as np
import matplotlib.pyplot as plt


# Control function
def x_k1_fcn(a, b, dt, x_k, u_k):
    return (1+a*dt)*x_k + b*dt*u_k

# Performance measure
def J_perf(x_N, l, dt, u_k):
    # Exclude the final control value.
    return x_N**2.0+l*dt*np.sum(u_k**2.0)


def problem(a,b,l,N,dt):
    admit_pts = integrate_fcn(a, b, dt, l, N)
    plot_J_grid(admit_pts, f"a={a}, lambda={l}, N={N}")

def p314():
    # a
    problem(0.0,1.0,2.0,2,1.0)
    problem(0.0,1.0,2.0,3,1.0)
    problem(0.0,1.0,4.0,2,1.0)
    problem(0.0,1.0,0.5,2,1.0)

def p316():
    # a
    problem(-0.4,1.0,2.0,2,1.0)
    problem(-0.4,1.0,2.0,3,1.0)
    problem(-0.4,1.0,4.0,2,1.0)
    problem(-0.4,1.0,0.5,2,1.0)


def integrate_fcn(a, b, dt, l, n_stages):
    # x_k grid values - inclusive endpoints
    x_kgrid = np.r_[0:1.501:0.02]
    u_kgrid = np.r_[-1.00:1.001:0.02]

    admit_pts = []
    # Try each initial condition
    for k_x, x_k in enumerate(x_kgrid):
        for k_u, u_k in enumerate(u_kgrid):
            # Stepwise integrate
            x = np.zeros(n_stages + 1)
            u = np.zeros(n_stages)
            # Initialize the first stage
            x[0] = x_k
            u[0] = u_k
            # Integrate
            admissable = True
            minJ = np.inf
            for stage in range(1,n_stages):
                x_tst = x_k1_fcn(a, b, dt, x[stage - 1], u[stage - 1])
                if 0.0 <= x_tst <= 1.5:
                    x[stage] = x_tst
                    minJ_step = np.inf
                    min_u = 0.0
                    for k_u2, u_k2 in enumerate(u_kgrid):
                        u[stage] = u_k2
                        x_tst = x_k1_fcn(a, b, dt, x[stage], u[stage])
                        if 0.0 <= x_tst <= 1.5:
                            J_tst = J_perf(x_tst, l, dt, u[:stage])
                            if J_tst < minJ_step:
                                minJ_step = J_tst
                                min_u = u_k2
                    u[stage] = min_u
                else:
                    admissable = False
                    break
            if admissable:
                admit_pts.append([x_k, u_k, J_perf(x[-1], l, dt, u)])

    # Plot the results as a function of x and u
    return admit_pts


def plot_J_grid(admit_pts, details_str):
    # Extract data points
    x_vals = [pt[0] for pt in admit_pts]
    u_vals = [pt[1] for pt in admit_pts]
    J_vals = [pt[2] for pt in admit_pts]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot state vs control
    ax1.scatter(x_vals, u_vals, c='blue', alpha=0.5)
    ax1.set_xlabel('State (x)')
    ax1.set_ylabel('Control (u)')
    ax1.set_title(f'State vs Control\n{details_str}')
    ax1.grid(True)

    # Plot state vs cost
    ax2.scatter(x_vals, J_vals, c='red', alpha=0.5)
    ax2.set_xlabel('State (x)')
    ax2.set_ylabel('Cost (J)')
    ax2.set_title(f'State vs Cost\n{details_str}')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p314()
    p316()