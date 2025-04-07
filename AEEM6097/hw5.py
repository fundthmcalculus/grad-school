import plotly.graph_objects as go
import numpy as np

from AEEM6097.aco_solver import solve_aco, AcoContinuousVariable


def f(x: np.complex128) -> np.float64:
    f_x = x**5+7*x**4+6*x**3-4*x**2-3*x+2
    return np.abs(f_x)**2


def aco_f(x: np.typing.NDArray[np.float64]) -> np.float64:
    # Each entry in the array is a coefficient zero, account for complex-conjugate pairs.
    soln_val = 0.0
    soln_val += f(x[0])
    soln_val += f(np.complex128(x[1],x[2]))
    soln_val += f(np.complex128(x[1],-x[2]))
    soln_val += f(np.complex128(x[3],x[4]))
    soln_val += f(np.complex128(x[3],-x[4]))
    # Convert complex error into magnitude error.
    return soln_val



def main():
    print("Homework 5")
    # TODO - We can get a better idea from a plot
    # Set up the ACO system - give decent initial guesses, or the system will take a long time to converge and find duplicate roots.
    aco_root_vars = [AcoContinuousVariable(f"real_{0}", -6, -4),
                     AcoContinuousVariable(f"real_{1}", -2, 0),
                     AcoContinuousVariable(f"imag_{1}", 0, 0.5),
                     AcoContinuousVariable(f"real_{2}", 0, 1),
                     AcoContinuousVariable(f"imag_{2}", 0, 0.5)]
    # Use a little algebra knowledge. order-5 polynomial has at least 1 real root. The others are complex conjugate pairs.
    best_soln, soln_history = solve_aco(
        aco_f,
        aco_root_vars,
        num_generations=100,
        num_ants=25,
        solution_archive_size=50,
        joblib_n_procs=1,
    )
    print(f"Best solution: {best_soln} with value: {soln_history[-1]}")
    print(f"Real root: {best_soln[0]:.4f}")
    print(f"Complex root 1: {best_soln[1]+best_soln[2]*1j:.4f}")
    print(f"Complex root 2: {best_soln[1]-best_soln[2]*1j:.4f}")
    print(f"Complex root 3: {best_soln[3]+best_soln[4]*1j:.4f}")
    print(f"Complex root 4: {best_soln[3]-best_soln[4]*1j:.4f}")
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
    fig.update_yaxes(type="log")
    fig.show()




if __name__ == "__main__":
    main()
