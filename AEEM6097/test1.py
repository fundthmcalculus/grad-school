import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from AEEM6097.aco_solver import solve_aco, AcoContinuousVariable
from AEEM6097.fuzzy import FuzzySet, MamdaniRule, FuzzyVariable, FuzzyInference
from AEEM6097.membership_functions import (
    create_triangle_memberships,
)


def plot_membership_functions(*varargs: FuzzySet) -> None:
    # Create the figure
    fig = make_subplots(
        rows=len(varargs),
        cols=1,
        subplot_titles=[
            f"Membership Functions of {fuzzy_set.var_name}" for fuzzy_set in varargs
        ],
    )
    for i_plot, fuzzy_set in enumerate(varargs):
        print(f"Plotting set: {fuzzy_set}")
        # Iterate through each membership function
        for mf in fuzzy_set.membership_functions:
            # Get the domain of the membership function
            domain = mf.domain()
            x = np.linspace(domain[0], domain[1], 1000)
            y = mf.mu(x)

            # Add the membership function to the plot
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=mf.name), row=i_plot + 1, col=1
            )

    # Show the plot
    fig.show()


def fuzzy_system_aco(x: np.typing.NDArray) -> np.float64:
    try:
        # Destructure the input into the arguments - flow, water level, power (in that order)
        all_rules, _, _, _ = create_system(*x)

        # Iterate through all possible steps and evaluate the rules!
        X, Y, Z, Z_ref, rms_err = run_default_simulation(all_rules)

        return rms_err
    except:
        return 1e6


def main():
    print("Test 1")
    # TODO - Use Pi Membership Functions for practice with Dr Cohen's symphony!
    # Input variables that can be optimized!
    flow_0 = 0
    flow_1 = 1000
    flow_2 = 2000
    flow_3 = 2700
    flow_4 = 4000

    level_0 = 0
    level_1 = 40
    level_2 = 80
    level_3 = 120
    level_4 = 150

    power_0 = 20
    power_1 = 40
    power_2 = 50
    power_3 = 90
    power_4 = 110

    # Disabled for simplicity.
    # flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0, power_1, power_2, power_3, power_4 = optimize_system(
    #     flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0, power_1, power_2,
    #     power_3, power_4)

    all_rules, flow_rate, water_level, power_output = create_system(flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4,
                              power_0, power_1, power_2, power_3, power_4)
    plot_membership_functions(flow_rate, water_level, power_output)

    # Iterate through all possible steps and evaluate the rules!
    X, Y, Z, Z_ref, rms_err = run_default_simulation(all_rules)

    fig = make_subplots(2,1,subplot_titles=["Fuzzy System Surface Plot", f"Normalized RMS Error: {rms_err:.4f}"],
                        specs=[[{'type': 'surface'}], [{'type': 'surface'}]],
                        )
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, name="Fuzzy System"), row=1, col=1, )
    fig.layout.scene.xaxis.title = "Flow Rate"
    fig.layout.scene.yaxis.title = "Water Level"
    fig.layout.scene.zaxis.title = "Power Output"
    fig.layout.scene.camera = dict(eye=dict(x=-1.5, y=-1.5, z=1.0))

    fig.add_trace(go.Surface(x=X, y=Y, z=Z_ref, name="Reference System"),row=2, col=1, )
    fig.layout.scene2.xaxis.title = "Flow Rate"
    fig.layout.scene2.yaxis.title = "Water Level"
    fig.layout.scene2.zaxis.title = "Power Output"
    fig.layout.scene2.camera = dict(eye=dict(x=-1.5, y=-1.5, z=1.0))

    # Add a color bar
    fig.update_traces(
        showscale=True, colorbar=dict(title="Power Output", thickness=20, len=0.7)
    )

    # Show the plot
    fig.show()


def optimize_system(flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0,
                    power_1, power_2, power_3, power_4):
    # Use the ACO solver to optimize the membership functions!
    best_soln, soln_history = solve_aco(fuzzy_system_aco, [
        AcoContinuousVariable("flow_0", -200, 4200, flow_0),
        AcoContinuousVariable("flow_1", -200, 4200, flow_1),
        AcoContinuousVariable("flow_2", -200, 4200, flow_2),
        AcoContinuousVariable("flow_3", -200, 4200, flow_3),
        AcoContinuousVariable("flow_4", -200, 4200, flow_4),

        AcoContinuousVariable("level_0", -50, 200, level_0),
        AcoContinuousVariable("level_1", -50, 200, level_1),
        AcoContinuousVariable("level_2", -50, 200, level_2),
        AcoContinuousVariable("level_3", -50, 200, level_3),
        AcoContinuousVariable("level_4", -50, 200, level_4),

        AcoContinuousVariable("power_0", 0, 120, power_0),
        AcoContinuousVariable("power_1", 0, 120, power_1),
        AcoContinuousVariable("power_2", 0, 120, power_2),
        AcoContinuousVariable("power_3", 0, 120, power_3),
        AcoContinuousVariable("power_4", 0, 120, power_4),
    ])
    print(f"Solution history: {soln_history}")
    print(f"Best solution: {best_soln} with value: {soln_history[-1]}")
    # Plot solution history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(soln_history)), y=soln_history, mode="lines", name="RMS Error"))
    fig.update_layout(title="RMS Error vs. Generation", xaxis_title="Generation", yaxis_title="RMS Error")
    fig.show()
    # Destructure the best solution
    flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0, power_1, power_2, power_3, power_4 = best_soln
    return flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0, power_1, power_2, power_3, power_4


def run_default_simulation(all_rules):
    N_steps = 15
    l_max = 150
    l_min = 0
    q_max = 4000
    q_min = 0
    q_steps = np.linspace(q_min, q_max, N_steps)
    l_steps = np.linspace(l_min, l_max, N_steps)
    X, Y = np.meshgrid(q_steps, l_steps)
    Z = simulate_system(all_rules, l_steps, q_steps)
    Z_ref, rms_err = get_rms_err(Z)
    return X, Y, Z, Z_ref, rms_err


def create_system(flow_0, flow_1, flow_2, flow_3, flow_4, level_0, level_1, level_2, level_3, level_4, power_0, power_1,
                  power_2, power_3, power_4):
    # Create the relevant membership functions
    flow_rate: FuzzySet = FuzzySet(
        "Flow Rate",
        create_triangle_memberships(
            {
                "Very Low": flow_0,
                "Low": flow_1,
                "Medium": flow_2,
                "High": flow_3,
                "Very High": flow_4,
            }
        ),
    )
    # Do the same for water level
    water_level: FuzzySet = FuzzySet(
        "Water Level",
        create_triangle_memberships(
            {
                "Almost Empty": level_0,
                "Low": level_1,
                "Medium": level_2,
                "High": level_3,
                "Almost Full": level_4,
            }
        ),
    )
    # Now we define the output variable
    power_output: FuzzySet = FuzzySet(
        "Power Output",
        create_triangle_memberships(
            {
                "Very Low": power_0,
                "Low": power_1,
                "Medium": power_2,
                "High": power_3,
                "Very High": power_4,
            }
        ),
    )
    # Now we define the rules!
    all_rules = [
        # Rule: IF flow rate very low or low and water level = almost empty, low, medium, then power output = very low
        MamdaniRule(
            "Rule 1",
            (flow_rate == ["Very Low", "Low"])
            & (water_level == ["Almost Empty", "Low", "Medium", "High"]),
            power_output,
            "Very Low",
        ),
        # Rule: IF flow rate very low or low and water level = almost full, then power output = medium
        MamdaniRule(
            "Rule 3",
            (flow_rate == ["Very Low", "Low"]) & (water_level == ["Almost Full"]),
            power_output,
            "Medium",
        ),
        # Rule: IF flow rate medium and water level = almost empty, low, medium, then power output = low
        MamdaniRule(
            "Rule 4",
            (flow_rate == ["Medium"])
            & (water_level == ["Almost Empty", "Low", "Medium"]),
            power_output,
            "Low",
        ),
        # Rule: IF flow rate medium and water level = almost full, then power output = high
        MamdaniRule(
            "Rule 5",
            (flow_rate == ["Medium"]) & (water_level == ["High"]),
            power_output,
            "Medium",
        ),
        # Rule: IF flow rate high or very high and water level = almost empty, low, medium, then power output = medium
        MamdaniRule(
            "Rule 6",
            (flow_rate == ["High", "Very High"])
            & (water_level == ["Almost Empty", "Low", "Medium"]),
            power_output,
            "Medium",
        ),
        # Rule: IF flow rate high or very high and water level = almost full, then power output = very high
        MamdaniRule(
            "Rule 7",
            (flow_rate == ["Medium", "High", "Very High"]) & (water_level == ["Almost Full"]),
            power_output,
            "Very High",
        ),
    ]
    return all_rules, flow_rate, water_level, power_output


def simulate_system(all_rules, l_steps, q_steps):
    Z = np.zeros((len(l_steps), len(q_steps)))
    for iy, q_i in enumerate(q_steps):
        for ix, l_j in enumerate(l_steps):
            q_v = FuzzyVariable("Flow Rate", q_i)
            l_v = FuzzyVariable("Water Level", l_j)
            fuzzy_vars = [q_v, l_v]
            rule_output: list[FuzzyInference] = [r(fuzzy_vars) for r in all_rules]
            # TODO - Handle TKS defuzzification!
            rule_result = [
                ro.output_set[ro.var_name].centroid() * ro.value for ro in rule_output
            ]
            mu_sum = np.sum([ro.value for ro in rule_output])
            if mu_sum == 0:
                # No rules fired, so we can't defuzzify
                logging.warning(f"No rules fired for {q_v}, {l_v}")
                Z[ix, iy] = 0
                continue
            defuzzified = np.sum(rule_result) / mu_sum
            # TODO - Handle different output variables!
            Z[ix, iy] = defuzzified
    return Z


def get_rms_err(Z):
    # Show the reference diagram.
    Z_ref = get_reference_surface()
    # Add a comment on the normalized RMS error
    rms_err = np.sqrt(np.sum((Z - Z_ref) ** 2))
    return Z_ref, rms_err

Z_ref = np.array(
    [
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 50, 50],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 50, 50],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 50, 50],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 50, 50],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 55, 60, 60],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 40, 70, 80, 80],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 40, 45, 75, 93, 93],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 60, 50, 80, 110, 110],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 65, 65, 80, 105, 105],
        [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 70, 80, 80, 105, 105],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 95, 90, 110, 110],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 110, 100, 110, 110],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 110, 100, 110, 110],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 110, 100, 110, 110],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 75, 110, 100, 110, 110],
    ]
).transpose()

def get_reference_surface():
    return Z_ref


if __name__ == "__main__":
    main()
