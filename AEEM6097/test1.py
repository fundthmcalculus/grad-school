import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from AEEM6097.fuzzy import FuzzySet, MamdaniRule, FuzzyVariable, FuzzyInference
from AEEM6097.membership_functions import (
    create_uniform_triangle_memberships,
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


def main():
    print("Test 1")
    # Create the relevant membership functions
    # Use Pi Membership Functions for practice with Dr Cohen's symphony!
    b_factor = 2.0
    N_fcns = 5
    q_max = 4000
    q_min = 0
    q_spacing = (q_max - q_min) / N_fcns
    q_width = q_spacing / b_factor
    q_first = q_spacing / 2.0
    flow_rate: FuzzySet = FuzzySet(
        "Flow Rate",
        create_triangle_memberships(
            {
                "Very Low": 0,
                "Low": 1000,
                "Medium": 2000,
                "High": 2700,
                "Very High": 4000,
            }
        ),
        # [
        #     PiMF("Very Low", q_first, q_width),
        #     PiMF("Low", q_first + q_spacing, q_width),
        #     PiMF("Medium", q_first + q_spacing * 2, q_width),
        #     PiMF("High", q_first + q_spacing * 3, q_width),
        #     PiMF(
        #         "Very High", q_first + q_spacing * 4, q_width
        #     ),  # NOTE - Maybe should be offset more?
        # ],
    )

    # Do the same for water level
    l_max = 150
    l_min = 0
    l_spacing = (l_max - l_min) / N_fcns
    l_width = l_spacing / b_factor
    l_first = l_spacing / 2.0
    water_level: FuzzySet = FuzzySet(
        "Water Level",
        create_triangle_memberships({
            "Almost Empty": 10,
            "Low": 50,
            "Medium": 90,
            "High": 120,
            "Almost Full": 140,
        }),
        # [
        #     PiMF("Almost Empty", l_first, l_width),
        #     PiMF("Low", l_first + l_spacing, l_width),
        #     PiMF("Medium", l_first + l_spacing * 2, l_width),
        #     PiMF("High", l_first + l_spacing * 3, l_width),
        #     PiMF("Almost Full", l_first + l_spacing * 4, l_width),
        # ],
    )

    # Now we define the output variable
    power_output: FuzzySet = FuzzySet(
        "Power Output",
        create_triangle_memberships(
            {
                "Very Low": 20,
                "Low": 40,
                "Medium": 50,
                "High": 80,
                "Very High": 110,
            }
        ),
        # [
        #     PiMF("Very Low", 20, 10),
        #     PiMF("Low", 40, 10),
        #     PiMF("Medium", 60, 10),
        #     PiMF("High", 80, 10),
        #     PiMF("Very High", 100, 10),
        # ],
    )
    plot_membership_functions(flow_rate, water_level, power_output)

    # Now we define the rules!
    all_rules = [
        # Rule: IF flow rate very low or low and water level = almost empty, low, medium, then power output = very low
        # Rule: IF flow rate very low or low and water level = high, then power output = low
        # Rule: IF flow rate very low or low and water level = almost full, then power output = medium
        # Rule: IF flow rate medium and water level = almost empty, low, medium, then power output = low
        # Rule: IF flow rate medium and water level = almost full, then power output = high
        # Rule: IF flow rate high or very high and water level = almost empty, low, medium, then power output = medium
        # Rule: IF flow rate high or very high and water level = almost full, then power output = very high
        # Rule: IF flow rate very low or low and water level = almost empty, low, medium, then power output = very low
        MamdaniRule(
            "Rule 1",
            (flow_rate == ["Very Low", "Low"])
            & (water_level != ["Almost Full"]),
            power_output,
            "Very Low",
        ),
        # Rule: IF flow rate very low or low and water level = almost full, then power output = medium
        MamdaniRule(
            "Rule 3",
            (flow_rate == ["Very Low", "Low"]) & (water_level == "Almost Full"),
            power_output,
            "Medium",
        ),
        # Rule: IF flow rate medium and water level = almost empty, low, medium, then power output = low
        MamdaniRule(
            "Rule 4",
            (flow_rate == ["Medium"])
            & (water_level == ["Almost Empty", "Low", "Medium", "High"]),
            power_output,
            "Low",
        ),
        # Rule: IF flow rate medium and water level = almost full, then power output = high
        MamdaniRule(
            "Rule 5",
            (flow_rate == ["Medium"]) & (water_level == ["Almost Full"]),
            power_output,
            "High",
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
            (flow_rate == ["High", "Very High"]) & (water_level == ["Almost Full"]),
            power_output,
            "Very High",
        ),
        # Rule: IF flow rate high or very high and water level = almost full, then power output = very high
        MamdaniRule(
            "Rule 7",
            (flow_rate == ["High", "Very High"]) & (water_level == ["High"]),
            power_output,
            "Very High",
        ),
    ]

    # Iterate through all possible steps and evaluate the rules!
    N_steps = 15
    q_steps = np.linspace(q_min, q_max, N_steps)
    l_steps = np.linspace(l_min, l_max, N_steps)
    Z = np.zeros((N_steps, N_steps))
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

    X, Y = np.meshgrid(q_steps, l_steps)

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

    # Update the layout with title and axis labels
    fig.update_layout(
        title="3D Surface Plot",
        scene=dict(
            xaxis_title="Flow Rate",
            yaxis_title="Water Level",
            zaxis_title="Power Output",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)),  # Adjust camera view
        ),
        width=800,
        height=800,
    )

    # Customize the color scale (optional)
    fig.update_traces(colorscale="viridis")

    # Add a color bar
    fig.update_traces(
        showscale=True, colorbar=dict(title="Z value", thickness=20, len=0.7)
    )

    # Show the plot
    fig.show()

    # Show the reference diagram.
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
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_ref)])

    # Update the layout with title and axis labels
    fig.update_layout(
        title="Reference 3D Surface Plot",
        scene=dict(
            xaxis_title="Flow Rate",
            yaxis_title="Water Level",
            zaxis_title="Power Output",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)),  # Adjust camera view
        ),
        width=800,
        height=800,
    )

    # Customize the color scale (optional)
    fig.update_traces(colorscale="viridis")

    # Add a color bar
    fig.update_traces(
        showscale=True, colorbar=dict(title="Z value", thickness=20, len=0.7)
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    main()
