from AEEM6097.fuzzy import FuzzySet, FuzzyRule
from AEEM6097.membership_functions import PiMF


def main():
    print("Test 1")
    # Create the relevant membership functions
    # Use Pi Membership Functions for practice with Dr Cohen's symphony!
    N_fcns = 5.0
    q_spacing = (4000 - 0) / N_fcns
    q_width = q_spacing/4.0
    q_first = q_spacing / 2.0
    flow_rate: FuzzySet = FuzzySet(
        "Flow Rate",
        [
            PiMF("Very Low", q_first, q_width),
            PiMF("Low", q_first + q_spacing, q_width),
            PiMF("Medium", q_first + q_spacing * 2, q_width),
            PiMF("High", q_first + q_spacing * 3, q_width),
            PiMF("Maximum", q_first + q_spacing * 4, q_width), # NOTE - Maybe should be offset more?
        ],
    )

    print(flow_rate)

    # Do the same for water level
    l_spacing = (150-0)/N_fcns
    l_width = l_spacing / 4.0
    l_first = l_spacing / 2.0
    water_level: FuzzySet = FuzzySet(
        "Water Level",
        [
            PiMF("Almost Empty", l_first, l_width),
            PiMF("Low", l_first + l_spacing, l_width),
            PiMF("Medium", l_first + l_spacing * 2, l_width),
            PiMF("High", l_first + l_spacing * 3, l_width),
            PiMF("Almost Full", l_first + l_spacing * 4, l_width),
        ],
    )

    print(water_level)

    # Now we define the output variable
    power_output: FuzzySet = FuzzySet(
        "Power Output",
        [
            PiMF("Very Low", 20, 10),
            PiMF("Low", 40, 10),
            PiMF("Medium", 60, 10),
            PiMF("High", 80, 10),
            PiMF("Very High", 100, 10),
        ],
    )
    print(power_output)

    # Now we define the rules!
    # Rule 1: IF flow rate is very low AND water level is not almost full THEN power output is very low
    rule_1 = FuzzyRule(
        antecedent=flow_rate  & water_level["Almost Full"],
        consequent=power_output["Very Low"],
    )
    print(rule_1)


if __name__ == "__main__":
    main()
