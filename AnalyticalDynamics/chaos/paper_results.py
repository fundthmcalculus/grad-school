"""Reported results from arXiv:2504.13453, taken at face value.

Transcribed from the bar-chart data labels of Figs. 11, 12, 13, 18B, 18C, 18D
and the heatmaps of Fig. 22. RMSE is in the paper's scaled units: each
trajectory is min-max normalised to [0, 1] per angle column before pooling, so
an RMSE of 0.027 is 2.7% of that trajectory's own angular range. See
METHOD_AND_PARAMETERS.md section 3.

Where the paper quotes a value twice and the two disagree, both are noted in the
comment and the figure's data label is used, since that is what the code
produced.

Keys are (system, friction, setting):
  system   : "double" | "triple"
  friction : False | True
  setting  : "trained" (an IC that was in the training grid: [120, 0(, 0)])
             "holdout" (the in-between IC [120, 2.05] / [120, 0, 2.05])
Values map model name -> (rmse, r2), or None where the paper did not run it.
"""

#: Display order, and the order the paper's own bar charts use.
MODEL_ORDER = ["LSTM", "FFNN", "AR", "VRNN", "GRU", "BIRNN", "SRNN", "MLP"]

RESULTS = {
    # --- Double pendulum, frictionless (paper Fig. 11) ---
    ("double", False, "trained"): {
        "LSTM": (2.701e-2, 0.991527),
        "FFNN": (6.010e-2, 0.963700),
        "AR": (1.415e-1, 0.769966),
        "VRNN": (4.073e-2, 0.981180),
        "GRU": (3.838e-2, 0.982088),
        "BIRNN": (4.015e-2, 0.980485),
        "SRNN": (7.760e-2, 0.927200),
        "MLP": (7.547e-2, 0.938207),
    },
    # Only the LSTM was run here. Section 3.2 gives RMSE 0.26 / R^2 0.23; the
    # section 4 conclusion gives 3.1e-1 for the same cell. Section 3.2 is the
    # narrative that reports the experiment, so it is used.
    ("double", False, "holdout"): {
        "LSTM": (0.26, 0.23),
        "FFNN": None, "AR": None, "VRNN": None, "GRU": None,
        "BIRNN": None, "SRNN": None, "MLP": None,
    },
    # --- Double pendulum, friction (paper Fig. 12) ---
    ("double", True, "trained"): {
        "LSTM": (9.546e-3, 0.998731),
        "FFNN": (4.353e-2, 0.972693),
        "AR": (1.147e-1, 0.613597),
        "VRNN": (1.498e-2, 0.996860),
        "GRU": (1.496e-2, 0.996853),
        "BIRNN": (2.151e-2, 0.993485),
        "SRNN": (3.890e-2, 0.978743),
        "MLP": (2.356e-2, 0.992257),
    },
    # --- Double pendulum, friction, unknown IC [120, 2.05] (paper Fig. 13) ---
    ("double", True, "holdout"): {
        "LSTM": (1.529e-2, 0.996431),
        "FFNN": (3.136e-2, 0.985698),
        "AR": (7.868e-2, 0.919674),
        "VRNN": (1.663e-2, 0.995972),
        "GRU": (1.813e-2, 0.995240),
        "BIRNN": (1.791e-2, 0.995323),
        "SRNN": (2.816e-2, 0.987623),
        "MLP": (2.356e-2, 0.992594),
    },
    # --- Triple pendulum, frictionless (paper Fig. 18B) ---
    # The paper's own bar labels for this panel are inconsistent between the
    # RMSE axis and the Fig. 22 heatmap (e.g. GRU 9.85e-3 vs 0.017). The
    # heatmap is the paper's summary artefact and is internally consistent, so
    # RMSE comes from Fig. 22 and R^2 from the Fig. 18B labels.
    ("triple", False, "trained"): {
        "LSTM": (2.2e-2, 0.982643),
        "FFNN": (1.1e-1, 0.804925),
        "AR": (1.5e-1, 0.699268),
        "VRNN": (8.6e-2, 0.890089),
        "GRU": (1.7e-2, 0.985333),
        "BIRNN": (6.0e-2, 0.942589),
        "SRNN": (1.1e-1, 0.927063),
        "MLP": (8.4e-2, 0.889062),
    },
    # --- Triple pendulum, friction (paper Fig. 18C) ---
    ("triple", True, "trained"): {
        "LSTM": (1.0019e-2, 0.997459),
        "FFNN": (1.8070e-2, 0.992468),
        "AR": (9.6546e-2, 0.795768),
        "VRNN": (2.4355e-2, 0.987377),
        "GRU": (9.1125e-3, 0.998233),
        "BIRNN": (1.7299e-2, 0.993719),
        "SRNN": (2.3769e-2, 0.986479),
        "MLP": (4.4068e-2, 0.970511),
    },
    # --- Triple pendulum, friction, unknown IC [120, 0, 2.05] (Fig. 18D) ---
    ("triple", True, "holdout"): {
        "LSTM": (8.3652e-3, 0.998542),
        "FFNN": (2.4366e-2, 0.987363),
        "AR": (9.0742e-2, 0.812740),
        "VRNN": (2.1210e-2, 0.990385),
        "GRU": (6.4975e-3, 0.999093),
        "BIRNN": (1.8216e-2, 0.992288),
        "SRNN": (2.9657e-2, 0.981304),
        "MLP": (1.6545e-2, 0.994005),
    },
    # The paper never ran a frictionless triple-pendulum holdout: it abandoned
    # frictionless holdout testing after the double-pendulum LSTM result.
    ("triple", False, "holdout"): {m: None for m in MODEL_ORDER},
}


def best(system, friction, setting, by="rmse"):
    """(model, rmse, r2) for the paper's best model in a cell, or None."""
    cell = {k: v for k, v in RESULTS[(system, friction, setting)].items() if v}
    if not cell:
        return None
    idx = 0 if by == "rmse" else 1
    name = (min if by == "rmse" else max)(cell, key=lambda m: cell[m][idx])
    return (name,) + cell[name]


def as_rows(system, friction, setting):
    """[(model, rmse|None, r2|None)] in the paper's display order."""
    cell = RESULTS[(system, friction, setting)]
    return [(m, *(cell.get(m) or (None, None))) for m in MODEL_ORDER]
