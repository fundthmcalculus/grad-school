"""The neural reference search space, shared by flm.match_mults and flm.grids_v4 (no imports,
so regenerating matched_mults.json never needs the file it produces)."""

# neural reference candidates (MLP FFN). Softmax uses RoPE, which needs an even head
# dimension, so it only gets 4 heads (d / 4 is even at every width used).
NEURAL = {
    "delta-h4": ["--mixer", "delta", "--decay", "data", "--ffn", "mlp", "--heads", "4"],
    "delta-h8": ["--mixer", "delta", "--decay", "data", "--ffn", "mlp", "--heads", "8"],
    "gla-h4": ["--mixer", "linear", "--decay", "data", "--ffn", "mlp", "--heads", "4"],
    "gla-h8": ["--mixer", "linear", "--decay", "data", "--ffn", "mlp", "--heads", "8"],
    "softmax-h4": ["--mixer", "softmax", "--ffn", "mlp", "--heads", "4"],
}
WIDTHS = (40, 56, 72, 104)
SHORTCONVS = (4, 8, 16)
REF_LRS = ("2e-2", "3e-2", "4e-2", "6e-2")
