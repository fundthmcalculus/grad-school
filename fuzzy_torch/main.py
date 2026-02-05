
import torch
import torch.nn as nn

class FuzzificationLayer(nn.Module):
    def __init__(self, n_inputs, n_memberships):
        super(FuzzificationLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        # Initialize means and sigmas for Gaussian membership functions
        # Shape: (n_inputs, n_memberships)
        self.means = nn.Parameter(torch.randn(n_inputs, n_memberships))
        self.sigmas = nn.Parameter(torch.ones(n_inputs, n_memberships))

    def forward(self, x):
        # x shape: (batch_size, n_inputs)
        # Reshape x to (batch_size, n_inputs, 1)
        x = x.unsqueeze(-1)
        # Compute Gaussian membership: exp(-0.5 * ((x - mean) / sigma)^2)
        # output shape: (batch_size, n_inputs, n_memberships)
        return torch.exp(-0.5 * torch.pow((x - self.means) / self.sigmas, 2))

class FuzzyLogicLayer(nn.Module):
    def __init__(self, n_inputs, n_memberships, n_rules):
        super(FuzzyLogicLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_rules = n_rules
        # Rule antecedents: which membership function of which input is used in each rule
        # For simplicity, let's use a soft selection / weight matrix for now
        # or a fixed mapping if we want to follow a grid of rules.
        # Given n_rules, we can have a learnable mapping.
        # Shape: (n_rules, n_inputs, n_memberships)
        self.rule_weights = nn.Parameter(torch.randn(n_rules, n_inputs, n_memberships))

    def forward(self, fuzzified_x):
        # fuzzified_x shape: (batch_size, n_inputs, n_memberships)
        # We want rule strengths. Using Product T-norm.
        # Use softmax on rule_weights to "select" one membership per input per rule
        weights = torch.softmax(self.rule_weights, dim=-1) # (n_rules, n_inputs, n_memberships)
        
        # Weighted sum of memberships for each input in each rule
        # (batch_size, 1, n_inputs, n_memberships) * (1, n_rules, n_inputs, n_memberships)
        # sum over n_memberships -> (batch_size, n_rules, n_inputs)
        selected_memberships = torch.sum(fuzzified_x.unsqueeze(1) * weights.unsqueeze(0), dim=-1)
        
        # Product T-norm across inputs for each rule
        # (batch_size, n_rules)
        rule_strengths = torch.prod(selected_memberships, dim=-1)
        return rule_strengths

class DefuzzificationLayer(nn.Module):
    def __init__(self, n_rules, n_outputs):
        super(DefuzzificationLayer, self).__init__()
        self.n_rules = n_rules
        self.n_outputs = n_outputs
        # Consequent parameters (e.g., Takagi-Sugeno or singleton weights)
        self.consequents = nn.Parameter(torch.randn(n_rules, n_outputs))

    def forward(self, rule_strengths):
        # rule_strengths shape: (batch_size, n_rules)
        # Weighted average defuzzification
        # (batch_size, n_rules) @ (n_rules, n_outputs) -> (batch_size, n_outputs)
        numerator = torch.matmul(rule_strengths, self.consequents)
        denominator = torch.sum(rule_strengths, dim=-1, keepdim=True) + 1e-8
        return numerator / denominator

class TorchFuzzy(nn.Module):
    def __init__(self, n_inputs, n_memberships, n_rules, n_outputs=1):
        super(TorchFuzzy, self).__init__()
        self.fuzzification = FuzzificationLayer(n_inputs, n_memberships)
        self.logic = FuzzyLogicLayer(n_inputs, n_memberships, n_rules)
        self.defuzzification = DefuzzificationLayer(n_rules, n_outputs)

    def forward(self, x):
        x = self.fuzzification(x)
        x = self.logic(x)
        x = self.defuzzification(x)
        return x


    
def main():
    # Parameters
    n_inputs = 2
    n_memberships = 3
    n_rules = 5
    n_outputs = 1
    batch_size = 4

    # Create model
    model = TorchFuzzy(n_inputs, n_memberships, n_rules, n_outputs)
    print("Model created successfully.")

    # Create dummy input
    x = torch.randn(batch_size, n_inputs)

    # Forward pass
    output = model(x)
    print(f"Forward pass successful. Output shape: {output.shape}")

    # Backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass successful. Gradients computed.")

    # Check if parameters have gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name} is present.")
        else:
            print(f"MISSING gradient for {name}")


if __name__ == "__main__":
    main()