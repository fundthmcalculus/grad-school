import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm

class GaussianFuzzification(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, means, sigmas):
        # x: (batch_size, n_inputs, 1)
        # means: (n_inputs, n_memberships)
        # sigmas: (n_inputs, n_memberships)
        
        # Reshape means and sigmas to (1, n_inputs, n_memberships) for broadcasting
        means = means.unsqueeze(0)
        sigmas = sigmas.unsqueeze(0)
        
        diff = x - means
        z = diff / sigmas
        y = torch.exp(-0.5 * torch.pow(z, 2))
        
        ctx.save_for_backward(x, means, sigmas, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, means, sigmas, y = ctx.saved_tensors

        # grad_output shape: (batch_size, n_inputs, n_memberships)
        # y shape: (batch_size, n_inputs, n_memberships)
        # x shape: (batch_size, n_inputs, 1)
        # means shape: (1, n_inputs, n_memberships)
        # sigmas shape: (1, n_inputs, n_memberships)

        diff = x - means
        sigmas_sq = torch.pow(sigmas, 2)
        sigmas_cub = torch.pow(sigmas, 3)

        # grad_x = sum over memberships of (grad_output * dy/dx)
        # dy/dx = -y * (x - mean) / sigma^2
        grad_x = grad_output * (-y * diff / sigmas_sq)
        grad_x = grad_x.sum(dim=-1, keepdim=True) # (batch_size, n_inputs, 1)

        # grad_means = sum over batch of (grad_output * dy/dmean)
        # dy/dmean = y * (x - mean) / sigma^2
        grad_means = grad_output * (y * diff / sigmas_sq)
        grad_means = grad_means.sum(dim=0) # (n_inputs, n_memberships)

        # grad_sigmas = sum over batch of (grad_output * dy/dsigma)
        # dy/dsigma = y * (x - mean)^2 / sigma^3
        grad_sigmas = grad_output * (y * torch.pow(diff, 2) / sigmas_cub)
        grad_sigmas = grad_sigmas.sum(dim=0) # (n_inputs, n_memberships)

        return grad_x, grad_means, grad_sigmas


class FuzzificationLayer(nn.Module):
    def __init__(self, n_inputs: int, n_memberships: int):
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
        return GaussianFuzzification.apply(x, self.means, self.sigmas)


class FuzzyLogicLayer(nn.Module):
    def __init__(self, n_inputs: int, n_memberships: int, n_rules: int):
        super(FuzzyLogicLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_memberships = n_memberships
        self.n_rules = n_rules
        # Rule antecedents: which membership function of which input is used in each rule
        # For simplicity, let's use a soft selection / weight matrix for now
        # or a fixed mapping if we want to follow a grid of rules.
        # Given n_rules, we can have a learnable mapping.
        # Shape: (n_rules, n_inputs)
        # self.input_selectors = nn.Parameter(torch.randint(0,self.n_memberships, (n_inputs, n_rules)), requires_grad=False)
        self.input_selectors = nn.Parameter(torch.rand((n_inputs, n_rules))) # [N/A] & [0, self.n_memberships) & [N/A]
        a = 1

    def forward(self, fuzzified_x):
        # fuzzified_x shape: (batch_size, n_inputs, n_memberships)
        # input_selectors: (n_inputs, n_rules)
        # output: (batch_size, n_rules)

        # Expand input_selectors to match batch dimension
        # Shape: (1, n_inputs, n_rules)
        selectors_expanded = self.input_selectors.unsqueeze(0)
        # Scale to the actual indexes
        # selectors_expanded = torch.round(selectors_expanded * (1+self.n_memberships)).to(torch.int32)
        selectors_expanded = torch.round(selectors_expanded * (0+self.n_memberships)).to(torch.int32)

        # Expand for batch size
        # Shape: (batch_size, n_inputs, n_rules)
        selectors_batched = selectors_expanded.expand(fuzzified_x.size(0), -1, -1)

        # Augment fuzzified_x with an additional column of unity, for "membership not used"
        fuzzified_x = torch.cat([
            # torch.ones_like(fuzzified_x[..., :1]),
            fuzzified_x,
            torch.ones_like(fuzzified_x[..., :1])], dim=-1)

        # Gather membership values based on selectors
        # Shape: (batch_size, n_inputs, n_rules)
        selected_memberships = torch.gather(fuzzified_x, dim=2, index=selectors_batched)

        # Compute rule strengths by taking product across inputs
        # Shape: (batch_size, n_rules)
        rule_strengths = torch.prod(selected_memberships, dim=1)

        return rule_strengths


class DefuzzificationLayer(nn.Module):
    def __init__(self, n_rules: int, n_inputs: int):
        super(DefuzzificationLayer, self).__init__()
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        # Consequent parameters (e.g., Takagi-Sugeno or singleton weights)
        self.consequents = nn.Parameter(torch.randn(n_inputs+1, n_rules))

    def forward(self, rule_strengths, inputs):
        # rule_strengths shape: (batch_size, n_rules)
        # inputs shape: (batch_size, n_inputs)
        # consequents shape: (n_inputs+1, n_rules)
        # Takagi-Sugeno defuzzification: each rule has linear output c0 + c1*x1 + c2*x2 + ...

        # Add bias term (constant 1) to inputs
        # Shape: (batch_size, n_inputs+1)
        ones = torch.ones(inputs.size(0), 1, device=inputs.device)
        inputs_with_bias = torch.cat([ones, inputs], dim=1)

        # Compute consequent outputs for each rule: z_i = c0_i + c1_i*x1 + c2_i*x2 + ...
        # (batch_size, n_inputs+1) @ (n_inputs+1, n_rules) -> (batch_size, n_rules)
        rule_outputs = torch.matmul(inputs_with_bias, self.consequents)

        # Weighted average defuzzification
        numerator = torch.sum(rule_strengths * rule_outputs, dim=-1, keepdim=True)
        denominator = torch.sum(rule_strengths, dim=-1, keepdim=True) + 1e-8 # Stability guarantee
        return numerator / denominator


class TorchFuzzy(nn.Module):
    def __init__(self, n_inputs: int, n_memberships: int, n_rules: int):
        super(TorchFuzzy, self).__init__()
        self.fuzzification = FuzzificationLayer(n_inputs, n_memberships)
        self.logic = FuzzyLogicLayer(n_inputs, n_memberships, n_rules)
        self.defuzzification = DefuzzificationLayer(n_rules, n_inputs)

    def forward(self, x):
        in_fuzzy_x = self.fuzzification(x)
        out_fuzzy_x = self.logic(in_fuzzy_x)
        y = self.defuzzification(out_fuzzy_x, x)
        return y


    
def main2d():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Parameters for the fuzzy system
    # TODO - Allow changing the number of memberships by input.
    n_samples = 1000
    p_test = 0.2
    test_samples = int(p_test * n_samples)
    n_inputs = 2
    n_memberships = 3  # Increase memberships for better approximation
    n_rules = n_inputs ** n_memberships  # Increase rules for better approximation
    # Create model
    model = TorchFuzzy(n_inputs, n_memberships, n_rules)
    print("Model created.")

    # Generate training data: Z = cos(X) * sin(Y)
    # Range: [-pi, pi] for both X and Y
    x_train = (torch.rand(n_samples, n_inputs) * 2 - 1) * 3.14159
    z_train = torch.cos(x_train[:, 0])
    if n_inputs == 2:
        z_train *= torch.sin(x_train[:, 1])
        pass

    z_train = z_train.unsqueeze(-1)  # (n_samples, 1)

    x_train_normalized = normalize(x_train)
    z_train_normalized = normalize(z_train)

    train_torch_fuzzy(model, test_samples, x_train_normalized, z_train_normalized)

    print("\nTraining complete. The fuzzy system now approximates Z = cos(X)*sin(Y).")


def train_torch_fuzzy(model: TorchFuzzy, test_samples: int | float, x_train_normalized: Tensor, z_train_normalized: Tensor):
    # TODO - Randomly permute the points?
    if isinstance(test_samples, float):
        test_samples = int(test_samples*len(z_train_normalized))
    x_test = x_train_normalized[:test_samples]
    z_target = z_train_normalized[:test_samples]

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.SmoothL1Loss()

    # Training loop
    epochs = 300
    loss_history = []
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        # Forward pass
        predictions = model(x_train_normalized)
        loss = criterion(predictions, z_train_normalized)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Update the progress bar with the current loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    # Plot loss function
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        z_pred_normalized = model(x_test)
        # Denormalize predictions
        z_pred = z_pred_normalized

    # Plot comparison of target vs predicted values
    plt.figure(figsize=(8, 5))
    plt.plot(z_target.cpu().numpy(), label='Target', marker='o')
    plt.plot(z_pred.detach().cpu().numpy(), label='Predicted', marker='x')
    plt.title('Target vs Predicted on Test Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Z value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def normalize(x_train: Tensor) -> Tensor:
    x_min = x_train.min(dim=0, keepdim=True)[0]
    x_max = x_train.max(dim=0, keepdim=True)[0]
    x_train_normalized = (x_train - x_min) / (x_max - x_min + 1e-8)
    return x_train_normalized


def main_iris():
    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X = iris.data.features
    y = iris.data.targets

    n_inputs = X.shape[1]
    n_memberships = 3
    n_rules = n_inputs ** n_memberships  # TODO - Come up with a better one.
    model = TorchFuzzy(n_inputs, n_memberships, n_rules)
    x_train = torch.tensor(X.values, dtype=torch.float32)
    z_train = torch.tensor(pd.Categorical(y.values.ravel()).codes, dtype=torch.float32).unsqueeze(-1)

    # Permute the order of the dataset
    perm_indices = torch.randperm(x_train.size(0))
    x_train = x_train[perm_indices]
    z_train = z_train[perm_indices]

    x_train = normalize(x_train)
    z_train = normalize(z_train)

    train_torch_fuzzy(model, 1.0, x_train, z_train)


def main_wine():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    n_inputs = X.shape[1]
    n_memberships = 3
    n_rules = n_inputs * n_memberships # n_inputs ** n_memberships  # TODO - Come up with a better one.
    print("FIS Size:\n"
          "---------\n"
          f"n_inputs: {n_inputs}\n"
          f"n_memfcn: {n_memberships}\n"
          f"n_rules : {n_rules}")
    model = TorchFuzzy(n_inputs, n_memberships, n_rules)
    x_train = torch.tensor(X.values, dtype=torch.float32)
    z_train = torch.tensor(y.values, dtype=torch.float32)

    # Permute the order of the dataset
    perm_indices = torch.randperm(x_train.size(0))
    x_train = x_train[perm_indices]
    z_train = z_train[perm_indices]

    x_train = normalize(x_train)
    z_train = normalize(z_train)

    train_torch_fuzzy(model, 0.1, x_train, z_train)


if __name__ == "__main__":
    torch.manual_seed(42)
    # main2d()
    # main_iris()
    main_wine()