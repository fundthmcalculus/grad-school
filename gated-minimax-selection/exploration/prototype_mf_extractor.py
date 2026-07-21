"""
Prototype-based membership function extraction from VAT/IVAT ultrametric structure.

Decouples the dissimilarity-space block extraction from the membership function
parametrization, allowing the model builder to select from prototype families
(triangular, trapezoidal, Gaussian, sigmoid, exponential) and extract their
parameters via dissimilarity-space ramps.

Core idea: block + prototype → fit parameters → executable MF.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean


# ============================================================================
# Prototype Base Classes
# ============================================================================

@dataclass
class MFParameters:
    """Holder for fitted membership function parameters."""
    prototype_type: str
    params: dict
    medoid_idx: int
    block_members: set


class PrototypeMF(ABC):
    """Abstract base for a parametric membership function."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, x: float, params: dict) -> float:
        """
        Evaluate the MF at a single dissimilarity value.

        Args:
            x: dissimilarity value (in normalized [0, 1] space, or raw if unbounded)
            params: dict of fitted parameters for this MF

        Returns:
            membership value in [0, 1]
        """
        pass

    @abstractmethod
    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Fit MF parameters from a dissimilarity ramp and masks.

        Args:
            dissim_ramp: 1-D array of distances from block medoid to all points
            core_mask: boolean array True for points in the block core
            support_mask: boolean array True for points in the block support

        Returns:
            dict of fitted parameters (prototype-specific)
        """
        pass

    @abstractmethod
    def params_to_string(self, params: dict) -> str:
        """Human-readable representation of parameters."""
        pass


# ============================================================================
# Prototype Implementations
# ============================================================================

class TriangularMF(PrototypeMF):
    """
    Triangular membership function: linear rise, peak, linear fall.

    Parameters: {a, b, c} where:
      a = left foot (start of rise)
      b = peak (maximum at 1.0)
      c = right foot (end of fall)
    """

    def __init__(self):
        super().__init__("triangular")

    def evaluate(self, x: float, params: dict) -> float:
        a, b, c = params['a'], params['b'], params['c']
        if x < a or x > c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if b != a else 0.0
        else:  # x > b
            return (c - x) / (c - b) if c != b else 0.0

    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Extract triangular parameters from dissimilarity ramp.

        Strategy:
        - b (peak) = median dissimilarity in core (or 0 if core is tight)
        - a = dissim at 10th percentile of support
        - c = dissim at 90th percentile of support
        """
        support_dissims = dissim_ramp[support_mask]
        core_dissims = dissim_ramp[core_mask]

        b = np.min(core_dissims) if len(core_dissims) > 0 else np.percentile(support_dissims, 25)
        a = np.percentile(support_dissims, 10)
        c = np.percentile(support_dissims, 90)

        # Ensure ordering
        a = min(a, b)
        c = max(c, b)

        return {'a': float(a), 'b': float(b), 'c': float(c)}

    def params_to_string(self, params: dict) -> str:
        return f"Triangle(a={params['a']:.3f}, peak={params['b']:.3f}, c={params['c']:.3f})"


class TrapezoidalMF(PrototypeMF):
    """
    Trapezoidal membership function: linear rise, plateau, linear fall.

    Parameters: {a, b, c, d} where:
      a = left foot
      b = left plateau start
      c = right plateau end
      d = right foot
    """

    def __init__(self):
        super().__init__("trapezoidal")

    def evaluate(self, x: float, params: dict) -> float:
        a, b, c, d = params['a'], params['b'], params['c'], params['d']
        if x < a or x > d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if b != a else 0.0
        else:  # x > c
            return (d - x) / (d - c) if d != c else 0.0

    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Extract trapezoidal parameters: wide core, gentle feet.

        - b, c = left/right bounds of the core region
        - a, d = support boundaries
        """
        core_dissims = dissim_ramp[core_mask]
        support_dissims = dissim_ramp[support_mask]

        if len(core_dissims) > 0:
            b = np.percentile(core_dissims, 25)
            c = np.percentile(core_dissims, 75)
        else:
            b = np.percentile(support_dissims, 25)
            c = np.percentile(support_dissims, 50)

        a = np.percentile(support_dissims, 5)
        d = np.percentile(support_dissims, 95)

        a = min(a, b)
        d = max(d, c)

        return {'a': float(a), 'b': float(b), 'c': float(c), 'd': float(d)}

    def params_to_string(self, params: dict) -> str:
        return f"Trap(a={params['a']:.3f}, b={params['b']:.3f}, c={params['c']:.3f}, d={params['d']:.3f})"


class GaussianMF(PrototypeMF):
    """
    Gaussian membership function: μ(x) = exp(-((x - μ) / σ)²).

    Parameters: {mu, sigma}
    """

    def __init__(self):
        super().__init__("gaussian")

    def evaluate(self, x: float, params: dict) -> float:
        mu, sigma = params['mu'], params['sigma']
        if sigma == 0:
            return 1.0 if x == mu else 0.0
        return np.exp(-((x - mu) / sigma) ** 2)

    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Extract Gaussian parameters: mean and std of the dissimilarity ramp.
        """
        core_dissims = dissim_ramp[core_mask]
        support_dissims = dissim_ramp[support_mask]

        # Mean at the tightest core point
        mu = np.min(core_dissims) if len(core_dissims) > 0 else np.min(support_dissims)

        # Standard deviation from support spread
        sigma = np.std(support_dissims) if len(support_dissims) > 1 else (np.max(support_dissims) - mu)
        sigma = max(sigma, 1e-6)  # Avoid zero std

        return {'mu': float(mu), 'sigma': float(sigma)}

    def params_to_string(self, params: dict) -> str:
        return f"Gaussian(μ={params['mu']:.3f}, σ={params['sigma']:.3f})"


class SigmoidMF(PrototypeMF):
    """
    Sigmoid membership function (soft step): μ(x) = 1 / (1 + exp(-k·(x - x₀))).

    Parameters: {x0, k} where k is steepness (higher = sharper step).

    Use case: "at least as far as this threshold" or soft boundaries.
    """

    def __init__(self):
        super().__init__("sigmoid")

    def evaluate(self, x: float, params: dict) -> float:
        x0, k = params['x0'], params['k']
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Extract sigmoid parameters.

        - x0 = the inflection point, roughly at the boundary between core and support
        - k = steepness, inversely proportional to the transition width
        """
        core_dissims = dissim_ramp[core_mask]
        support_dissims = dissim_ramp[support_mask]

        # Inflection at the transition
        x0 = np.percentile(support_dissims, 50) if len(support_dissims) > 0 else np.max(core_dissims)

        # Steepness: inverse of the transition width (75th - 25th percentile)
        transition_width = np.percentile(support_dissims, 75) - np.percentile(support_dissims, 25)
        k = 4.0 / max(transition_width, 1e-6)  # Scaled for [0.25, 0.75] transition

        return {'x0': float(x0), 'k': float(k)}

    def params_to_string(self, params: dict) -> str:
        return f"Sigmoid(x₀={params['x0']:.3f}, k={params['k']:.3f})"


class ExponentialDecayMF(PrototypeMF):
    """
    Exponential decay: μ(x) = exp(-λ·x).

    Parameters: {lambda_} (decay rate).

    Interpretation: reachability-based membership; higher λ = sharper cutoff.
    """

    def __init__(self):
        super().__init__("exponential_decay")

    def evaluate(self, x: float, params: dict) -> float:
        lam = params['lambda']
        return np.exp(-lam * x)

    def extract_parameters(self, dissim_ramp: np.ndarray, core_mask: np.ndarray,
                          support_mask: np.ndarray) -> dict:
        """
        Extract exponential decay rate.

        Strategy: fit μ(x) = exp(-λ·x) to the data.
        At x = distance_to_boundary, μ should drop to 0.5.
        So: 0.5 = exp(-λ·d_boundary) → λ = -ln(0.5) / d_boundary
        """
        support_dissims = dissim_ramp[support_mask]

        # Boundary distance (where membership drops below 0.5)
        d_boundary = np.percentile(support_dissims, 50) if len(support_dissims) > 0 else 1.0
        d_boundary = max(d_boundary, 1e-6)

        lam = -np.log(0.5) / d_boundary  # ~ 0.693 / d_boundary

        return {'lambda': float(lam)}

    def params_to_string(self, params: dict) -> str:
        return f"ExpDecay(λ={params['lambda']:.3f})"


# ============================================================================
# Metric Signature & Auto-Selector
# ============================================================================

@dataclass
class MetricSignature:
    """Signature of a block used to auto-select prototype."""
    cohesion: float      # (h_d - h_b) / mean intra-cluster distance
    symmetry: float      # left-to-right balance (1.0 = symmetric)
    concentration: float # fraction of points in tight core


def compute_metric_signature(dissim_ramp: np.ndarray, core_mask: np.ndarray,
                            support_mask: np.ndarray) -> MetricSignature:
    """
    Compute metric signature for prototype auto-selection.

    Args:
        dissim_ramp: 1-D array of distances from medoid
        core_mask, support_mask: boolean arrays

    Returns:
        MetricSignature with cohesion, symmetry, concentration
    """
    core_dissims = dissim_ramp[core_mask]
    support_dissims = dissim_ramp[support_mask]

    if len(core_dissims) == 0:
        # Degenerate: no core, only support
        core_dissims = support_dissims[:len(support_dissims)//2]

    # Cohesion: ratio of death-birth interval to mean internal distance
    h_b = np.min(core_dissims)
    h_d = np.max(support_dissims) if len(support_dissims) > 0 else np.max(core_dissims)
    mean_internal = np.mean(dissim_ramp[support_mask | core_mask])
    cohesion = (h_d - h_b) / max(mean_internal, 1e-6)

    # Symmetry: left-to-right distance balance
    median_d = np.median(dissim_ramp[support_mask | core_mask])
    left = dissim_ramp[(dissim_ramp <= median_d) & (support_mask | core_mask)]
    right = dissim_ramp[(dissim_ramp > median_d) & (support_mask | core_mask)]
    sum_left = np.sum(median_d - left) if len(left) > 0 else 1.0
    sum_right = np.sum(right - median_d) if len(right) > 0 else 1.0
    symmetry = sum_left / max(sum_right, 1e-6)

    # Concentration: fraction of points in core
    concentration = len(core_dissims) / max(len(core_dissims) + len(support_dissims), 1)

    return MetricSignature(
        cohesion=float(cohesion),
        symmetry=float(symmetry),
        concentration=float(concentration)
    )


def auto_select_prototype(signature: MetricSignature) -> str:
    """
    Auto-select a prototype based on metric signature.

    Heuristic rules (tunable):
    - High cohesion + high concentration → triangular (tight peak)
    - Low cohesion → trapezoidal (wide support)
    - Symmetric, moderate concentration → gaussian (smooth blend)
    - Asymmetric → exponential (one-sided decay)
    """
    if signature.cohesion > 0.7 and signature.concentration > 0.6:
        return "triangular"
    elif signature.cohesion < 0.4:
        return "trapezoidal"
    elif 0.8 < signature.symmetry < 1.2 and 0.4 < signature.concentration < 0.8:
        return "gaussian"
    else:
        return "exponential_decay"


# ============================================================================
# Main Extractor
# ============================================================================

class VAT_MF_Extractor:
    """
    Prototype-based membership function extractor.

    Workflow:
    1. User specifies prototype family and core-selection strategy
    2. For each block in the VAT/IVAT hierarchy:
       a. Compute dissimilarity ramp from block medoid to all points
       b. Define core and support regions
       c. Fit the selected prototype's parameters
       d. (Optional) normalize to partition of unity
    3. Return parametric MFs ready for linguistic rule construction or feature-space embedding
    """

    PROTOTYPES = {
        'triangular': TriangularMF(),
        'trapezoidal': TrapezoidalMF(),
        'gaussian': GaussianMF(),
        'sigmoid': SigmoidMF(),
        'exponential_decay': ExponentialDecayMF(),
    }

    def __init__(self, prototype: str = 'auto', core_selection: str = 'persistence',
                 ruspini_normalize: bool = False, verbose: bool = False):
        """
        Args:
            prototype: 'triangular', 'trapezoidal', 'gaussian', 'sigmoid',
                      'exponential_decay', or 'auto' (select per-block)
            core_selection: 'persistence' (use block birth), or 'manual' (caller specifies)
            ruspini_normalize: if True, normalize membership functions to partition of unity
            verbose: print parameter extraction details
        """
        self.prototype_name = prototype
        self.core_selection = core_selection
        self.ruspini_normalize = ruspini_normalize
        self.verbose = verbose

        if prototype != 'auto':
            if prototype not in self.PROTOTYPES:
                raise ValueError(f"Unknown prototype: {prototype}")
            self.prototype = self.PROTOTYPES[prototype]
        else:
            self.prototype = None  # Will auto-select per block

    def extract_membership_functions(self, Dstar: np.ndarray, blocks: list,
                                     medoids: list, birth_death_heights: dict) -> list:
        """
        Extract membership functions from VAT blocks.

        Args:
            Dstar: minimax distance matrix (n x n)
            blocks: list of sets, each containing member indices for a block
            medoids: list of medoid indices (one per block)
            birth_death_heights: dict {block_id: (h_b, h_d)}

        Returns:
            List of MFParameters, one per block
        """
        n = Dstar.shape[0]
        mf_params = []

        for block_id, (block_members, medoid_idx) in enumerate(zip(blocks, medoids)):
            if len(block_members) == 0:
                continue

            # Compute dissimilarity ramp from medoid
            dissim_ramp = Dstar[medoid_idx, :]

            # Define core and support
            h_b, h_d = birth_death_heights.get(block_id, (dissim_ramp[list(block_members)].min(),
                                                          dissim_ramp[list(block_members)].max()))

            core_mask = np.zeros(n, dtype=bool)
            support_mask = np.zeros(n, dtype=bool)

            for idx in block_members:
                if dissim_ramp[idx] <= h_b:
                    core_mask[idx] = True
                    support_mask[idx] = True
                elif dissim_ramp[idx] <= h_d:
                    support_mask[idx] = True

            # Select prototype (auto or fixed)
            if self.prototype_name == 'auto':
                sig = compute_metric_signature(dissim_ramp, core_mask, support_mask)
                proto_name = auto_select_prototype(sig)
                prototype = self.PROTOTYPES[proto_name]
                if self.verbose:
                    print(f"Block {block_id}: signature={sig} → prototype={proto_name}")
            else:
                prototype = self.prototype
                proto_name = self.prototype_name

            # Extract parameters
            params = prototype.extract_parameters(dissim_ramp, core_mask, support_mask)

            mf = MFParameters(
                prototype_type=proto_name,
                params=params,
                medoid_idx=medoid_idx,
                block_members=block_members
            )
            mf_params.append(mf)

            if self.verbose:
                print(f"Block {block_id}: {prototype.params_to_string(params)}")

        # Normalize to partition of unity if requested
        if self.ruspini_normalize:
            mf_params = self._normalize_partition_of_unity(mf_params, Dstar)

        return mf_params

    def _normalize_partition_of_unity(self, mf_params: list, Dstar: np.ndarray) -> list:
        """
        Normalize membership functions so that ∑_c μ_c(x) ≈ 1 everywhere.

        For now: simple sum-and-normalize at each point (expensive on large data,
        but illustrates the concept).
        """
        # This is a stub; full implementation would require sampling/grid approach
        # for computational efficiency.
        if len(mf_params) > 1:
            # Post-process: for each point x, normalize μ_c(x) / ∑_c' μ_c'(x)
            # Left as an exercise for the caller with actual data
            pass

        return mf_params

    def evaluate_membership(self, mf: MFParameters, dissim_value: float) -> float:
        """Evaluate a single MF at a dissimilarity value."""
        prototype = self.PROTOTYPES[mf.prototype_type]
        return prototype.evaluate(dissim_value, mf.params)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Toy example: synthetic blocks with fixed dissimilarity ramps
    np.random.seed(42)

    # Simulate two blocks
    block1_dissims = np.concatenate([
        np.random.normal(0.1, 0.05, 20),  # tight core
        np.random.uniform(0.1, 0.4, 10)   # support tail
    ])

    block2_dissims = np.concatenate([
        np.random.normal(0.2, 0.15, 15),  # diffuse core
        np.random.uniform(0.2, 0.6, 15)   # wide support
    ])

    print("=" * 70)
    print("Prototype MF Extraction: Toy Example")
    print("=" * 70)

    # Test each prototype on block 1 (tight)
    print("\nBlock 1 (tight cluster):")
    core_mask = block1_dissims < 0.15
    support_mask = block1_dissims < 0.4

    for proto_name, prototype in [("triangular", TriangularMF()),
                                   ("gaussian", GaussianMF()),
                                   ("exponential_decay", ExponentialDecayMF())]:
        params = prototype.extract_parameters(block1_dissims, core_mask, support_mask)
        print(f"  {prototype.params_to_string(params)}")

    # Auto-select for both blocks
    print("\nAuto-selection:")
    for block_id, dissims in [(1, block1_dissims), (2, block2_dissims)]:
        core_mask = dissims < np.percentile(dissims, 30)
        support_mask = dissims < np.percentile(dissims, 80)
        sig = compute_metric_signature(dissims, core_mask, support_mask)
        selected = auto_select_prototype(sig)
        print(f"  Block {block_id}: {sig} → {selected}")

    print("\n" + "=" * 70)
