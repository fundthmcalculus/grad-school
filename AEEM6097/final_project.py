from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcmeans import FCM
from numpy.typing import NDArray
from scipy import stats
from pyclustertend.visual_assessment_of_tendency import ivat
from scipy.signal import find_peaks, peak_widths, peak_prominences
from tqdm import tqdm

from AEEM6097.aco_solver import AcoContinuousVariable, solve_gradiant
from AEEM6097.midterm_project import tsk_rule, mu_poly_set, mu_poly

# Some typing information shorthand!
i64 = np.int64
f64 = np.float64
af64 = NDArray[f64]
ai64 = NDArray[i64]

def load_data() -> tuple[af64, af64, f64, f64, list[str]]:
    # Load the `project-data/Concrete_Data.xls`
    # The columns were renamed for simplicity
    concrete_df = pd.read_excel('project-data/Concrete_Data.xls')
    # Convert to numpy array
    all_data = concrete_df.to_numpy()
    labels = concrete_df.columns.tolist()
    # Randomly permute the rows
    all_data = np.random.permutation(all_data)

    # Normalize all data
    data_min = all_data.min(axis=0)
    data_max = all_data.max(axis=0)
    all_data = (all_data - data_min)/(data_max-data_min)

    # Take 75% of the data for training
    train_idx = len(all_data) // 4 * 3
    train_data = all_data[:train_idx]
    test_data = all_data[train_idx:]
    return train_data, test_data, data_min, data_max, labels

def ivat_vis(data: af64) -> None:
    # Perform IVAT to identify cluster count
    # data_mat = ivat(data, return_odm=True)
    # plt.title("IVAT Clustering")
    # plt.show()
    pass

def fuzzy_cluster(data: af64) -> None:
    # Use fuzzy c-means and the Fuzzy Partition Coefficient to identify the total number
    # of clusters and their centroids
    fuzzy_models = []
    num_clusters = np.r_[2:10]
    for k in num_clusters:
        fcm = FCM(n_clusters=k,m=4,random_state=37)
        fcm.fit(data)
        fuzzy_models.append(fcm)

    # plt.figure()
    # y_pec = [x.partition_entropy_coefficient for x in fuzzy_models]
    # y_pc = [x.partition_coefficient for x in fuzzy_models]
    # plt.plot(num_clusters, y_pec)
    # plt.title("Fuzzy C-Means Partition Entropy Coefficient")
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("FPEC value")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(y_pec,y_pc)
    # plt.title("Fuzzy C-Means Partition Coefficient vs Entropy")
    # plt.xlabel("FPEC")
    # plt.ylabel("FPC")
    # plt.show()

@dataclass
class PeakInfo:
    x: i64
    left_base: i64
    right_base: i64
    prominence: f64
    y: f64
    half_width: f64

# For some reason, no matter how many clusters we pick, we get the same centers. Let's plot the normalized distribution diagrams.
def kernel_density_partitioning(norm_data: af64, labels: list[str]) -> list[list[PeakInfo]]:
    # Colors for different distributions
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    # Create a smooth x-axis for plotting
    n_samples = len(norm_data)
    x = np.linspace(0, 1, n_samples)

    # Plot density for each column as a line plot
    peak_data: list[list[PeakInfo]] = []
    for i in range(norm_data.shape[1]):
        # Use kernel density estimation for smooth distribution curves
        density = stats.gaussian_kde(norm_data[:, i])
        data_pdf = density(x)
        # Insert a 0 on each end so the prominence calculation works
        peak_data.append(get_peak_data(n_samples, data_pdf, x))
        plt.plot(x, data_pdf, color=colors[i], label=labels[i], linewidth=2)
    # Mark the peaks after, so we can legend easily
    for ij, peak_d in enumerate(peak_data):
        plt.plot([p.x for p in peak_d], [p.y for p in peak_d], "x", color=colors[ij])

    # Adding plot details
    plt.xlabel('Value (Normalized to [0,1])')
    plt.ylabel('Density')
    plt.title('Distribution of Normalized Data Columns')
    plt.grid(True, alpha=0.3)
    plt.legend(labels)
    plt.xlim(0, 1)  # Since data is normalized to [0,1]
    # Show the plot
    plt.tight_layout()
    plt.show()

    print("Number of peaks for each variable:")
    print("\n".join([f"{labels[ij]}:{len(x)}" for ij, x in enumerate(peak_data)]))

    return peak_data


def get_peak_data(n_samples: i64, data_pdf: af64, x: af64) -> list[PeakInfo]:
    mod_pdf = np.zeros(data_pdf.shape[0] + 2)
    mod_pdf[1:-1] = data_pdf
    peak_indexes, peak_info = find_peaks(mod_pdf, prominence=0.3)
    prominences = peak_prominences(mod_pdf, peak_indexes)
    results_half = peak_widths(mod_pdf, peak_indexes, prominence_data=prominences)
    cur_peak_lst = []
    for ij, peak_idx in enumerate(peak_indexes):
        peak_obj = PeakInfo(x=x[peak_idx],
                            y=mod_pdf[peak_idx],
                            left_base=x[max(0,peak_info['left_bases'][ij])],
                            right_base=x[min(peak_info['right_bases'][ij],len(x)-1)],
                            prominence=peak_info['prominences'][ij],
                            half_width=results_half[2][ij] / n_samples)  # Order is left-half-idx, half-y, width in samples, right-half-idx
        cur_peak_lst.append(peak_obj)
    return cur_peak_lst


# NOTE - Most of the outputs are uniform distribution (approximately), so having
# multiple clusters will make little sense. Instead of clustering by sample, we
# should pick out 1-4 clusters along each feature individually, from the normalized
# distribution plot.
# Cement: 1
# Slag: 2
# Fly Ash: 2
# Water: 2?
# SuperPlasticizer: 2!
# Coarse Agg: 1
# Fine Agg: 1
# Age: 2,3?
# Compressive Strength: 1, but N/A (output)

class VariablesInfo:
    def __init__(self):
        # TODO - Refactor this to ACO solver?
        self.variables: list[AcoContinuousVariable] = []
        self.mu_indexes: list[list[int]] = []
        self.rule_op_indexes: list[int] = []
        self.rule_coeff_indexes: list[list[int,]] = []
        self.rule_args: ai64 = None

    def find_variable(self, name_prefix: str) -> tuple[int, AcoContinuousVariable | None]:
        for idx, variable in enumerate(self.variables):
            if variable.name.startswith(name_prefix):
                return idx, variable
        return -1, None

    def append_variable(self, x: AcoContinuousVariable | list[AcoContinuousVariable]) -> None:
        # Check the name for type information.
        var_idx: int | list[int] = 0
        var_name: str = ""
        if isinstance(x, AcoContinuousVariable):
            var_idx = len(self.variables)
            self.variables.append(x)
            var_name = x.name
        else:
            var_idx = list(range(len(self.variables),len(self.variables)+len(x)))
            self.variables.extend(x)
            var_name = x[0].name
        if "and/or-op" in var_name:
            if isinstance(x, AcoContinuousVariable):
                self.rule_op_indexes.append(var_idx)
            else:
                raise Exception("lists of Operator variables are not supported")
        elif "mu_" in var_name:
            if isinstance(x, list):
                self.mu_indexes.append(var_idx)
            else:
                raise Exception("single element membership functions are not supported")
        elif "rule-coeff" in var_name:
            if isinstance(x, list):
                self.rule_coeff_indexes.append(var_idx)
            else:
                # TODO - Handle 0th-order and 1st-order
                raise Exception("single element TSK rules are not supported")
        else:
            raise Exception("variable name is not supported")


    @property
    def n_membership_fcns(self) -> int:
        return len(self.mu_indexes)

    @property
    def n_rules(self) -> int:
        return len(self.rule_args)


# For now, pick the peaks, and rely on the optimizer to get the width and exact location correct.
# NOTE - By normalizing all data, we can constrain the solution space on the optimizer and the TSK rules! :)

def all_rule_permutations(states_per_var: list[int]) -> af64:
    # Create ranges for each variable based on its number of states
    ranges = [list(range(n_states)) for n_states in states_per_var]
    # Use itertools.product to generate the Cartesian product
    permutations = np.array(list([list(x) for x in product(*ranges)]))
    # Go through each state max and sum it on subsequent indexes.
    for start_col, state_count in enumerate(states_per_var):
        permutations[:,start_col+1:] += state_count
    return permutations

def create_tsk_variables(peak_data: list[list[PeakInfo]]) -> VariablesInfo:
    aco_variables = []
    # The output feature doesn't count.
    n_features = len(peak_data) - 1
    # NOTE - This caching is a performance optimization
    info = VariablesInfo()
    # Create the membership functions
    for i_var, peak_d in enumerate(peak_data):
        for j_mu in range(len(peak_d)):
            peak_d_j = peak_d[j_mu].y
            width_d_j = peak_d[j_mu].half_width / 2.0
            # TODO - Handle different membership functions on different variables!
            mu_var_params = [
                AcoContinuousVariable(
                    f"mu_{i_var + 1}({j_mu + 1})-a", 0.0, 1.0, peak_d_j,
                ),
                AcoContinuousVariable(
                    f"mu_{i_var + 1}({j_mu + 1})-b", 0.0001, 1, width_d_j
                )]
            info.append_variable(mu_var_params)

    # Generate the cartesian product of rule states
    info.rule_args = all_rule_permutations([len(x) for x in peak_data])

    print("Number of membership functions:", info.n_membership_fcns)
    print("Number of rules:", info.n_rules)

    for k in range(info.n_rules):
        info.append_variable(
            # NOTE - This is the exploratory magic of degree of or/and
            AcoContinuousVariable(f"rule-and/or-op-{k+1}", 1.0, 1.0, 1.0),
        )
        info.rule_op_indexes.append(len(aco_variables)-1)
        coeff_max = n_features
        feat_coeffs = [AcoContinuousVariable(f"rule-coeff-{k+1}-f-a{ij+1}", -coeff_max, coeff_max, 0.1) for ij in range(n_features)]
        feat_coeffs.append(AcoContinuousVariable(f"rule-coeff-{k+1}-f-c", -coeff_max, coeff_max, -0.1))
        info.append_variable(feat_coeffs)

    return info


def mu_poly2(x: af64, mu_ab: af64) -> af64:
    a = mu_ab[0]
    b = mu_ab[1]
    return 1.0 / (1.0 + ((x - a) / b) ** 2)


def extract_mu_from_args(var_args: af64, pts: af64, variables_info: VariablesInfo) -> af64:
    mu_vars = np.zeros((pts.shape[0],variables_info.n_membership_fcns))
    for col_idx in range(pts.shape[1]):
        for ivar,mu_idxs in enumerate(variables_info.mu_indexes):
            mu_coeff = var_args[mu_idxs]
            mu_vars[:,col_idx] = mu_poly2(pts[:,col_idx], mu_coeff)
    return mu_vars


def fuzzy_or(s: af64, axis=0) -> af64:
    if axis > 1:
        raise NotImplementedError("fuzzy_or for 3D arrays")
    if axis == 1:
        s = s.T
    o = s[:, 0]
    for col in range(1, s.shape[1]):
        o = o + s[:, col] - o * s[:, col]
    return o

def fuzzy_and(s: af64, axis=0) -> af64:
    if axis > 1:
        raise NotImplementedError("fuzzy_and for 3D arrays")
    if axis == 1:
        s = s.T
    a = s[:,0]
    for col in range(1, s.shape[1]):
        a *= s[:, col]
    return a


def compute_fuzzy_system(var_args: af64, pts: af64, variables_info: VariablesInfo)-> tuple[f64, af64]:
    mu_vars = extract_mu_from_args(var_args, pts, variables_info)
    # Do the rules in order
    sum_R = 0.0
    sum_ZR = 0.0
    for rule_idx in range(variables_info.n_rules):
        and_c = var_args[variables_info.rule_op_indexes[rule_idx]]
        tsk_coeffs = var_args[variables_info.rule_coeff_indexes[rule_idx]]
        # TODO - Get membership function indexes for each variable!
        var_idxs = variables_info.rule_args[rule_idx,:]
        # NOTE - This is magic!
        r_eval_and = fuzzy_and(mu_vars[:, var_idxs])
        r_eval_or = fuzzy_or(mu_vars[:, var_idxs])
        # Exclude the last column we don't include the output.
        z_eval = tsk_rule(pts[:,:-1], tsk_coeffs)
        r_eval = (1-and_c)*r_eval_or+and_c*r_eval_and
        # Fuzzify!
        sum_R += r_eval
        sum_ZR += r_eval*z_eval

    # Weighted defuzzy!
    z_defuzzy = sum_ZR / sum_R
    # Compute the RMS error
    rms_error = np.sqrt(np.mean((z_defuzzy - pts[:, -1]) ** 2))
    return rms_error, z_defuzzy


def main():
    train_data, test_data, data_min, data_max, labels = load_data()
    peak_data = kernel_density_partitioning(train_data, labels)
    variables_info = create_tsk_variables(peak_data)
    print("Number of domain variables:", len(variables_info.variables))

    # Here is the goal-seeking function
    nfev = 0
    min_err = 1.0E10
    with tqdm() as pbar:
        def fuzzy_test2(x: np.ndarray) -> f64:
            nonlocal nfev, min_err
            rms_err, _ = compute_fuzzy_system(x,train_data,variables_info)
            # This is an inversion of control, but `scipy.optimize.minimize` has limitations
            min_err = min(rms_err, min_err)
            nfev += 1
            pbar.update(nfev)
            pbar.set_description(f"err={min_err}")

            return rms_err

        best_soln, soln_history = solve_gradiant(fuzzy_test2, variables_info.variables)
        rms_err_test, test_result = compute_fuzzy_system(best_soln, test_data, variables_info)
        print(f"Train error={soln_history[-1]}, Test error={rms_err_test}")

        # TODO - Show the plot of test result vs defuzzified test result (ideally they overlap)
        plt.figure()
        plt.title("Test Output Comparison")
        plt.plot(test_data[:,-1],'r',label="test data")
        plt.plot(test_result,'b',label="predicted data")
        plt.xlabel("Sample")
        plt.legend()
        plt.ylabel("Normalized Value")
        plt.show()


if __name__ == "__main__":
    main()