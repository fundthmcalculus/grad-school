from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from fcmeans import FCM
from numpy.linalg import LinAlgError
from pyclustertend import ivat
from scipy import stats
from tqdm import tqdm

from AEEM6097.aco_solver import AcoContinuousVariable, solve_gradiant, solve_aco
from AEEM6097.fuzzy_fit import PeakInfo, VariablesInfo, PeakCollection, FuzzyDataSet
from AEEM6097.fuzzy_types import *
from AEEM6097.midterm_project import tsk_rule


def load_concrete_data() -> tuple[af64, list[str]]:
    concrete_df = pd.read_excel("project-data/Concrete_Data.xls")
    # Convert to numpy array
    all_data = concrete_df.to_numpy()
    labels: list[str] = concrete_df.columns.tolist()
    return all_data, labels


def load_sonar_data() -> tuple[af64, list[str]]:
    # Load the CSV file
    sonar_data = np.genfromtxt("project-data/sonar.all-data", delimiter=",")
    # Ignore the last column. Since that is the classification value, we will deal with that later
    sonar_data = sonar_data[:, :60]
    sonar_txt_data = np.genfromtxt("project-data/sonar.all-data", delimiter=",", dtype=str)
    sonar_classification_col = sonar_txt_data[:, 60]
    # Convert "R" to a 1 in the rock col, "M" to a 1 in the mine col
    rock_col: np.ndarray = (sonar_classification_col == "R").astype(float)
    mine_col: np.ndarray = (sonar_classification_col == "M").astype(float) * -1.0
    # Now that one is "1", and the other is "-1", we can have confidence for each?
    output_col: np.ndarray = rock_col + mine_col

    # NOTE - There isn't much in the way of clustering, so we can ignore that.
    sonar_full_data = np.hstack((sonar_data, np.reshape(output_col, (len(output_col), 1))))
    labels = [f"F-Pow-{ij+1}" for ij in range(sonar_data.shape[1])]
    labels.append("Class:rock=1,mine=-1")
    return sonar_full_data, labels


def load_data() -> FuzzyDataSet:
    # Load the `project-data/Concrete_Data.xls`
    # The columns were renamed for simplicity
    all_data, labels = load_concrete_data()
    # all_data, labels = load_sonar_data()

    # Compute the linear spectra
    lin_spec(all_data)

    # Randomly permute the rows
    all_data = np.random.permutation(all_data)

    # Take 75% of the data for training
    test_pct = 0.75

    return FuzzyDataSet.create_from_data(all_data, test_pct, labels)


def lin_spec(x: af64) -> tuple[af64, af64]:
    X = np.fft.fft(x, axis=0)
    f = np.fft.fftfreq(x.shape[0])
    n_freq = len(f) // 2
    n_x = x.shape[1]
    n_i = n_x -1
    n_o = n_x - n_i
    f = f[:n_freq]
    X = X[:n_freq, :]
    X = 2.0*X
    X[:,0] /= 2.0
    plt.figure()
    plt.semilogy(f, np.abs(X))
    plt.show()
    plt.title("Linear spectra")
    G = np.zeros(shape=(n_freq,n_x,n_x))
    H1 = np.zeros(shape=(n_freq,n_o,n_i))
    for f_i in range(n_freq):
        try:
            G[f_i,:,:] = X[f_i,:].reshape((n_x,1)) @ np.conj(X[f_i,:].reshape(1,n_x))
            H1[f_i,:,:] = G[f_i,:n_o,:n_i] @  np.linalg.inv(G[f_i,-n_i:,-n_i:])
        except LinAlgError:
            pass
    plt.figure()
    plt.semilogy(f, np.abs(H1.squeeze()))
    plt.show()

    return f, X


def ivat_vis(data: af64) -> None:
    # Perform IVAT to identify cluster count
    data_mat = ivat(data, return_odm=True)
    plt.title("IVAT Clustering")
    plt.show()

def fuzzy_cluster(data: af64) -> None:
    # Use fuzzy c-means and the Fuzzy Partition Coefficient to identify the total number
    # of clusters and their centroids
    fuzzy_models = []
    num_clusters = np.r_[2:10]
    for k in num_clusters:
        fcm = FCM(n_clusters=k,m=4)
        fcm.fit(data)
        fuzzy_models.append(fcm)

    plt.figure()
    y_pec = [x.partition_entropy_coefficient for x in fuzzy_models]
    y_pc = [x.partition_coefficient for x in fuzzy_models]
    plt.plot(num_clusters, y_pec)
    plt.title("Fuzzy C-Means Partition Entropy Coefficient")
    plt.xlabel("Number of Clusters")
    plt.ylabel("FPEC value")
    plt.show()

    plt.figure()
    plt.plot(y_pec,y_pc)
    plt.title("Fuzzy C-Means Partition Coefficient vs Entropy")
    plt.xlabel("FPEC")
    plt.ylabel("FPC")
    plt.show()

def get_color(idx: int) -> str:
    # Colors for different distributions
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    return colors[idx % len(colors)]

# For some reason, no matter how many clusters we pick, we get the same centers. Let's plot the normalized distribution diagrams.
def kernel_density_partitioning(norm_data: af64, labels: list[str]) -> list[list[PeakInfo]]:
    # Create a smooth x-axis for plotting
    n_samples = len(norm_data)
    x = np.linspace(0, 1, n_samples)

    # Plot density for each column as a line plot
    peak_data: list[PeakCollection] = []
    for ij in range(norm_data.shape[1]):
        # Use kernel density estimation for smooth distribution curves
        density = stats.gaussian_kde(norm_data[:, ij])
        data_pdf = density(x)
        # Insert a 0 on each end so the prominence calculation works
        peak_data.append(PeakCollection(x,data_pdf))
        plt.plot(x, data_pdf, label=labels[ij], color=get_color(ij), linewidth=2)
    # Mark the peaks after, so we can legend easily
    for ij, peak_d in enumerate(peak_data):
        plt.plot([p.x for p in peak_d], [p.y for p in peak_d], "x", color=get_color(ij), )

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

# For now, pick the peaks and find their half-height widths as an initial guess.
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
            width_d_j = peak_d[j_mu].half_width / 2.0 # Half power width is full width, we need only one side.
            # TODO - Handle different membership functions on different variables!
            mu_var_params = [
                AcoContinuousVariable(
                    f"mu_{i_var + 1}({j_mu + 1})-a", 0.0, 1.0, peak_d_j,
                ),
                AcoContinuousVariable(
                    f"mu_{i_var + 1}({j_mu + 1})-b", 0.0001, 1, width_d_j
                )]
            info.append_variables(mu_var_params)

    # Generate the cartesian product of rule states
    info.rule_args = all_rule_permutations([len(x) for x in peak_data])

    print("Number of membership functions:", info.n_membership_fcns)
    print("Number of rules:", info.n_rules)

    for k in range(info.n_rules):
        info.append_variables(AcoContinuousVariable(f"rule-and/or-op-{k + 1}", 0.0, 1.0, 0.5))
        info.rule_op_indexes.append(len(aco_variables)-1)
        # Because everything is normalized, we can
        coeff_max = 3.0*n_features
        feat_coeffs = [AcoContinuousVariable(f"rule-coeff-{k+1}-f-a{ij+1}", -coeff_max, coeff_max, 0.1) for ij in range(n_features)]
        feat_coeffs.append(AcoContinuousVariable(f"rule-coeff-{k+1}-f-c", -coeff_max, coeff_max, -0.1))
        info.append_variables(feat_coeffs)

    return info


def mu_poly2(x: af64, mu_ab: af64) -> af64:
    a = mu_ab[0]
    b = mu_ab[1]
    return 1.0 / (1.0 + ((x - a) / b) ** 2)


def mu_expsq2(x: af64, mu_ab: af64) -> af64:
    a = mu_ab[0]
    b = mu_ab[1]
    return np.exp(-(((x - a) / b) ** 2.0))


def extract_mu_from_args(var_args: af64, pts: af64, variables_info: VariablesInfo) -> af64:
    mu_vars = np.zeros((pts.shape[0],variables_info.n_membership_fcns))
    for col_idx in range(pts.shape[1]):
        for ivar,mu_idxs in enumerate(variables_info.mu_indexes):
            mu_coeff = var_args[mu_idxs]
            mu_vars[:,col_idx] = mu_poly2(pts[:,col_idx], mu_coeff)
            # TODO - Why doesn't exp work?
            # mu_vars[:,col_idx] = mu_expsq2(pts[:,col_idx], mu_coeff)
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
    # If constrained to classifier, round to nearest option
    # Compute the RMS error
    p = 4.0
    rms_error = np.power(np.mean((z_defuzzy - pts[:, -1]) ** p),1.0/p)
    return rms_error, z_defuzzy


def main():
    dataset = load_data()
    peak_data = kernel_density_partitioning(dataset.train_data, dataset.labels)
    variables_info = create_tsk_variables(peak_data)
    print("Number of domain variables:", len(variables_info.variables))

    # Here is the goal-seeking function
    nfev = 0
    min_err = 1.0E10
    with tqdm() as pbar:
        def fuzzy_test2(x: np.ndarray) -> f64:
            nonlocal nfev, min_err
            rms_err, _ = compute_fuzzy_system(x, dataset.train_data, variables_info)
            # This is an inversion of control, but `scipy.optimize.minimize` has limitations
            min_err = min(rms_err, min_err)
            nfev += 1
            pbar.update(nfev)
            pbar.set_description(f"err={min_err:.2%}")

            return rms_err

        # best_soln, soln_history = solve_aco(fuzzy_test2, variables_info.variables,
        #                                     joblib_n_procs=1, solution_archive_size=50,num_ants=30)
        best_soln, soln_history = solve_gradiant(fuzzy_test2, variables_info.variables)

        rms_err_test, test_result = compute_fuzzy_system(best_soln, dataset.test_data, variables_info)
        print(f"NORMALIZED RMS: Train error={soln_history[-1]:.1%}, Test error={rms_err_test:.1%}")

        # Show the plot of test result vs defuzzified test result (ideally they overlap)
        plt.figure()
        plt.title("Test Output Comparison")
        plt.plot(dataset.test_data[:,-1],'r',label="test data")
        plt.plot(test_result,'b',label="predicted data")
        plt.xlabel("Sample")
        plt.legend()
        plt.ylabel("Normalized Value")

        # Produce the denormalized result and show that!
        denorm_test_result = dataset.unscale_data(test_result, index=-1)
        denorm_test_data = dataset.unscale_data(dataset.test_data[:,-1], index=-1)

        # Compute average, max error
        err = np.abs(denorm_test_result - denorm_test_data) / denorm_test_data
        print(f"AVERAGE ERROR: {np.mean(err):.1%}")
        print(f"MAX ERROR: {np.max(err):.1%}")

        plt.figure()
        plt.title("True Units Comparison")
        plt.plot(denorm_test_data, 'r', label="test data")
        plt.plot(denorm_test_result, 'b', label="predicted data")
        plt.xlabel("Sample")
        plt.ylabel("True Units Error")
        plt.show()


if __name__ == "__main__":
    main()