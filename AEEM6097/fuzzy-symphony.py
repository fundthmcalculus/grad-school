from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy.optimize import minimize
from numpy.typing import NDArray
from fcmeans import FCM

from AEEM6097.mod_vat import compute_ivat_ordered_dissimilarity_matrix2

# Some typing information shorthand!
f64 = np.float64
af64 = NDArray[f64]

# Handle different numbers of membership functions!
N_mu = 3
# TODO - Don't be caching the relevant evaluation domain
X_min = -15
X_max = 15
mu_b = (X_max - X_min) / (2*N_mu+1)
x_eval = np.linspace(X_min,X_max,200)

def mu(x: af64, a:f64, b: f64) -> af64:
    """Membership function, use the gaussian quadratic"""
    # return quadinv(x,a,b)
    return expsq(x,a,b)

def quadinv(x: af64, a:f64, b: f64) -> af64:
    return 1.0/(1.0+((x-a)/b)**2.0)

def f(x: af64, a: f64, b:f64) -> af64:
    return a*x+b

# Mmmmm, I love Python magic
mu_lst = [mu]*N_mu
f_lst = [f]*N_mu

def expsq(x: af64, a: f64, b: f64) -> af64:
    """exponential quadratic membership function"""
    return np.exp(-((x-a)/b)**2.0)

def expsq_dx(x: af64, a: f64, b: f64) -> af64:
    """exponential quadratic membership function first derivative"""
    return expsq(x, a, b)*-2.0*(x-a)/b

# From Dr Cohen's fuzzy symphony notes
def y_def(x: af64) -> af64:
    # return 0.3*x**3+0.2*x**2-5*x-3.0
    # Ackley function
    return -20.0*np.exp(-0.2*np.sqrt(0.5*x**2))-np.exp(0.5*np.cos(2*np.pi*x))+np.e+20


def y_crisp(x: af64, mu_x: list[Callable[[af64,f64,f64],af64]], f_x: list[Callable[[af64,f64,f64],af64]],args: af64) -> af64:
    # Weighted average of TSK functions
    n_fcns = len(f_x)
    # NOTE - I know this isn't the right actual type, but it gets overwritten anyway _shrugs_
    y_num: af64 = 0.0
    mu_denom: af64 = 0.0
    N_args = len(args)//n_fcns

    for idx in range(n_fcns):
        a_mu_i = args[N_args*idx]
        b_mu_i = args[N_args*idx+1]
        a_f_i = args[N_args*idx+2]
        b_f_i = args[N_args*idx+3]
        mu_i = mu_x[idx](x,a_mu_i,b_mu_i)
        f_i = f_x[idx](x,a_f_i,b_f_i)
        y_num += mu_i*f_i
        mu_denom += mu_i

    return y_num/mu_denom

def J(args: af64) -> f64:
    """This is the fitness function that is the integrated error"""
    return np.sum((y_def(x_eval)-y_crisp(x_eval,mu_lst, f_lst, args))**2.0)


def main_ackley(min_fcn) -> None:
    # Figure out the optimal a1,b1,a2,b2,a3,b3 that fit the solution!
    # NOTE - We are trying to minimize J = int(from=x_low to=x_high of y_d - y_crisp dx)
    x0 = [
        -10, # mu1-a1
        5,   # mu1-b1
        3,   # f1-a1
        -10, # f1-b1 and so on...
    ]
    bounds = [
        (X_min,X_max), # mu_a
        (0.01,10), # mu_b
        (-10,10), # f_a
        (-1000,1000), # f_b
    ]
    n_args = len(x0)
    # Replicate for each arg set
    x0 = x0 * N_mu
    bounds = bounds * N_mu
    # Locate the membership functions a bit better
    mu_a = np.r_[X_min:X_max:mu_b]
    for ij in range(N_mu):
        x0[ij*n_args] = mu_a[2*ij+1]

    res = minimize(min_fcn, np.array(x0), bounds=bounds)
    print("Result Information:",res)
    x1 = res.x
    # See if this is an optima, the Hessian matrix should be positive definite
    # L-BFGS-B returns an operator of inverse hessian, so:
    # (iH*I)^-1 = H
    hess = np.linalg.inv(res.hess_inv.matmat(np.eye(len(x1))))
    # It is positive definite if all eigenvalues are positive
    eigs = np.linalg.eigvalsh(hess)
    print(f"Positive eigs={np.all(eigs > 0.0)}")

    # Plot
    fig, axs = plt.subplots(2,1)
    axs[0].plot(x_eval,y_def(x_eval), label='y_d')
    axs[0].plot(x_eval,y_crisp(x_eval, mu_lst, f_lst, x1), label='y_crisp')
    axs[0].legend()
    axs[0].set(xlabel='x', ylabel='y')
    axs[0].set_title('Comparison of Defined Function `y_d` and `y_crisp`')
    # Plot the membership functions
    for ij, mu_i in enumerate(mu_lst):
        axs[1].plot(x_eval, mu_i(x_eval,x1[n_args*ij],x1[n_args*ij+1]), label=f'mu_{ij}')
    axs[1].legend()
    axs[1].set(xlabel='x', ylabel='mu(x)')
    axs[1].set_title('Membership Functions')
    plt.show()


def main_sonar() -> None:
    # Load the CSV file
    sonar_data = np.genfromtxt("project-data/sonar.all-data", delimiter=",")
    # Ignore the last column. Since that is the classification value, we will deal with that later
    sonar_data = sonar_data[:,:60]
    sonar_txt_data = np.genfromtxt("project-data/sonar.all-data", delimiter=",", dtype=str)
    sonar_classification_col = sonar_txt_data[:,60]
    # Convert "R" to a 1 in the rock col, "M" to a 1 in the mine col
    rock_col: np.ndarray = (sonar_classification_col == "R").astype(float)
    mine_col: np.ndarray = (sonar_classification_col == "M").astype(float) * -1.0
    # Now that one is "1", and the other is "-1", we can have confidence for each?
    output_col: np.ndarray = rock_col + mine_col

    # NOTE - There is little in the way of clustering, so we can ignore that.
    sonar_full_data = np.hstack((sonar_data, np.reshape(output_col, (len(output_col), 1))))
    # Permute the rows randomly
    sonar_full_data = np.random.permutation(sonar_full_data)

    plot_ivat(sonar_full_data)

    # Use the fuzzy C-means
    # Perform the fuzzy c-means clustering
    fcm = FCM(n_clusters=2, m=4, max_iter=100, error=0.0001, n_jobs=1)
    error_history = fcm.fit(sonar_full_data)
    plot_error_history(error_history)
    # fcm_centers = fcm.centers
    # print("Fuzzy C-Means Cluster Centers:\n", fcm_centers)
    membership_degree = fcm.soft_predict(sonar_full_data)
    # Order the entries so that the cluster-1 are front of dataset, cluster-2 end of the dataset.
    cluster_number = np.argmax(membership_degree, axis=1)
    member_order = np.argsort(cluster_number)
    # sonar_full_data = sonar_full_data[member_order, :]
    membership_degree = membership_degree[member_order, :]
    # Plot the membership degree for the clusters
    plot_membership_degree(membership_degree, fcm)

    # Plot the linear spectra
    plot_lin_spec(sonar_full_data)
    # Use the cluster centers as the two rules?
    plt.show()


def plot_lin_spec(x: np.ndarray) -> None:
    x_cat = x[:,60:]
    x = x[:,:60]
    x_spec = lin_spec(x)
    # Normalize along the second axis
    x_spec = x_spec / np.max(np.abs(x_spec),axis=1, keepdims=True)
    # Now, plot in two blocks on the same axis: Those which are category == 1.0, those which are category == -1.0
    rock_indices = x_cat.flatten() == 1.0
    mine_indices = x_cat.flatten() == -1.0
    plt.figure()
    plt.semilogy(np.abs(x_spec[rock_indices]).T, 'b', alpha=0.3, label='Rock')
    plt.semilogy(np.abs(x_spec[mine_indices]).T, 'r', alpha=0.3, label='Mine')
    plt.title('Linear Spectra')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend(['Cluster Blue (Rock)', 'Cluster Red (Mine)'])
    plt.show()


def lin_spec(x: np.ndarray) -> np.ndarray:
    N = x.shape[1]
    X = fft.fft(x, axis=1)
    # Take the positive frequencies
    X = X[:,:N//2]
    # Double all but the DC term to preserve energy
    X[1:] *= 2
    return X


def plot_ivat(sonar_full_data):
    # Plot the clustering using IVAT
    ivat_sonar_data, _, _, _ = compute_ivat_ordered_dissimilarity_matrix2(sonar_full_data)
    fig = plt.figure()
    plt.imshow(ivat_sonar_data)
    plt.title('Sonar Data IVAT')


def plot_error_history(error_history):
    # Plot the error history
    fig2 = plt.figure()
    plt.semilogy(error_history)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Fuzzy C-Means Error History')


def plot_membership_degree(membership_degree, fcm: FCM) -> None:
    plt.figure()
    plt.plot(membership_degree)
    plt.title(f'Membership Degree for clusters k={fcm.n_clusters}, m={fcm.m}')
    plt.ylabel('Membership Degree')
    plt.xlabel('Data Points')
    # =Handle any number of clusters
    plt.legend([f'Cluster {ij+1}' for ij in range(len(membership_degree))])


if __name__ == "__main__":
    main_sonar()
    # main_ackley(J)
