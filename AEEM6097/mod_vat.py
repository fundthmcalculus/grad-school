import numpy as np
from sklearn.metrics import pairwise_distances


def compute_ordered_dis_njit(matrix_of_pairwise_distance: np.ndarray):  # pragma: no cover
    """
    The ordered dissimilarity matrix is used by visual assessment of tendency. It is a just a a reordering
    of the dissimilarity matrix.


    Parameter
    ----------

    x : matrix
        numpy array

    Return
    -------

    ODM : matrix
        the ordered dissimilarity matrix

    """

    # Step 1 :

    distance_shape_ = matrix_of_pairwise_distance.shape[0]
    observation_path = np.zeros(distance_shape_, dtype="int")

    list_of_int = np.zeros(distance_shape_, dtype="int")

    index_of_maximum_value = np.argmax(matrix_of_pairwise_distance)

    column_index_of_maximum_value = (
            index_of_maximum_value // distance_shape_
    )

    list_of_int[0] = column_index_of_maximum_value
    observation_path[0] = column_index_of_maximum_value

    K = np.linspace(
        0,
        distance_shape_ - 1,
        distance_shape_,
    ).astype(np.int32)

    J = np.delete(K, column_index_of_maximum_value)

    for r in range(1, distance_shape_):

        p, q = (-1, -1)

        mini = np.max(matrix_of_pairwise_distance)

        for candidate_p in observation_path[0:r]:
            for candidate_j in J:
                if matrix_of_pairwise_distance[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = matrix_of_pairwise_distance[p, q]

        list_of_int[r] = q
        observation_path[r] = q
        ind_q = np.where(J == q)[0][0]
        J = np.delete(J, ind_q)

    # Step 3

    ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)

    for column_index_of_maximum_value in range(ordered_matrix.shape[0]):
        for j in range(ordered_matrix.shape[0]):
            ordered_matrix[
                column_index_of_maximum_value, j
            ] = matrix_of_pairwise_distance[
                list_of_int[column_index_of_maximum_value], list_of_int[j]
            ]

    # Step 4 :

    return ordered_matrix, observation_path


def compute_ordered_dissimilarity_matrix(x: np.ndarray) -> tuple[np.ndarray, list]:
    matrix_of_pairwise_distance = pairwise_distances(x)
    # Append a column of the initial ordering
    dis_matrix, observation_path = compute_ordered_dis_njit(matrix_of_pairwise_distance)
    return dis_matrix, observation_path


def compute_ivat_ordered_dissimilarity_matrix(x: np.ndarray):
    """The ordered dissimilarity matrix is used by ivat. It is a just a reordering of the dissimilarity matrix.


    Parameters
    ----------

    x : matrix
        numpy array

    Return
    -------

    D_prim : matrix
        the ordered dissimilarity matrix

    """
    ordered_matrix, observation_path = compute_ordered_dissimilarity_matrix(x)
    re_ordered_observation_path = observation_path.copy()
    n_features = ordered_matrix.shape[0]
    re_ordered_matrix = np.zeros((n_features, n_features))

    for r in range(1, n_features):
        # Step 1 : find j for which D[r,j] is minimum and j ipn [1:r-1]
        j = np.argmin(ordered_matrix[r, 0:r])

        # Step 2 :

        om_rj = ordered_matrix[r, j]
        re_ordered_matrix[r, j] = om_rj
        re_ordered_matrix[j, r] = om_rj

        # Exchange path
        # exchange(j, r, re_ordered_observation_path)

        # Step 3 : for c : 1, r-1 with c !=j
        c_tab = np.arange(r)
        c_tab = c_tab[c_tab != j]

        for c in c_tab:
            rom_jc = re_ordered_matrix[j, c]
            if om_rj >= rom_jc:
                re_ordered_matrix[r, c] = om_rj
                # if r == n_features-1:
                # exchange(j, c, re_ordered_observation_path)
                # exchange(c, r, re_ordered_observation_path)
            else:
                re_ordered_matrix[r, c] = rom_jc
                # if r == n_features - 1:
                #     exchange(j, r, re_ordered_observation_path)
                exchange(j, r, re_ordered_observation_path)
            # re_ordered_matrix[r, c] = max(om_rj, rom_jc)
            re_ordered_matrix[c, r] = re_ordered_matrix[r, c]


    return re_ordered_matrix, re_ordered_observation_path, ordered_matrix, observation_path


def exchange(j, r, re_ordered_observation_path):
    rt = re_ordered_observation_path[r]
    re_ordered_observation_path[r] = re_ordered_observation_path[j]
    re_ordered_observation_path[j] = rt
