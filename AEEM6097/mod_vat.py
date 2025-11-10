import numba
import numpy as np
from sklearn.metrics import pairwise_distances

from AEEM6097.prim_graph import vat_prim_mst


@numba.jit
def compute_ordered_dis_njit2(matrix_of_pairwise_distance: np.ndarray):
    # Step 1 :
    N = matrix_of_pairwise_distance.shape[0]
    list_of_int = np.zeros(N, dtype="int")

    index_of_maximum_value = np.argmax(matrix_of_pairwise_distance)

    column_index_of_maximum_value = (index_of_maximum_value // N)

    list_of_int[0] = column_index_of_maximum_value

    K = np.linspace(0,N - 1,N).astype(np.int32)

    J = np.delete(K, column_index_of_maximum_value)

    # mst_pairs = []

    for r in range(1, N):
        p, q = (-1, -1)
        mini = np.max(matrix_of_pairwise_distance)

        for candidate_p in list_of_int[0:r]:
            for candidate_j in J:
                if matrix_of_pairwise_distance[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = matrix_of_pairwise_distance[p, q]

        list_of_int[r] = q
        ind_q = np.where(J == q)[0][0]
        J = np.delete(J, ind_q)

        # mst_pairs.append((p,q))

    # Step 3
    ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)

    for column_index_of_maximum_value in range(N):
        for j in range(N):
            ordered_matrix[
                column_index_of_maximum_value, j
            ] = matrix_of_pairwise_distance[
                list_of_int[column_index_of_maximum_value], list_of_int[j]
            ]
    # print("Max Col: ", list_of_int[0])
    # print("Sequence: ", mst_pairs)

    # Step 4 :
    return ordered_matrix, list_of_int


@numba.jit
def compute_ordered_dis_njit_merge(matrix_of_pairwise_distance: np.ndarray):
    N = matrix_of_pairwise_distance.shape[0]
    ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)
    p: list[int] = vat_prim_mst(matrix_of_pairwise_distance)
    # Replace the only "-1" with the index of the maximum value.
    p = np.array(p).astype(np.int32)
    # Step 3:
    for column_index_of_maximum_value in range(N):
        for j in range(N):
            ordered_matrix[
                column_index_of_maximum_value, j
            ] = matrix_of_pairwise_distance[
                p[column_index_of_maximum_value], p[j]
            ]

    # Step 4 :
    return ordered_matrix, p


def compute_merge_sort_dissimilarity_matrix(x: np.ndarray) -> np.ndarray:
    # Copy the input matrix
    n = x.shape[0]
    A = x
    B = x.copy()
    top_down_split_merge(0, n - 1, B, A)
    return A  # TODO - Index ordering return, please.


def top_down_split_merge(begin, end, A, B):
    if begin >= end:
        return
    mid = (end + begin) // 2
    top_down_split_merge(begin, mid, A, B)
    top_down_split_merge(mid + 1, end, A, B)
    top_down_merge(begin, mid, end, A, B)


def top_down_merge(begin, mid, end, A, B):
    n1 = mid - begin + 1
    n2 = end - mid

    ij = 0
    jk = 0
    kl = begin

    while ij < n1 and jk < n2:
        did_operation = False
        ix = begin + ij
        iy = mid + 1 + jk
        try:
            for offset in range(0, n2):
                ix1 = ix + offset
                iy1 = iy + offset
                if ix1 <= mid and iy1 <= end and B[ix1, ix] < B[iy1, iy]:
                    A[kl, :] = B[ix, :]
                    A[:, kl] = B[:, ix]
                    ij += 1
                    did_operation = True
                    break
                elif ix1 <= mid and iy1 <= end and B[ix1, ix] > B[iy1, iy]:
                    A[kl, :] = B[jk, :]
                    A[:, kl] = B[:, jk]
                    jk += 1
                    did_operation = True
                    break
                else:
                    continue
        except IndexError:
            pass
        finally:
            if not did_operation:
                A[kl, :] = B[ix, :]
                A[:, kl] = B[:, ix]
                ij += 1

        kl += 1

    while ij < n1:
        A[kl, :] = B[ij + begin, :]
        A[:, kl] = B[:, ij + begin]
        ij += 1
        kl += 1

    while jk < n2:
        A[kl, :] = B[jk + mid + 1, :]
        A[:, kl] = B[:, jk + mid + 1]
        jk += 1
        kl += 1

    pass


def top_down_merge_2d(begin, mid, end, A, direction=0):
    n0 = end - begin
    n1 = mid - begin + 1
    n2 = end - mid

    L = np.zeros((n1, n1))
    R = np.zeros((n2, n2))
    # Copy from the A matrix into the source arrays.
    np.copyto(L, A[begin:mid + 1, begin:mid + 1])
    np.copyto(R, A[mid:end, mid:end])

    ij = 0
    jk = 0
    kl = begin
    # NOTE - This is an individual operation
    # Merge both halves if you can.
    while ij < n1 and jk < n2:
        ij_c, ij_r, jk_c, jk_r, kl_c, kl_r = _get_2d_offset(ij, jk, kl, n0, n1, n2)

        if L[ij_r, ij_c] <= R[jk_r, jk_c]:
            A[kl_r, kl_c] = L[ij_r, ij_c]
            ij += 1
        else:
            A[kl_r, kl_c] = R[jk_r, jk_c]
            jk += 1
        kl += 1

    # Copy remaining elements from the left half.
    while ij < n1:
        ij_c, ij_r, jk_c, jk_r, kl_c, kl_r = _get_2d_offset(ij, jk, kl, n0, n1, n2)
        A[kl_r, kl_c] = L[ij_r, ij_c]
        ij += 1
        kl += 1
    # Copy remaining elements from the right half.
    while jk < n2:
        ij_c, ij_r, jk_c, jk_r, kl_c, kl_r = _get_2d_offset(ij, jk, kl, n0, n1, n2)
        A[kl_r, kl_c] = R[jk_r, jk_c]
        jk += 1
        kl += 1


def _get_2d_offset(ij: int, jk: int, kl: int, n0: int, n1: int, n2: int) -> tuple[int, int, int, int, int, int]:
    ij_r = ij // n1
    ij_c = ij % n1
    jk_r = jk // n2
    jk_c = jk % n2
    kl_r = kl // n0
    kl_c = kl % n0
    return ij_c, ij_r, jk_c, jk_r, kl_c, kl_r


def compute_ordered_dissimilarity_matrix2(x: np.ndarray) -> tuple[np.ndarray, list]:
    matrix_of_pairwise_distance = pairwise_distances(x)
    dis_matrix, observation_path = compute_ordered_dis_njit2(matrix_of_pairwise_distance)
    return dis_matrix, observation_path


def compute_ivat_ordered_dissimilarity_matrix2(x: np.ndarray):
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
    ordered_matrix, observation_path = compute_ordered_dissimilarity_matrix2(x)
    re_ordered_observation_path = observation_path.copy()
    n_features = ordered_matrix.shape[0]
    re_ordered_matrix = ordered_matrix.copy()

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
    op_r = re_ordered_observation_path[r]
    re_ordered_observation_path[r] = re_ordered_observation_path[j]
    re_ordered_observation_path[j] = op_r
    return re_ordered_observation_path
