import os
import numpy as np
import networkx as nx
from copy import deepcopy


def preprocess_y(y_in, data_type):

    """
    input:
    - Y: numpy array for Entity-to-feature
    - nscf: Dict. the dict-key is the index of categorical variable V_l in Y, and dict-value is the number of
    sub-categorie b_v (|V_l|) in categorical feature V_l.

    App_ly Z-scoring, preprocessing by range, and Prof. Mirkin's 3-stage pre-processing methods.
    return: Original entity-to-feature data matrix, Z-scored preprocessed matrix, 2-stages preprocessed matrix,
    3-stages preprocessed matrix and their coresponding relative contribution
    """

    TY = np.sum(np.multiply(y_in, y_in))  # data scatter, the letter T stands for data scatter
    TY_v = np.sum(np.multiply(y_in, y_in), axis=0)  # feature scatter
    y_rel_cntr = TY_v / TY  # relative contribution

    Y_mean = np.mean(y_in, axis=0)
    Y_std = np.std(y_in, axis=0)

    y_z = np.divide(np.subtract(y_in, Y_mean), Y_std)  # Z-score

    Ty_z = np.sum(np.multiply(y_z, y_z))
    Ty_z_v = np.sum(np.multiply(y_z, y_z), axis=0)
    y_z_rel_cntr = Ty_z_v / Ty_z

    # scale_min_Y = np.min(y_in, axis=0)
    # scale_max_Y = np.max(y_in, axis=0)
    # rng_Y = scale_max_Y - scale_min_Y
    rng_Y = np.ptp(y_in, axis=0)

    y_rng = np.divide(np.subtract(y_in, Y_mean), rng_Y)  # 3 steps pre-processing (Range-without follow-up division)
    Ty_rng = np.sum(np.multiply(y_rng, y_rng))
    Ty_rng_v = np.sum(np.multiply(y_rng, y_rng), axis=0)
    y_rng_rel_cntr = Ty_rng_v / Ty_rng

    # This section is not used for sy_nthetic data, because no categorical data is generated.
    y_rng_rs = deepcopy(y_rng)  # 3 steps preprocessing (Range-with follow-up division)

    nscf = {}
    if data_type.lower() == 'c':
        for i in range(y_in.shape[1]):
            nscf[str(i)] = len(set(y_in[:, i]))

        col_counter = 0
        for k, v in nscf.items():
            col_counter += v
            if int(k) == 0:
                y_rng_rs[:, 0: col_counter] = y_rng_rs[:, 0: col_counter] / np.sqrt(int(v))
            if 0 < int(k) < y_in.shape[1]:
                y_rng_rs[:, col_counter: col_counter + v] = y_rng_rs[:, col_counter: col_counter + v] / np.sqrt(int(v))
            if int(k) == y_in.shape[1]:
                y_rng_rs[:, col_counter:] = y_rng_rs[:, col_counter:] / np.sqrt(int(v))

    # y_rng_rs = (Y_rescale - Y_mean)/ rng_Y
    Ty_rng_rs = np.sum(np.multiply(y_rng_rs, y_rng_rs))
    Ty_rng_v_rs = np.sum(np.multiply(y_rng_rs, y_rng_rs), axis=0)
    y_rng_rel_cntr_rs = Ty_rng_v_rs / Ty_rng_rs

    # y_rng, y_rng_rel_cntr
    # Yin, Y_rel_cntr, Yz, Yz_rel_cntr, Yrng_rs, Yrng_rel_cntr_rs

    return y_in, y_rel_cntr, y_z, y_z_rel_cntr, y_rng, y_rng_rel_cntr,


def preprocess_p(p):

    """
    input: Adjacency matrix
    App_ly Uniform, Modularity, Lapin preprocessing methods.
    return: Original Adjanceny matrix, Uniform preprocessed matrix, Modularity preprocessed matrix, and
    Lapin preprocessed matrix and their coresponding relative contribution
    """
    N, V = p.shape
    p_sum_sim = np.sum(p)
    p_ave_sim = np.sum(p) / N * (V - 1)
    cnt_rnd_interact = np.mean(p, axis=1)  # constant random interaction

    # Uniform method
    p_u = p - cnt_rnd_interact
    p_u_sum_sim = np.sum(p_u)
    p_u_ave_sim = np.sum(p_u) / N * (V - 1)

    # Modularity method (random interaction)
    p_row = np.sum(p, axis=0)
    p_col = np.sum(p, axis=1)
    p_tot = np.sum(p)
    rnd_interact = np.multiply(p_row, p_col) / p_tot  # random interaction formula
    p_m = p - rnd_interact
    p_m_sum_sim = np.sum(p_m)
    p_m_ave_sim = np.sum(p_m) / N * (V - 1)

    # Lapin (Lap_lacian Inverse Transform)
    # Lap_lacian
    """
    r, c = P.shape
    P = (P + P.T) / 2  # to warrant the symmetry
    Pr = np.sum(P, axis=1)
    D = np.diag(Pr)
    D = np.sqrt(D)
    Di = LA.p_inv(D)
    L = eye(r) - Di @ P @ Di

    # pseudo-inverse transformation
    L = (L + L.T) / 2
    M, Z = LA.eig(L)  # eig-val, eig-vect
    ee = np.diag(M)
    print("ee:", ee)
    ind = list(np.nonzero(ee > 0)[0])  # indices of non-zero eigenvalues
    Zn = Z[ind, ind]
    print("Z:", Z)
    print("M:")
    print(M)
    print("ind:", ind)
    Mn = np.diag(M[ind])  # previously: Mn =  np.asarray(M[ind])
    print("Mn:", Mn)
    Mi = LA.inv(Mn)
    p_l = Zn@Mi@Zn.T
    """
    g = nx.from_numpy_array(p)
    g_p_l = nx.laplacian_matrix(g)
    p_l = np.asarray(g_p_l.todense())
    p_l_sum_sim = np.sum(p_l)
    p_l_ave_sim = np.sum(p_l) / N * (V - 1)

    return p, p_sum_sim, p_ave_sim, p_u, p_u_sum_sim, p_u_ave_sim, p_m, p_m_sum_sim, \
        p_m_ave_sim, p_l, p_l_sum_sim, p_l_ave_sim


def flat_cluster_results(cluster_results):

    labels_pred_indices = []
    for k, v in cluster_results.items():
        labels_pred_indices += [i for i in v]

    labels_pred = np.zeros(len(labels_pred_indices))
    for k, v in cluster_results.items():
        for vv in v:
            labels_pred[vv] = k

    return labels_pred, labels_pred_indices


def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval + vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices

