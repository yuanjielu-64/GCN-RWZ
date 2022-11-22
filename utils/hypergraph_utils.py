import numpy as np


def H_KNN_distance(dis, k_neighbor):
    H = np.zeros((dis.shape[0], dis.shape[1]))
    for idx in range(dis.shape[0]):
        dis_vec = dis[idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

        if not np.any(nearest_idx[:k_neighbor] == idx):
            nearest_idx[k_neighbor - 1] = idx

        for node_idx in nearest_idx[:k_neighbor]:
            H[node_idx, idx] = 1.0

    return H

def G_from_H(H, variable_weight):
    H = np.array(H).squeeze()
    W = np.ones(H.shape[1])
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G