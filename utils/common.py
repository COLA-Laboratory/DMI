import numpy as np
from sklearn.cluster import KMeans

from scipy.spatial.distance import chebyshev

def findKBest(X, n_clusters, value=None, previous_optima_at0=False):
    n_sample = X.shape[0]
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(X)
    if value is None:
        all_dis = np.min(cluster.transform(X),axis=1)
    else:
        all_dis = value
    ret = []
    for i in range(n_clusters):
        best_dis = np.inf
        best = -1
        for j in range(n_sample):
            if cluster.labels_[j] == i and all_dis[j] < best_dis:
                best_dis = all_dis[j]
                best = j
        ret.append(best)
    if previous_optima_at0 is True:
        ret[cluster.labels_[0]] = 0
    return ret, X[ret, :]


def tchebycheff(x,w,ideal=None,normalize=False):
    """
    :param x:  data points np array with (1, n_var) or (n_sample, n_var)
    :param w:  weights np array with (1, n_var) or (n_sample, n_var)
    :param ideal:
    :param normalize:
    :param return_index:
    :return: np array with (n_sample, )
    """
    x = np.atleast_2d(x)
    w = np.atleast_2d(w)

    n_sample = x.shape[0]
    n_weight = w.shape[0]

    if n_sample == 1 and n_weight != 1:
        x = np.tile(x,(n_weight,1))
    if n_weight == 1 and n_sample != 1:
        w = np.tile(w,(n_sample,1))

    if ideal is None:
        ideal = np.zeros((1,x.shape[1]))
    if normalize:
        norm_x = (x - x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)) - ideal
    else:
        norm_x = x - ideal

        return np.expand_dims(np.max(norm_x * w,axis=1),-1)


def weighted_sum(x,w,ideal=None,normalize=False):
    x = np.atleast_2d(x)
    w = np.atleast_2d(w)
    n_sample = x.shape[0]
    n_weight = w.shape[0]
    if n_sample == 1 and n_weight != 1:
        x = np.tile(x,(n_weight,1))
    if n_weight == 1 and n_sample != 1:
        w = np.tile(w,(n_sample,1))

    if ideal is None:
        ideal = np.zeros((1,x.shape[1]))
    if normalize:
        norm_x = (x - np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0)) - ideal
    else:
        norm_x = x - ideal
    return np.expand_dims(np.sum(norm_x * w,axis=1),-1)


def find_pareto_front(y, return_index=False, eps=1e-8):
    if len(y) == 0:
        return np.array([])
    sorted_indices = np.argsort(y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((y[idx] - y > -eps).all(axis=1), (y[idx] - y > eps).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = np.atleast_2d(y[pareto_indices].copy())

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front


def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))

def get_grid(dim):
    grids_list = {  # only support obj = 3/5/8
        1: np.expand_dims(np.linspace(0, 1, 2500), -1),
        2: np.mgrid[0:1:50j, 0:1:50j].reshape(2, -1).T,
        3: np.mgrid[0:1:10j, 0:1:10j, 0:1:8j].reshape(3, -1).T,
        4: np.mgrid[0:1:8j, 0:1:8j, 0:1:8j, 0:1:8j].reshape(4, -1).T,
        5: np.mgrid[0:1:7j, 0:1:7j, 0:1:7j, 0:1:7j, 0:1:7j].reshape(5, -1).T,
        6: np.mgrid[0:1:6j, 0:1:6j, 0:1:6j, 0:1:6j, 0:1:6j, 0:1:6j].reshape(6, -1).T,
        7: np.mgrid[0:1:5j, 0:1:5j, 0:1:5j, 0:1:5j, 0:1:5j, 0:1:5j, 0:1:5j].reshape(7, -1).T
    }
    return grids_list[dim]