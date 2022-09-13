import math
import numpy as np


def _set_weight(w, c, v, unit, s, n_obj, dim):
    if dim == n_obj:
        v = np.zeros(shape=(n_obj, 1))
    if dim == 1:
        c = c + 1
        v[0] = unit - s
        w[:, c - 1] = v[:, 0]
        return w, c

    for i in range(unit - s + 1):
        v[dim - 1] = i
        w, c = _set_weight(w, c, v, unit, s + i, n_obj, dim - 1)
    return w, c


def _no_weight(unit, s, dim):
    m = 0
    if dim == 1:
        m = 1
        return m
    for i in range(unit - s + 1):
        m = m + _no_weight(unit, s + i, dim - 1)
    return m


def init_weight(n_obj, n_sample):
    if n_obj == 1:
        return np.expand_dims(np.linspace(0,1,n_sample),-1)
    u = math.floor(math.pow(n_sample, 1.0 / (n_obj - 1))) - 2

    m = 0
    while m < n_sample:
        u = u + 1
        m = _no_weight(u, 0, n_obj)
    if m != n_sample:
        print(f'Warning number of weights {n_sample} except {m}!')
    w = np.zeros(shape=(n_obj, m))
    c = 0
    v = np.zeros(shape=(n_obj, 1))
    w, c = _set_weight(w, c, v, u, 0, n_obj, n_obj)
    w = w / (u + 0.0)
    return w.T