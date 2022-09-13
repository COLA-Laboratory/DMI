from problems.problem import Problem
import numpy as np
from utils.common import find_pareto_front
from utils.weights import init_weight
import ctypes
import glob
import math
from utils.common import get_grid

class DTLZ(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.k = args.n_var - args.n_obj + 1
        if self.k < 2:
            raise Exception(f"Provide wrong number of objectives {args.n_obj} and var {args.n_var}, expect n_var> n_obj")
        self._lib_setup()
        self.testcase = -1
        self.pareto_front = None

    def evaluate(self, x):
        n_sample = x.shape[0]
        data_in = np.reshape(x, (x.size,))
        data_in = np.array(data_in, dtype=np.float64)
        data_out = np.zeros((n_sample, self.n_obj), dtype=np.float64)
        self.lib.libDTLZ(n_sample, self.n_var, self.n_obj, self.testcase, data_in, data_out)
        data_out = data_out.reshape((n_sample, self.n_obj))
        return data_out

    def get_pareto_front(self, n_sample: int = 100):
        return None

    def _lib_setup(self):
        self.lib = ctypes.CDLL(glob.glob('problems/clib/libDTLZ.so')[0])
        self.lib.libDTLZ.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

        self.lib.libDTLZ.restype = ctypes.c_int


class DTLZ1(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 1
        self.nadir_point *= 0.5

    def get_pareto_front(self, n_sample: int = 100):
        ref_dirs = init_weight(self.n_obj, n_sample)
        return 0.5 * ref_dirs

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,0.5*np.ones((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ2(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 2

    def get_pareto_front(self, n_sample: int = 100):
        ref_dirs = init_weight(self.n_obj, n_sample)
        return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,0.5*np.ones((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ3(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 3

    def get_pareto_front(self, n_sample: int = 100):
        ref_dirs = init_weight(self.n_obj, n_sample)
        return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,0.5*np.ones((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ4(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 4

    def get_pareto_front(self, n_sample: int = 100):
        ref_dirs = init_weight(self.n_obj, n_sample)
        return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,0.5*np.ones((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ7(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 7
        self.ideal_point[0,-1] += 2
        self.nadir_point[0,-1] = (self.nadir_point[0,-1]) * (2.0*self.n_obj)

    def get_pareto_front(self, n_sample=100):
        '''
        grids = get_grid(self.n_obj-1)
        A = 3.0
        alpha = 0.0
        beta = 1.0
        h = 2 * np.atleast_2d(self.n_obj-np.sum(grids/2.0*(1.0+np.power(grids,alpha)*np.sin(A*math.pi*np.power(grids,beta))),axis=1)).T
        # h = (h - 2.0)/(2.0*self.n_obj-2.0)
        return find_pareto_front(np.vstack((grids.T, h.T)).T)
        '''
        return self.evaluate(self.get_pareto_set(n_sample))
    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,np.zeros((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ71(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 71
        self.ideal_point[0,-1] += 2
        self.nadir_point[0,-1] = (self.nadir_point[0,-1]) * (2.0*self.n_obj)

    def get_pareto_front(self, n_sample=100):
        grids = get_grid(self.n_obj-1)
        A = 5.0
        alpha = 0.0
        beta = 1.0
        h = 2 * np.atleast_2d(self.n_obj-np.sum(grids/2.0*(1.0+np.power(grids,alpha)*np.sin(A*math.pi*np.power(grids,beta))),axis=1)).T
        # h = (h - 2.0)/(2.0*self.n_obj-2.0)
        return find_pareto_front(np.vstack((grids.T, h.T)).T)

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,np.zeros((grids.shape[0],self.n_var-self.n_obj+1))))


class DTLZ72(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 72
        self.ideal_point[0,-1] += 2
        self.nadir_point[0,-1] = (self.nadir_point[0,-1]) * (2.0*self.n_obj)

    def get_pareto_front(self, n_sample=100):
        grids = get_grid(self.n_obj-1)
        A = 3.0
        alpha = 0.0
        beta = 2.0
        h = 2 * np.atleast_2d(self.n_obj-np.sum(grids/2.0*(1.0+np.power(grids,alpha)*np.sin(A*math.pi*np.power(grids,beta))),axis=1)).T
        # h = (h - 2.0)/(2.0*self.n_obj-2.0)
        return find_pareto_front(np.vstack((grids.T, h.T)).T)

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,np.zeros((grids.shape[0],self.n_var-self.n_obj+1))))

class DTLZ73(DTLZ):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 73
        self.ideal_point[0,-1] += 2
        self.nadir_point[0,-1] = (self.nadir_point[0,-1]) * (2.0*self.n_obj)

    def get_pareto_front(self, n_sample=100):
        grids = get_grid(self.n_obj-1)
        A = 3.0
        alpha = 3.0
        beta = 1.0
        h = 2 * np.atleast_2d(self.n_obj-np.sum(grids/2.0*(1.0+np.power(grids,alpha)*np.sin(A*math.pi*np.power(grids,beta))),axis=1)).T
        # h = (h - 2.0)/(2.0*self.n_obj-2.0)
        return find_pareto_front(np.vstack((grids.T, h.T)).T)

    def get_pareto_set(self,n_sample=100):
        grids = get_grid(self.n_obj-1)
        return np.hstack((grids,np.zeros((grids.shape[0],self.n_var-self.n_obj+1))))
