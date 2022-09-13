from problems.problem import Problem
import numpy as np
from utils.common import find_pareto_front, get_grid
from utils.weights import init_weight
import ctypes
import glob
import math

class WFG(Problem):
    def __init__(self, args):
        super().__init__(args)
        self.k = args.n_var - args.n_obj + 1
        if self.k < 2:
            raise Exception(f"Provide wrong number of objectives {args.n_obj} and var {args.n_var}, expect n_var> n_obj")
        self._lib_setup()
        self.testcase = -1
        self.pareto_front = None
        for i in range(self.n_obj):
            self.nadir_point[0,i] *= 2 * (i+1)
        self.var_bound = np.array([[0.0, 1.0]] * args.n_var)
        for i in range(self.n_var):
            self.var_bound[i,1] *= 2 * (i+1)

    def evaluate(self, x):
        n_sample = x.shape[0]
        data_in = np.reshape(x, (x.size,))
        data_in = np.array(data_in, dtype=np.float64)
        data_out = np.zeros((n_sample, self.n_obj), dtype=np.float64)
        self.lib.libWFG(n_sample, self.n_var, self.n_obj, self.testcase, data_in, data_out)
        data_out = data_out.reshape((n_sample, self.n_obj))
        return data_out

    def get_shape(self,x):
        n_sample = x.shape[0]
        data_in = np.reshape(x, (x.size,))
        data_in = np.array(data_in, dtype=np.float64)
        data_out = np.zeros((n_sample, self.n_obj), dtype=np.float64)
        self.lib.libWFGShape(n_sample, self.n_obj, self.testcase, data_in, data_out)
        data_out = data_out.reshape((n_sample, self.n_obj))
        return data_out

    def _lib_setup(self):
        self.lib = ctypes.CDLL(glob.glob('problems/clib/libWFG.so')[0])
        self.lib.libWFG.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

        self.lib.libWFG.restype = ctypes.c_int

        self.lib.libWFGShape.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

        self.lib.libWFGShape.restype = ctypes.c_int

    def get_pareto_front(self, n_sample: int = 100):
        data_in = get_grid(self.n_obj - 1)
        n_sample = data_in.shape[0]
        data_in = np.hstack((data_in,np.zeros((n_sample,1))))
        return self.get_shape(data_in)

    def get_pareto_set(self, n_sample: int = 100):
        grids = get_grid(self.n_obj - 1)
        ret = np.hstack((grids, 0.35 * np.ones((grids.shape[0], self.n_var - self.n_obj + 1))))
        for i in range(self.n_var):
            ret[:,i] *= (i+1)*2
        return ret

class WFG1(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 1

class WFG2(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 2


class WFG3(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 3


class WFG4(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 4


class WFG5(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 5


class WFG6(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 6


class WFG42(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 42


class WFG43(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 43


class WFG44(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 44


class WFG45(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 45


class WFG46(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 46


class WFG47(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 47


class WFG48(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 48


class WFG21(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 21

class WFG22(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 22

class WFG23(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 23

class WFG24(WFG):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 24