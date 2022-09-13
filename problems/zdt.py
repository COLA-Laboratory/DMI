from problems.problem import Problem
import numpy as np
from utils.common import find_pareto_front
from utils.weights import init_weight
import ctypes
import glob


class ZDT(Problem):
    def __init__(self, args):
        super().__init__(args)
        if args.n_obj != 2:
            raise Exception(f"Provide wrong number of objectives {args.n_obj}, expect n_obj = 2")
        if args.n_const != 0:
            raise Exception(f"Provide wrong number of constraints {args.n_const}, expect n_const = 0")
        self._lib_setup()
        self.testcase = -1
        self.pareto_front = None

    def evaluate(self, x):
        n_sample = x.shape[0]
        data_in = np.reshape(x, (x.size,))
        data_in = np.array(data_in, dtype=np.float64)
        data_out = np.zeros((n_sample, self.n_obj), dtype=np.float64)
        self.lib.libZDT(n_sample, self.n_var, self.n_obj, self.testcase, data_in, data_out)
        data_out = data_out.reshape((n_sample, self.n_obj))
        return data_out

    def get_pareto_front(self, n_sample: int = 100):
        if self.pareto_front is None:
            x0 = init_weight(1, n_sample)
            n_sample = x0.shape[0]
            x = np.hstack((x0,np.zeros((n_sample,self.n_var-1))))
            self.pareto_front = self.evaluate(x)
        return self.pareto_front

    def get_pareto_set(self,n_sample: int = 1000):
        x0 = init_weight(1, n_sample)
        return np.hstack((x0, np.zeros((n_sample, self.n_var - 1))))

    def _lib_setup(self):
        self.lib = ctypes.CDLL(glob.glob('problems/clib/libZDT.so')[0])
        self.lib.libZDT.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

        self.lib.libZDT.restype = ctypes.c_int


class ZDT1(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 1


class ZDT2(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 2


class ZDT3(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 3
        self.ideal_point = np.array([[0,-1]]) - 0.1


class ZDT4(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 4


class ZDT6(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 6


class ZDT31(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 31
        self.ideal_point = np.array([[0, -1]]) - 0.1


class ZDT32(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 32
        self.ideal_point = np.array([[0, -1]]) - 0.1


class ZDT33(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 33
        self.ideal_point = np.array([[0, -1]]) - 0.1


class ZDT34(ZDT):
    def __init__(self, args):
        super().__init__(args)
        self.testcase = 34
        self.ideal_point = np.array([[0, -1]]) - 0.1
