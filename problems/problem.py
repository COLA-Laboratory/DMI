from abc import abstractmethod

import numpy as np


class Problem:
    def __init__(self,args):
        r"""
            args is a namespace includes:
                seed: int
                    The random seed
                n_var: int
                    The number of variables
                n_obj: int
                    The number of objectives
                n_const: int
                    The number of constraints
                var_bound: numpy.array
                    The bound for variables
                    n_var * 2 array, with lower bound in x_bound[i,0] and upper bound in x_bound[i,1]
                return_list: list of strings
                    Provide a list of strings which defines the values that are returned.

                    Allowed is ["F", "CV", "dF", "dCV", "hF", "hCV", "feasible"] where the d stands for derivative and
                            h stands for hessian matrix.
        """
        self.seed = args.seed
        self.n_var = args.n_var
        self.n_obj = args.n_obj
        self.n_const = args.n_const
        self.ideal_point = np.zeros((1,self.n_obj)) - 0.1
        self.nadir_point = np.ones((1, self.n_obj)) + 0.1

        if not hasattr(args, 'var_bound') or args.var_bound is None:
            self.var_bound = np.array([[0.0, 1.0]] * args.n_var)
        else:
            np.array(args.n_var_bound).reshape(args.n_var, 2)

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def get_pareto_front(self, n_sample=100):
        pass

    @abstractmethod
    def get_pareto_set(self, n_sample=100):
        pass

    def get_ideal_point(self):
        return self.ideal_point

    def get_nadir_point(self):
        return self.nadir_point
