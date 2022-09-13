import GPy, GPyOpt
import numpy as np

from algorithms.multiobjective_optimization import MultiObjectiveOptimization
from utils.weights import init_weight
from revision.weighted_gpmodel import WeightedGPModel
from revision.multiobjective_bayesian_optimization import MultiObjectiveBayesianOptimization
from revision.multiobjective_EI import MultiObjectiveAcquisitionEI


class ParEGO(MultiObjectiveOptimization):
    def __init__(self, args):
        super().__init__(args)
        if not hasattr(args, 'n_weight') or args.n_weight is None:
            self.n_weight = args.n_iter
        else:
            self.n_weight = args.n_weight
        self.weight = init_weight(self.n_obj, self.n_weight)
        self.n_weight = self.weight.shape[0]
        # 1. Generate model(s)
        ## ParEGO only need 1 GP
        self.kernel = GPy.kern.RBF(input_dim=self.n_var)
        self.model = [WeightedGPModel(kernel=self.kernel, verbose=False, exact_feval=True)]

        bounds = []
        for i in range(self.n_var):
            bounds.append({'name': f'var{i}', 'type': 'continuous',
                           'domain': (self.problem.var_bound[i, 0], self.problem.var_bound[i, 1])})


        self.weight_ptr = 0
        self.weight_direction = 1

        self.optimizer = MultiObjectiveBayesianOptimization(f=self.problem.evaluate, n_obj=self.n_obj, model=self.model,
                                                            domain=bounds, initial_design_numdata = self.n_init,
                                                            acquisition=MultiObjectiveAcquisitionEI,
                                                            callback_after_init=self._after_init,
                                                            callback_before_fit=self._before_fit,
                                                            callback_after_fit=self._after_fit,
                                                            callback_before_propose=self._before_propose,
                                                            callback_user_define_propose=self._user_define_propose,
                                                            callback_after_propose=self._after_propose,
                                                            callback_end_iteration=self._end_iteration,
                                                            callback_after_finish=self._after_finish,
                                                            )

    def run(self):
        self.optimizer.run_optimization(max_iter=self.n_iter)

    def _before_fit(self,optimizer):
        super()._before_fit(optimizer)
        optimizer.model[0].set_weight(np.expand_dims(self.weight[self.weight_ptr, :], 0))
        return

    def _after_fit(self,optimizer):
        super()._after_fit(optimizer)
        self._change_weight()
        return

    def _change_weight(self):
        n_weight = self.weight.shape[0]
        if self.weight_ptr == n_weight - 1:
            self.weight_direction = -1
        if self.weight_ptr == 0:
            self.weight_direction = 1
        self.weight_ptr += self.weight_direction
