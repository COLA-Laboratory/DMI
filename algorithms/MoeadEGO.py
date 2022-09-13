import GPy, GPyOpt
import numpy as np

from algorithms.multiobjective_optimization import MultiObjectiveOptimization
from utils.weights import init_weight
from utils.common import findKBest
from revision.multiobjective_bayesian_optimization import MultiObjectiveBayesianOptimization
from revision.weighted_gpmodel import WeightedGPModel
from revision.multiobjective_EI import MultiObjectiveAcquisitionEI


class MoeadEGO(MultiObjectiveOptimization):
    def __init__(self, args):
        super().__init__(args)
        if not hasattr(args, 'n_weight') or args.n_weight is None:
            self.n_weight = args.n_iter/2
        else:
            self.n_weight = args.n_weight
        if not hasattr(args, 'batch') or args.batch is None:
            self.n_batch = args.n_weight
        else:
            self.n_batch = args.batch
        self.weight = init_weight(self.n_obj, self.n_weight)
        self.n_weight = self.weight.shape[0]

        if self.n_batch > self.n_weight:
            self.n_batch = self.n_weight
        # 1. Generate model(s)
        ## MoeadEGO only need n_weight GP
        self.model = []
        for i in range(self.n_weight):

            self.model.append(WeightedGPModel(kernel=GPy.kern.RBF(input_dim=self.n_var), verbose=False, exact_feval=True))

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

    def _after_init(self, optimizer):
        super()._after_init(optimizer)
        for i in range(self.n_weight):
            optimizer.model[i].set_weight(np.expand_dims(self.weight[i, :], 0))
        return

    def _user_define_propose(self,optimizer):
        optimizer.suggested_sample = []
        optimizer.callback_parameter['acquisition_value'] = []
        for i in range(optimizer.n_model):
            optimizer.acquisition.set_output_dim(i)
            suggested_sample = optimizer._compute_next_evaluations()
            optimizer.suggested_sample.append(suggested_sample)
            optimizer.callback_parameter['acquisition_value'].append(optimizer.acquisition.acquisition_function(suggested_sample))
        optimizer.suggested_sample = np.vstack(optimizer.suggested_sample)
        return

    def _after_propose(self, optimizer):
        super()._after_propose(optimizer)
        n_batch = self.n_batch
        if n_batch + optimizer.num_acquisitions > optimizer.max_iter:
            n_batch = optimizer.max_iter - optimizer.num_acquisitions
        X = optimizer.suggested_sample

        if X.shape[0] > self.n_batch:
            acquisition_value = np.vstack(optimizer.callback_parameter['acquisition_value'])
            _, optimizer.suggested_sample = findKBest(X, n_batch, acquisition_value)
        return