import GPy
import GPyOpt
import time
import numpy as np
from pymoo.model.problem import Problem as pymooProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from algorithms.multiobjective_optimization import MultiObjectiveOptimization
from utils.weights import init_weight
from utils.common import findKBest, weighted_sum, find_pareto_front
from utils.lhs import lhs
from utils.weight_associated_samples import WeightAssociatedSamples
from revision.multiobjective_bayesian_optimization import MultiObjectiveBayesianOptimization
from revision.weighted_gpmodel import WeightedGPModel
from revision.multiobjective_MGD import MultiObjectiveAcquisitionMGD
from utils.normalizer import StandardNormalizer, BoundedNormalizer
from utils.propose_next_batch import DecomposedHVI, DecomposedDist
from utils.plot import plot2D
from utils.common import find_pareto_front
from GPyOpt.core.task.space import Design_space
from scipy.optimize import minimize
from scipy.linalg import null_space


class MGD(MultiObjectiveOptimization):

    def __init__(self, args):
        super().__init__(args)

        if not hasattr(args, 'n_weight') or args.n_weight is None:
            self.n_weight = args.n_iter/2
        else:
            self.n_weight = args.n_weight
        self.weight = init_weight(self.n_obj, self.n_weight)

        self.n_weight = self.weight.shape[0]

        if not hasattr(args, 'batch') or args.batch is None:
            self.n_batch = args.n_weight
        else:
            self.n_batch = args.batch
        if not hasattr(args, 'n_gen') or args.n_gen is None:
            self.n_gen = 3
        else:
            self.n_gen = args.n_gen
        if not hasattr(args, 'pop_size') or args.pop_size is None:
            self.pop_size = self.n_weight
        else:
            self.pop_size = args.pop_size

        if not hasattr(args, 'ea_type'):
            self.ea_type = None
        else:
            self.ea_type = args.ea_type
        if not hasattr(args, 'approximation'):
            self.approximation = True
        else:
            self.approximation = args.approximation
        if self.n_batch > self.n_weight:
            self.n_batch = self.n_weight

        if not hasattr(args, 'selection_type'):
            self.selection_type = 'DHVI'
            self.selection = DecomposedHVI()
        else:
            if args.selection_type == 'MOEAD':
                self.selection_type = 'MOEAD'
                self.selection = DecomposedDist()
            else:
                self.selection_type = 'DHVI'
                self.selection = DecomposedHVI()

        # 1. Generate model(s)
        ## MoeadEGO only need n_weight GP
        self.model = []
        for i in range(self.n_obj):
            kernel = GPy.kern.RBF(input_dim=self.n_var,ARD=True, variance=1., lengthscale=np.ones(self.n_var)) +\
                     GPy.kern.White(input_dim=self.n_var, variance=1e-2)
            kernel['.*lengthscale*'].constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3), warning=False)
            kernel['.*rbf.variance*'].constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3), warning=False)
            kernel['.*white.variance*'].constrain_bounded(np.exp(-6), np.exp(0), warning=False)

            model = WeightedGPModel(kernel=kernel,verbose=False, exact_feval=True,
                                    X_normalizer=BoundedNormalizer(self.problem.var_bound),
                                    Y_normalizer=StandardNormalizer()
                                    )
            '''
            model = WeightedRBFModel(kernel=kernel,verbose=False, exact_feval=True,
                                    X_normalizer=BoundedNormalizer(self.problem.var_bound),
                                    Y_normalizer=StandardNormalizer()
                                    )
            '''
            self.model.append(model)

        domain = []
        for i in range(self.n_var):
            domain.append({'name': f'var{i}', 'type': 'continuous',
                           'domain': (self.problem.var_bound[i, 0], self.problem.var_bound[i, 1])})

        self.space = Design_space(domain, None)
        self.eps = 1e-2
        X_init = lhs(self.n_var, self.n_init) * (self.problem.var_bound[:,1] - self.problem.var_bound[:,0]) + self.problem.var_bound[:,0]
        self.optimizer = MultiObjectiveBayesianOptimization(f=self.problem.evaluate, n_obj=self.n_obj, model=self.model,
                                                            domain=domain, initial_design_numdata = self.n_init,
                                                            X=X_init,
                                                            acquisition=MultiObjectiveAcquisitionMGD,
                                                            callback_after_init=self._after_init,
                                                            callback_before_fit=self._before_fit,
                                                            callback_after_fit=self._after_fit,
                                                            callback_before_propose=self._before_propose,
                                                            callback_user_define_propose=self._user_define_propose,
                                                            callback_after_propose=self._after_propose,
                                                            callback_end_iteration=self._end_iteration,
                                                            callback_after_finish=self._after_finish,
                                                            )

    def _user_define_propose(self, optimizer):
        def eval_f(x):
            y = []
            for i in range(self.n_obj):
                # y.append(optimizer.model[i].predict(x)[0])
                y.append(np.expand_dims(optimizer.model[i].predict_skgp(x)['F'],-1))
            return np.hstack(y)

        def eval_df(x):
            dy = []
            for i in range(self.n_obj):
                # dy.append(optimizer.model[i].predictive_gradients(x)[0])
                dy.append(optimizer.model[i].predict_skgp(x,calc_gradient=True)['dF'])
            return np.transpose(np.array(dy),(1,0,2))

        pop_x = self._get_nd_X(optimizer.X_in_model, optimizer.Y_in_model)
        pop_y = eval_f(pop_x)
        association = WeightAssociatedSamples(self.weight,self.n_var,self.n_obj)
        current_label = 0
        ideal_point = np.zeros((1,self.n_obj))
        for i in range(self.n_obj):
            ideal_point[0,i] = optimizer.model[i].Y_normalizer.do(np.zeros((1,1)))
        association.set_ideal_point(ideal_point)
        association.insert(pop_x,pop_y,current_label * np.ones((pop_x.shape[0],)))
        current_label += 1
        pop_x = association.get(self.pop_size)
        pop_y = eval_f(pop_x)
        nd_id = NonDominatedSorting().do(pop_y, only_non_dominated_front=True)
        MGD_weight = init_weight(n_obj=self.n_obj,n_sample=50)
        for g in range(self.n_gen):
            print(f'generation {g}/{self.n_gen} with FEs {optimizer.X_in_model.shape[0]}')
            xs = self._mutation(pop_x.copy())
            ys = eval_f(xs)
            new_ideal_point = np.minimum(association.ideal_point, np.min(ys, axis=0))
            if (new_ideal_point != association.ideal_point).any():
                new_ideal_point -= self.eps
            fs = ys - new_ideal_point

            xs_opt = []
            xs_idx = []
            x_opts = []

            for i in range(xs.shape[0]):
                print(f"MGD search {i}/{xs.shape[0]}")
                for w in MGD_weight:
                    x_opts.append(self._MGD_search(np.atleast_2d(xs[i]),eval_f,eval_df,self.problem.var_bound,w))

            dys_opt = eval_df(np.vstack(x_opts))
            for x_opt,dy_opt in zip(x_opts,dys_opt):
                x_approximation = np.atleast_2d(x_opt) # dummy
                xs_opt.append(x_approximation)
                xs_idx.extend([xs.shape[0] + current_label] * len(x_approximation))


            current_label += len(xs_opt)
            xs_opt = np.vstack(xs_opt)
            xs_idx = np.array(xs_idx)
            new_ideal_point = np.minimum(new_ideal_point,association.ideal_point)
            ys_opt = eval_f(xs_opt)
            new_ideal_point = np.minimum(np.atleast_2d(np.min(ys_opt, axis=0)), new_ideal_point)
            association.set_ideal_point(new_ideal_point)
            association.insert(xs_opt,ys_opt,xs_idx)
            pop_x = association.get(self.pop_size)
        pop_x, pop_y, pop_dist, pop_idx = association.get_all()
        if self.approximation:
            pop_x, pop_y, pop_dist, pop_idx = association.sparse_approximation()
        self.selection.set_nadir_point(np.ones((1,self.n_obj))*1.1)
        if self.selection_type == 'DHVI':
            optimizer.suggested_sample = self.selection.select(self.n_batch, pop_x, find_pareto_front(optimizer.Y), optimizer.model, pop_idx)

    def run(self):
        self.optimizer.run_optimization(max_iter=self.n_iter)

    def _after_init(self, optimizer):
        optimizer.callback_parameter = super()._after_init(optimizer)
        weight = np.eye(self.n_obj)
        for i in range(self.n_obj):
            optimizer.model[i].set_weight_func(weighted_sum)
            optimizer.model[i].set_weight(np.expand_dims(weight[i, :], 0))
        return optimizer.callback_parameter

    def _after_propose(self, optimizer):
        optimizer.callback_parameter = super()._after_propose(optimizer)
        n_batch = self.n_batch
        if n_batch + optimizer.num_acquisitions > optimizer.max_iter:
            n_batch = optimizer.max_iter - optimizer.num_acquisitions
        X = optimizer.suggested_sample

        if X.shape[0] > self.n_batch:
            acquisition_value = np.vstack(optimizer.callback_parameter['acquisition_value'])
            _, optimizer.suggested_sample = findKBest(X, n_batch, acquisition_value)
        return optimizer.callback_parameter

    def _MGD_search(self, x, eval_func, eval_dfunc, bounds, weight):

        # optimization objective, see eq(4)
        def fun(x):
            fx = eval_func(np.atleast_2d(x))
            return fx[0] @ weight

        def jac(x):
            dfx = eval_dfunc(np.atleast_2d(x))
            return (weight @ dfx).squeeze()

        # do optimization using LBFGS
        res = minimize(fun, x, method='L-BFGS-B', jac=jac, bounds=bounds)
        return res.x


    @staticmethod
    def _local_search(x, y, f, eval_func, eval_dfunc, bounds, delta_s):
        # choose reference point z
        f_norm = np.linalg.norm(f)
        s = 2.0 * f / np.sum(f) - 1 - f / f_norm
        s /= np.linalg.norm(s)
        z = y + s * delta_s * np.linalg.norm(f)

        # optimization objective, see eq(4)
        def fun(x):
            fx = eval_func(np.atleast_2d(x))
            return np.linalg.norm(fx - z)

        def jac(x):
            fx = eval_func(np.atleast_2d(x))
            dfx = eval_dfunc(np.atleast_2d(x))
            return (((fx - z) / np.linalg.norm(fx - z)) @ dfx).squeeze()

        # do optimization using LBFGS
        res = minimize(fun, x, method='L-BFGS-B', jac=jac, bounds=bounds)
        return res.x

    def _get_nd_X(self, X, Y):
        np.random.seed(0)

        n_sample = X.shape[0]

        if n_sample >= self.pop_size:
            indices = NonDominatedSorting().do(Y)
            return X[np.concatenate(indices)][:self.pop_size]
        else:
            indices = NonDominatedSorting().do(Y)
            sorted_X = X[np.concatenate(indices)]
            addition_X = lhs(X.shape[1], self.pop_size - n_sample)
            return np.vstack([sorted_X,addition_X])

    def _mutation(self, xs):
        d = np.random.random(xs.shape)
        d /= np.expand_dims(np.linalg.norm(d, axis=1), axis=1)

        delta = np.random.random() * 10.0

        xs = xs + 1.0 / (2 ** delta) * d
        xs = np.clip(xs, self.problem.var_bound[:,0], self.problem.var_bound[:,1])
        return xs

    def _debug_plot(self, xs):
        pf = find_pareto_front(self.problem.get_pareto_front())
        f = self.problem.evaluate(xs)
        ax = None
        ax = plot2D(pf[:,0],pf[:,1],c='gray',marker=',',ax=ax)
        plot2D(f[:, 0], f[:, 1], c='black', marker='x', ax=ax, show=True)

