import GPy
import GPyOpt
import time
import numpy as np
from pymoo.model.problem import Problem as pymooProblem
from pymoo.algorithms.moead import MOEAD as pymooMOEAD
from pymoo.algorithms.nsga2 import NSGA2 as pymooNSGA2
from pymoo.algorithms.nsga3 import NSGA3 as pymooNSGA3
from pymoo.optimize import minimize as pymooMinimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from algorithms.multiobjective_optimization import MultiObjectiveOptimization
from utils.weights import init_weight
from utils.common import findKBest, weighted_sum, find_pareto_front
from utils.lhs import lhs
from utils.weight_associated_samples import WeightAssociatedSamples
from revision.multiobjective_bayesian_optimization import MultiObjectiveBayesianOptimization
from revision.weighted_gpmodel import WeightedGPModel
from revision.multiobjective_EI import MultiObjectiveAcquisitionEI
from utils.normalizer import StandardNormalizer, BoundedNormalizer
from utils.propose_next_batch import DecomposedHVI, DecomposedDist
from GPyOpt.core.task.space import Design_space
from scipy.optimize import minimize
from scipy.linalg import null_space


class DmiEA(MultiObjectiveOptimization):

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
            self.n_gen = 1
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
            return np.vstack(dy)

        def eval_hf(x):
            hy = []
            for i in range(self.n_obj):
                # dy.append(optimizer.model[i].predictive_gradients(x)[0])
                hy.append(optimizer.model[i].predict_skgp(x,calc_hessian=True)['hF'])
            return np.vstack(hy)

        ideal = np.min(optimizer.Y,axis=0)
        nadir = np.max(optimizer.Y,axis=0)

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
        nd_x = pop_x[nd_id]
        nd_y = pop_y[nd_id]
        ea_x = None
        dmi_x = None
        for g in range(self.n_gen):
            print(f'generation {g}/{self.n_gen} of {optimizer.X_in_model.shape[0]}')
            xs = self._mutation(pop_x.copy())
            ys = eval_f(xs)
            new_ideal_point = np.minimum(association.ideal_point, np.min(ys, axis=0))
            if (new_ideal_point != association.ideal_point).any():
                new_ideal_point -= self.eps
            fs = ys - new_ideal_point

            xs_opt = []
            xs_idx = []
            x_opts = []
            print("EA begin")
            if self.ea_type == 'moead':
                ea_problem = EAProblem(n_var=self.n_var, n_obj=self.n_obj, ideal=self.problem.ideal_point, nadir=self.problem.nadir_point,
                                       var_bound=self.problem.var_bound, model=optimizer.model, f=self.problem.evaluate)
                algorithm = pymooMOEAD(sampling=pop_x, ref_dirs=self.weight,repair=False,eliminate_duplicates=False)
                res = pymooMinimize(ea_problem, algorithm, ('n_gen', 100*self.n_var), verbose=False,seed=self.seed)
                x_opts = list(res.X)
                ea_x = res.X
            else:
                # use local search as default
                for i in range(xs.shape[0]):
                    x_opts.append(self._local_search(np.atleast_2d(xs[i]),np.atleast_2d(ys[i]),np.atleast_2d(fs[i]),eval_f,eval_df,self.problem.var_bound,0.3))
            print("dmi begin")
            for x_opt in x_opts:
                x_approximation = self._get_optimization_approximation(x_opt,eval_f(np.atleast_2d(x_opt)),
                                                               eval_df(np.atleast_2d(x_opt)),eval_hf(np.atleast_2d(x_opt)),
                                                               self.problem.var_bound)

                # x_approximation = np.atleast_2d(x_opt) # dummy

                xs_opt.append(x_approximation)
                xs_idx.extend([i + current_label] * len(x_approximation))
            print("association begin")
            current_label += len(xs_opt)
            xs_opt = np.vstack(xs_opt)
            xs_idx = np.array(xs_idx)
            new_ideal_point = np.minimum(new_ideal_point,association.ideal_point)
            ys_opt = eval_f(xs_opt)
            new_ideal_point = np.minimum(np.atleast_2d(np.min(ys_opt, axis=0)), new_ideal_point)
            association.set_ideal_point(new_ideal_point)
            dmi_x = xs_opt
            association.insert(xs_opt,ys_opt,xs_idx)
            pop_x = association.get(self.pop_size)
            pop_y = eval_f(pop_x)
        print("selection begin")
        pop_x, pop_y, pop_dist, pop_idx = association.get_all()
        if self.approximation:
            pop_x, pop_y, pop_dist, pop_idx = association.sparse_approximation()
        self.selection.set_nadir_point(np.ones((1,self.n_obj))*1.1)
        if self.selection_type == 'DHVI':
            optimizer.suggested_sample = self.selection.select(self.n_batch, pop_x, find_pareto_front(optimizer.Y), optimizer.model, pop_idx)
        elif self.selection_type == 'MOEAD':
            self.selection.set_ideal_point(self.problem.get_ideal_point())
            _, idx_nd = find_pareto_front(optimizer.Y,return_index=True)
            _, idx_nd_propose = find_pareto_front(pop_y,return_index=True)
            optimizer.suggested_sample = self.selection.select(self.n_batch, pop_x[idx_nd_propose,:], optimizer.X[idx_nd,:], optimizer.Y[idx_nd,:], optimizer.model, self.weight)

        np.savetxt(f"tmp/data/ea_{len(optimizer.Y)}_x.txt",np.vstack(ea_x))
        np.savetxt(f"tmp/data/dmi_{len(optimizer.Y)}_x.txt", np.vstack(dmi_x))
        np.savetxt(f"tmp/data/pop_{len(optimizer.Y)}_x.txt", np.vstack(pop_x))
        np.savetxt(f"tmp/data/evaluated_{len(optimizer.Y)}_x.txt", np.vstack(optimizer.X))
        np.savetxt(f"tmp/data/selected_{len(optimizer.Y)}_x.txt", np.vstack(optimizer.suggested_sample))

        np.savetxt(f"tmp/data/ea_{len(optimizer.Y)}.txt",self.problem.evaluate(np.vstack(ea_x)))
        np.savetxt(f"tmp/data/dmi_{len(optimizer.Y)}.txt", self.problem.evaluate(np.vstack(dmi_x)))
        np.savetxt(f"tmp/data/pop_{len(optimizer.Y)}.txt", self.problem.evaluate(np.vstack(pop_x)))
        np.savetxt(f"tmp/data/evaluated_{len(optimizer.Y)}.txt", self.problem.evaluate(np.vstack(optimizer.X)))
        np.savetxt(f"tmp/data/selected_{len(optimizer.Y)}.txt", self.problem.evaluate(np.vstack(optimizer.suggested_sample)))
        print(f"Output end {len(optimizer.Y)}")
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

    @staticmethod
    def _local_search(x, y, f, eval_func, eval_dfunc, bounds, delta_s):
        # choose reference point z
        f_norm = np.linalg.norm(f)
        s = 2.0 * f / np.sum(f) - 1 - f / f_norm
        s /= np.linalg.norm(s)
        z = y + s * delta_s * np.linalg.norm(f)
        def fun(x):
            fx = eval_func(np.atleast_2d(x))
            return np.linalg.norm(fx - z)
        def jac(x):
            fx = eval_func(np.atleast_2d(x))
            dfx = eval_dfunc(np.atleast_2d(x))
            return (((fx - z) / np.linalg.norm(fx - z)) @ dfx).squeeze()
        res = minimize(fun, x, method='L-BFGS-B', jac=jac, bounds=bounds)
        return res.x

    @staticmethod
    def _get_kkt_dual_variables(f, g, df, dg):
        n_obj = f.shape[1]
        n_const = len(g) if g is not None else 0
        if n_const > 0:  # when there are active constraints
            def fun(x, n_obj=n_obj, df=df, dg=dg):
                alpha, beta = x[:n_obj], x[n_obj:]
                objective = alpha @ df + beta @ dg
                return 0.5 * objective @ objective

            def jac(x, n_obj=n_obj, df=df, dg=dg):
                alpha, beta = x[:n_obj], x[n_obj:]
                objective = alpha @ df + beta @ dg
                return np.vstack([df, dg]) @ objective

            const = {'type': 'eq',
                     'fun': lambda x, n_obj=n_obj: np.sum(x[:n_obj]) - 1.0,
                     'jac': lambda x, n_obj=n_obj: np.concatenate([np.ones(n_obj), np.zeros_like(x[n_obj:])])}
        else:
            def fun(x, df=df):
                objective = x @ df
                return 0.5 * objective @ objective
            def jac(x, df=df):
                objective = x @ df
                return df @ objective
            const = {'type': 'eq',
                     'fun': lambda x: np.sum(x) - 1.0,
                     'jac': np.ones_like}
        bounds = np.array([[0.0, np.inf]] * (n_obj + n_const))
        alpha_init = np.random.random(f.shape[1])
        alpha_init /= np.sum(alpha_init)
        beta_init = np.zeros(n_const)  # zero initialization for beta
        x_init = np.concatenate([alpha_init, beta_init])
        res = minimize(fun, x_init, method='SLSQP', jac=jac, bounds=bounds, constraints=const)
        x_opt = res.x
        alpha_opt, beta_opt = x_opt[:n_obj], x_opt[n_obj:]
        return alpha_opt, beta_opt

    @staticmethod
    def _get_optimization_approximation(x_opt, f_opt, df_opt, hf_opt, bounds):
        eps = 1e-8
        n_grid_sample = 100

        n_var = len(x_opt)
        n_obj = f_opt.shape[1]
        const_idx = np.where(np.logical_or(bounds[:,1] - eps < x_opt, bounds[:,0] + eps > x_opt))[0]
        n_const = len(const_idx)

        upper_const_idx = np.where(bounds[:,1] - eps < x_opt)[0]
        lower_const_idx = np.where(bounds[:,0] + eps > x_opt)[0]
        g = np.zeros(n_const)
        dg = np.zeros((n_const, n_var))
        for i, idx in enumerate(const_idx):
            constraint = np.zeros(n_var)
            if idx in upper_const_idx:
                constraint[idx] = 1 # upper active
            elif idx in lower_const_idx:
                constraint[idx] = -1 # lower active
            dg[i] = constraint
        hg = np.zeros((n_const, n_var, n_var))

        alpha, beta = DmiEA._get_kkt_dual_variables(f_opt, g, df_opt, dg)

        if n_const > 0:
            h = hf_opt.T @ alpha + hg.T @ beta
        else:
            h = hf_opt.T @ alpha

        alpha_const = np.concatenate([np.ones(n_obj), np.zeros(n_const + n_var)])
        if n_const > 0:
            comp_slack_const = np.column_stack([np.zeros((n_const, n_obj + n_const)), dg])
            DxHx = np.vstack([alpha_const, comp_slack_const, np.column_stack([df_opt.T, dg.T, h])])
        else:
            DxHx = np.vstack([alpha_const, np.column_stack([df_opt.T, h])])
        directions = null_space(DxHx)

        eps = 1e-8
        directions[np.abs(directions) < eps] = 0.0
        d_alpha, d_beta, d_x = directions[:n_obj], directions[n_obj:n_obj + n_const], directions[-n_var:]
        if np.linalg.norm(d_x) < eps:  # direction is a zero vector
            return x_opt

        direction_dim = d_x.shape[1]

        if direction_dim > n_obj - 1:
            indices = np.random.choice(np.arange(direction_dim), n_obj - 1)
            while np.linalg.norm(d_x[:, indices]) < eps:
                indices = np.random.choice(np.arange(direction_dim), n_obj - 1)
            d_x = d_x[:, indices]
        elif direction_dim < n_obj - 1:
            return x_opt

        d_x /= np.linalg.norm(d_x)
        bound_scale = np.expand_dims(bounds[:,1] - bounds[:,0], axis=1)
        d_x *= bound_scale
        loop_count = 0

        x_samples = np.array([x_opt])
        np.random.seed(0)
        while len(x_samples) < n_grid_sample:
            # compute expanded samples
            curr_dx_samples = np.sum(np.expand_dims(d_x, axis=0) * np.random.random((n_grid_sample, 1, n_obj - 1)),
                                     axis=-1)
            curr_x_samples = np.expand_dims(x_opt, axis=0) + curr_dx_samples
            # check validity of samples
            valid_idx = np.where(np.logical_and((curr_x_samples <= bounds[:,1]).all(axis=1),
                                                (curr_x_samples >= bounds[:,0]).all(axis=1)))[0]
            x_samples = np.vstack([x_samples, curr_x_samples[valid_idx]])
            loop_count += 1
            if loop_count > 10:
                break
        x_samples = x_samples[:n_grid_sample]
        return x_samples

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


class EAProblem(pymooProblem):
    def __init__(self, n_var, n_obj,ideal, nadir, var_bound, model,f):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=var_bound[:,0], xu=var_bound[:,1], evaluation_of="auto")
        self.ideal = ideal
        self.nadir = nadir
        self.model = model
        self.f = f

    def _evaluate(self, x, out, *args, **kwargs):

        y = []
        for i in range(self.n_obj):
            y.append(np.expand_dims(self.model[i].predict_skgp(x)['F'], -1))
        out["F"] =  np.atleast_2d(( np.squeeze(np.hstack(y))  - self.ideal ))
        '''
        y = self.f(x)
        out["F"] = np.atleast_2d((y - self.ideal))
        '''
        return out
