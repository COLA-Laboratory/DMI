from GPyOpt.models.gpmodel import GPModel
import GPy
import numpy as np
from utils.common import tchebycheff,safe_divide

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from surrogate.rbfn import RBFNet


class WeightedRBFModel(GPModel):

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000,
                 optimize_restarts=5, sparse=False, num_inducing=10, verbose=True, ARD=False, mean_function=None,
                 weight=None, X_normalizer = None, Y_normalizer = None):

        super().__init__(kernel, noise_var, exact_feval, optimizer, max_iters,
                         optimize_restarts, sparse, num_inducing, verbose, ARD, mean_function)

        self.weight = weight
        self.weight_func = tchebycheff
        self.X_normalizer = X_normalizer
        self.Y_normalizer = Y_normalizer
        self.X_in_model = None
        self.Y_in_model = None

    def set_weight(self, weight):
        self.weight = weight

    def set_weight_func(self,weight_func):
        self.weight_func = weight_func

    def updateModel(self, X_all, Y_all, X_new, Y_new, recreate=True):
        """
        Updates the model with new observations.
        """
        self.X_in_model = X_all if self.X_normalizer is None else self.X_normalizer.do(X_all)

        Y_weighted = self.weight_func(Y_all, self.weight)
        if self.Y_normalizer is not None:
            self.Y_normalizer.fit(Y_weighted)
            self.Y_in_model = self.Y_normalizer.do(Y_weighted)
        else:
            self.Y_in_model = Y_weighted

        if recreate:
            if self.kernel is None:
                self.kernel = self.model.kern
            self.model = None
            self._create_model(self.X_in_model, self.Y_in_model)
        else:
            # TODO adapt _set_XY
            pass
        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts == 1:
                self.model.optimize(optimizer=self.optimizer, max_iters=self.max_iters, messages=False,
                                    ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer,
                                             max_iters=self.max_iters, verbose=self.verbose)

        '''
        begin for debug DGEMO 
        '''

        def constrained_optimization(obj_func, initial_theta, bounds):
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
            return opt_res.x, opt_res.fun

        main_kernel = RBF(length_scale=np.ones(X_all.shape[1]), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))

        self.rbfnet = RBFNet(k=X_all.shape[1]*5)
        self.rbfnet.fit(self.X_in_model, self.Y_in_model.squeeze())

    def predict_skgp(self, X, calc_gradient=False, calc_hessian=False):
        F, dF, hF = None, None, None # mean

        F = self.rbfnet.predict(X).squeeze()

        if not (calc_gradient or calc_hessian):
            out = {'F': F, 'dF': dF, 'hF': hF}
            return out
        dF = self.rbfnet.predictive_gradients(X)
        hF = self.rbfnet.predictive_hessian(X)
        out = {'F': F, 'dF': dF, 'hF': hF}
        return out

    '''
    end for debug DGEMO 
    '''