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


class WeightedGPModel(GPModel):

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

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-3), np.sqrt(1e3))) * \
                main_kernel + ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))

        self.skgp = GaussianProcessRegressor(kernel=kernel, optimizer=constrained_optimization)
        self.skgp.fit(self.X_in_model, self.Y_in_model.squeeze())

        '''
        end for debug DGEMO 
        '''

    def predict_skgp(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = None, None, None # mean
        S, dS, hS = None, None, None
        K = self.skgp.kernel_(X, self.skgp.X_train_)  # K: shape (N, N_train)
        y_mean = K.dot(self.skgp.alpha_)
        F = y_mean  # y_mean: shape (N,)
        if std:
            if self.skgp._K_inv is None:
                L_inv = solve_triangular(self.skgp.L_.T,np.eye(self.skgp.L_.shape[0]))
                self.skgp._K_inv = L_inv.dot(L_inv.T)

            y_var = self.skgp.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i",np.dot(K, self.skgp._K_inv), K)

            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                y_var[y_var_negative] = 0.0

            y_std = np.sqrt(y_var)

            S=y_std # y_std: shape (N,)

        if not (calc_gradient or calc_hessian):
            out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
            return out

        ell = np.exp(self.skgp.kernel_.theta[1:-1])  # ell: shape (n_var,)
        sf2 = np.exp(self.skgp.kernel_.theta[0])  # sf2: shape (1,)
        d = np.expand_dims(cdist(X / ell, self.skgp.X_train_ / ell), 2)  # d: shape (N, N_train, 1)
        X_, X_train_ = np.expand_dims(X, 1), np.expand_dims(self.skgp.X_train_, 0)
        dd_N = X_ - X_train_  # numerator
        dd_D = d * ell ** 2  # denominator
        dd = safe_divide(dd_N, dd_D)  # dd: shape (N, N_train, n_var)

        if calc_gradient or calc_hessian:

            dK = -sf2 * np.exp(-0.5 * d ** 2) * d * dd

            dK_T = dK.transpose(0, 2, 1)  # dK: shape (N, N_train, n_var), dK_T: shape (N, n_var, N_train)

        if calc_gradient:
            dy_mean = dK_T @ self.skgp.alpha_  # gp.alpha_: shape (N_train,)
            dF=dy_mean  # dy_mean: shape (N, n_var)

            # TODO: check
            if std:
                K = np.expand_dims(K, 1)  # K: shape (N, 1, N_train)
                K_Ki = K @ self.skgp._K_inv  # gp._K_inv: shape (N_train, N_train), K_Ki: shape (N, 1, N_train)
                dK_Ki = dK_T @ self.skgp._K_inv  # dK_Ki: shape (N, n_var, N_train)

                dy_var = -np.sum(dK_Ki * K + K_Ki * dK_T, axis=2)  # dy_var: shape (N, n_var)
                dy_std = 0.5 * safe_divide(dy_var, y_std)  # dy_std: shape (N, n_var)
                dS=dy_std

        if calc_hessian:
            d = np.expand_dims(d, 3)  # d: shape (N, N_train, 1, 1)
            dd = np.expand_dims(dd, 2)  # dd: shape (N, N_train, 1, n_var)
            hd_N = d * np.expand_dims(np.eye(len(ell)), (0, 1)) - np.expand_dims(X_ - X_train_, 3) * dd  # numerator
            hd_D = d ** 2 * np.expand_dims(ell ** 2, (0, 1, 3))  # denominator
            hd = safe_divide(hd_N, hd_D)  # hd: shape (N, N_train, n_var, n_var)


            hK = -sf2 * np.exp(-0.5 * d ** 2) * ((1 - d ** 2) * dd ** 2 + d * hd)

            hK_T = hK.transpose(0, 2, 3,
                                1)  # hK: shape (N, N_train, n_var, n_var), hK_T: shape (N, n_var, n_var, N_train)

            hy_mean = hK_T @ self.skgp.alpha_  # hy_mean: shape (N, n_var, n_var)
            hF=hy_mean

            # TODO: check
            if std:
                K = np.expand_dims(K, 2)  # K: shape (N, 1, 1, N_train)
                dK = np.expand_dims(dK_T, 2)  # dK: shape (N, n_var, 1, N_train)
                dK_Ki = np.expand_dims(dK_Ki, 2)  # dK_Ki: shape (N, n_var, 1, N_train)
                hK_Ki = hK_T @ self.skgp._K_inv  # hK_Ki: shape (N, n_var, n_var, N_train)

                hy_var = -np.sum(hK_Ki * K + 2 * dK_Ki * dK + K_Ki * hK_T, axis=3)  # hy_var: shape (N, n_var, n_var)
                hy_std = 0.5 * safe_divide(hy_var * y_std - dy_var * dy_std, y_var)  # hy_std: shape (N, n_var, n_var)
                hS=hy_std
        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out