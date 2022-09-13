import numpy as np
import time
from GPyOpt.methods import BayesianOptimization
from GPyOpt.util.general import best_value, normalize
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.space import Design_space, bounds_to_space
from GPyOpt.core.task.cost import CostModel
from GPyOpt.util.arguments_manager import ArgumentsManager
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.core.bo import BO
import GPyOpt

from revision.multiobjective import MultiObjective
from utils.common import find_pareto_front


try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class MultiObjectiveBayesianOptimization(BayesianOptimization):
    def __init__(self, f, n_obj, domain=None, constraints=None, cost_withGradients=None, model_type='GP', X=None,
                 Y=None, initial_design_numdata=5, initial_design_type='random', acquisition_type='EI',
                 normalize_Y=True, exact_feval=False, acquisition_optimizer_type='lbfgs', model_update_interval=1,
                 evaluator_type='sequential', batch_size=1, num_cores=1, verbosity=False, verbosity_model=False,
                 maximize=False, de_duplication=False, **kwargs
                 ):

        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs
        self.n_obj = n_obj

        # --- Handle the arguments passed via kwargs
        self.problem_config = ArgumentsManager(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)

        # --- CHOOSE objective function
        self.maximize = maximize
        if 'objective_name' in kwargs:
            self.objective_name = kwargs['objective_name']
        else:
            self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = MultiObjective(func=self.f, n_obj=self.n_obj,
                                            num_cores=self.batch_size, objective_name=self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.
        self.model_type = model_type
        self.exact_feval = exact_feval  # note that this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y
        self.n_model = 1
        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print('Using a model defined by the used.')
            elif isinstance(kwargs['model'],list):
                self.n_model = len(kwargs['model'])
                self.model = kwargs['model']
                print(f'{self.n_model} models in BO')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type

        # This states how the discrete variables are handled (exact search or rounding)
        kwargs.update({'model': self.model})
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type,
                                                          **kwargs)  ## more arguments may come here

        # --- CHOOSE acquisition function. If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type

        if 'acquisition' in self.kwargs:
            self.acquisition = kwargs['acquisition'](self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients)
            self.acquisition_type = 'User defined acquisition used.'
            print('Using an acquisition defined by the used.')
        else:
            self.acquisition = self._acquisition_chooser()

        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()
        self.next_size = 0
        # --- Create optimization space
        BO.__init__(self, model=self.model,
                    space=self.space,
                    objective=self.objective,
                    acquisition=self.acquisition,
                    evaluator=self.evaluator,
                    X_init=self.X,
                    Y_init=self.Y,
                    cost=self.cost,
                    normalize_Y=self.normalize_Y,
                    model_update_interval=self.model_update_interval,
                    de_duplication=self.de_duplication)

        self.callback_after_init = kwargs['callback_after_init'] if 'callback_after_init' in kwargs else None
        self.callback_before_fit = kwargs['callback_before_fit'] if 'callback_before_fit' in kwargs else None
        self.callback_after_fit = kwargs['callback_after_fit'] if 'callback_after_fit' in kwargs else None
        self.callback_before_propose = kwargs['callback_before_propose'] if 'callback_before_propose' in kwargs else None
        self.callback_user_define_propose = kwargs['callback_user_define_propose'] if 'callback_user_define_propose' in kwargs else None
        self.callback_after_propose = kwargs['callback_after_propose'] if 'callback_after_propose' in kwargs else None
        self.callback_end_iteration = kwargs['callback_end_iteration'] if 'callback_end_iteration' in kwargs else None
        self.callback_after_finish = kwargs['callback_after_finish'] if 'callback_after_finish' in kwargs else None
        self.callback_parameter = {}

    def run_optimization(self, max_iter=0, max_time=np.inf, eps=1e-8, context=None, verbosity=False,
                         save_models_parameters=True, report_file=None, evaluations_file=None, models_file=None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel)
                    or isinstance(self.model, GPyOpt.models.GPModel_MCMC)
                    or isinstance(self.model, list)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        if self.callback_after_init is not None:
            self.callback_after_init(self)
        # --- Initialize time cost of the evaluations
        while self.max_time > self.cum_time:
            # --- Update model

            if self.callback_before_fit is not None:
                self.callback_before_fit(self)
            try:
                for i in range(self.n_model):
                    self._update_models(i, self.normalization_type)

            except np.linalg.linalg.LinAlgError:
                break
            if self.callback_after_fit is not None:
                self.callback_after_fit(self)
            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            if self.callback_before_propose is not None:
                self.callback_before_propose(self)

            if self.callback_user_define_propose is not None:
                self.callback_user_define_propose(self)
            else:
                self.suggested_sample = self._compute_next_evaluations()

            if self.callback_after_propose is not None:
                self.callback_after_propose(self)

            # --- Augment X
            self.next_size = self.suggested_sample.shape[0]
            self.X = np.vstack((self.X, self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()
            if self.callback_end_iteration is not None:
                self.callback_end_iteration(self)

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += self.next_size

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()
        if self.callback_after_finish is not None:
            self.callback_after_finish(self)
        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)

    @property
    def X_in_model(self):

        return self.model[0].X_in_model


    @property
    def Y_in_model(self):
        ret = []

        for i in range(self.n_model):
            ret.append(self.model[i].Y_in_model)
        ret = np.hstack(ret)

        return ret

    def _compute_results(self):
        self.Y_best, index = find_pareto_front(y=self.Y,return_index=True)
        self.x_opt = self.X[index,:]
        self.fx_opt = self.Y_best

    def _update_model(self, normalization_type='stats'):
        print('MOBO should not use the old _update_model')

    def _update_models(self, idx=0, normalization_type='stats'):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if self.num_acquisitions % self.model_update_interval == 0:
            self.model[idx].updateModel(self.space.unzip_inputs(self.X), self.Y, None, None)

        # Save parameters of the model
        if self.model_parameters_iterations is None:
            self.model_parameters_iterations = self.model[idx].get_model_parameters()
        else:
            self.model_parameters_iterations = np.vstack(
                (self.model_parameters_iterations, self.model[idx].get_model_parameters()))



