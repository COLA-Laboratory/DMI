# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.util.general import get_quantiles
from GPyOpt.core.task.cost import constant_cost_withGradients

class MultiObjectiveAcquisitionEI(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """


    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        self.model = model
        self.space = space
        self.optimizer = optimizer

        self.analytical_gradient_acq = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction  # flag from the model to test if gradients are available
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients
        self.jitter = jitter
        self.n_model = list(self.model)
        self.output_dim = 0

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return MultiObjectiveAcquisitionEI(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def set_output_dim(self,dim):
        self.output_dim = dim

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model[self.output_dim].predict(x)
        fmin = self.model[self.output_dim].get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model[self.output_dim].get_fmin()
        m, s, dmdx, dsdx = self.model[self.output_dim].predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu
