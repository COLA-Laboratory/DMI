# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.util.general import get_quantiles
from GPyOpt.core.task.cost import constant_cost_withGradients
import numpy as np
class MultiObjectiveAcquisitionMGD(AcquisitionBase):
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
        self.n_model = len(self.model)
        self.output_dim = 0

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return MultiObjectiveAcquisitionMGD(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m = np.zeros((self.n_model,))
        dmdx = np.zeros((self.n_model,))
        for i in range(self.n_model):
            m[i], _, dmdx[i], _ = self.model[self.output_dim].predict_withGradients(x)

        f_acqu = m[0]
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        m = np.zeros((self.n_model,))
        dmdx = np.zeros((self.n_model,))
        for i in range(self.n_model):
            m[i], _, dmdx[i], _ = self.model[self.output_dim].predict_withGradients(x)
        lamb = np.linalg.norm(dmdx[::-1])

        f_acqu = m[0]
        df_acqu = m * lamb
        return f_acqu, df_acqu
