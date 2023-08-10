from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np


class Gaussian(DataLikelihood):

    def __init__(self):
        super().__init__("Gaussian", 1)
        self.bounds = [(1, 1e8)]

    def negLogLike(self, data, prediction, params):

        std_dev = params[0]

        res = 0.5 * np.log(2 * np.pi * std_dev**2) + \
                (data - prediction)**2 / (2 * std_dev**2)

        return -np.sum(res)

    def sample(self, prediction, params):
        return np.random.normal(prediction, *params)
