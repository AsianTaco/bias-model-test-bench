from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np


class Poisson(DataLikelihood):

    def __init__(self):
        super().__init__("Poisson", 0)

    def negLogLike(self, data, prediction, params):
        zero_offset = 1e-6

        if np.any(prediction + zero_offset < 0):
            return np.inf

        if np.any(np.isnan(prediction)):
            return np.inf

        res = np.sum(prediction + zero_offset) - np.sum(data * np.log(prediction + zero_offset))
        return res

    def sample(self, prediction, params):
        return np.random.poisson(prediction)
