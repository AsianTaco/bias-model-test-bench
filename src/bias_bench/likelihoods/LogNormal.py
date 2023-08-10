from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np
import sys

class LogNormal(DataLikelihood):

    def __init__(self):
        super().__init__("LogNormal", 1)
        self.bounds = [(1, 1e8)]

    def negLogLike(self, data, prediction, params):

        sigma = params[0].clip(min=0.01)

        zero_offset = 1e-6

        term1 = np.log(data+zero_offset)
        term2 = np.log(sigma)
        term3 = 0.5 * np.log(2 * np.pi)
        term4 = ((np.log(data+zero_offset) - prediction.clip(min=0.000001))**2) / (2 * sigma**2)

        return -np.sum(term1 + term2 + term3 + term4)


    def sample(self, prediction, params):
        sigma = params[0].clip(min=0.01)
        return np.random.lognormal(mean=prediction.clip(min=0.000001), sigma=sigma)
