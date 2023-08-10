from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np
from scipy.stats import nbinom


class NegativeBinomial(DataLikelihood):

    def __init__(self):
        super().__init__("Negative Binomial", 1)
        self.bounds = [(1e-6, 1e8)]

    def negLogLike(self, data, prediction, params):
        zero_offset = 1e-6

        r = params[0]
        return - np.sum(
            -data * np.log(prediction + zero_offset) + data * np.log(r + prediction) + r * np.log(1 + prediction / r))

    def sample(self, prediction, params):
        r = params[0]
        return nbinom.rvs(r, r / (prediction + r)).reshape(prediction.shape)
