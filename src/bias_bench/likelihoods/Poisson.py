from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np


class Poisson(DataLikelihood):

    def __init__(self, params):
        super().__init__("Poisson", None)

    def negLogLike(self, data, prediction):
        # TODO: Deal with negative or zero count predictions
        return np.sum(prediction) - np.sum(data * np.log(prediction))
