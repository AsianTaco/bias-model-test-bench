from bias_bench.likelihoods.DataLikelihood import DataLikelihood
import numpy as np
from scipy.stats import skewnorm, norm


class LogSkewNormal(DataLikelihood):

    def __init__(self):
        super().__init__("LogSkewNormal", 2)
        self.bounds = [(1, 1e8), (-1e4,1e4)]

    def negLogLike(self, data, prediction, params):

        sigma = params[0].clip(min=0.001)
        alpha = params[1]

        zero_offset = 1e-6

        pred_clipped = prediction.clip(min=1e-6)
        fac = 2/sigma
        PDF = norm.pdf(np.log(data + zero_offset),np.log(pred_clipped),sigma)
        CDF = norm.cdf(alpha * np.log(data + zero_offset),
                    alpha * np.log(pred_clipped), sigma)

        return -np.sum(np.log(fac*PDF*CDF))

    def sample(self, prediction, params):
        sigma = params[0].clip(min=0.001)
        alpha = params[1]
        return np.exp(skewnorm.rvs(alpha, loc = prediction, scale = sigma)).reshape(prediction.shape)
