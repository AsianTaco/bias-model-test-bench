import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

from bias_bench.utils import bias_bench_print
from bias_bench.benchmark_models.BenchmarkModel import BenchmarkModel


class LinearBias(BenchmarkModel):

    def __init__(self):
        super().__init__("Linear Bias", 2)
        self.bounds = [(1., 1e8), (1., 1e8)]

    def get_mean_ngal(self, rho, nmean, b1):
        """
        Using the linear bias model, compute the mean number of
        galaxies expected for a given density.

        Incoming density field is expected to be 1D, i.e., the flattened 3D array.

        Parameters
        ----------
        rho : ndarray
            The 3D dm density field
        nmean : float
            ...
        b1 : float
            ...
        Returns
        -------
        ngal_mean : ndarray
            Expected number of galaxies for the given densities
        """

        # TODO: refactor this
        #nmean_bounds = self.bounds[0]
        #if nmean < nmean_bounds[0] or nmean > nmean_bounds[1]:
        #    return np.inf
        #
        #b1_bounds = self.bounds[1]
        #if b1 < b1_bounds[0] or b1 > b1_bounds[1]:
        #    return np.inf

        #LIN model:
        ngal_mean = nmean * (1 + b1 * rho)
        return ngal_mean

    def predict(self, delta, popt):
        # TODO: implement bounds explicitly

        # Predict expected value ngal from delta_dm (and model params popt).
        ngal = self.get_mean_ngal(delta.flatten(), *popt)

        if np.any(np.isnan(ngal)):
            return np.NaN

        return ngal.reshape(delta.shape)
